"""
FOE-Dreamer training loop.

Builds the factored world model, the sliding-window opponent encoder,
the actor and critic, and runs the prefill / collect+train / eval+ckpt
loop. Hyperparameters match Appendix C of the manuscript.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
import yaml

from envs import make_env
from models import (
    Critic, DiscountHead, DiscreteActor, FactoredRSSM, MLPDecoder, MLPEncoder,
    OpponentModel, RewardHead, actor_critic_loss, hard_update,
    opponent_prediction_loss, soft_update, trainable_params, world_model_loss,
)
from .replay import Episode, EpisodicReplay
from .rollout import LatentPolicy, RandomPolicy, collect_episode, evaluate_policy


_LOG = logging.getLogger("foedreamer.train")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelCfg:
    z_dim: int = 128
    u_dim: int = 128
    rssm_node_size: int = 256
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    tau_dim: int = 64
    K_window: int = 10
    predict_obs: bool = False
    actor_hidden: int = 256
    critic_hidden: int = 256


@dataclass
class TrainCfg:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    episodes_per_iteration: int = 8
    budget_days: int = 3
    total_steps: int = 200_000
    prefill_steps: int = 5_000
    batch_size: int = 64
    seq_len: int = 50
    train_iters: int = 1
    horizon: int = 30
    alpha_policy: float = 0.5
    beta_opponent: float = 0.3
    beta_z: float = 1.0
    beta_u: float = 0.5
    free_nats: float = 1.0
    kl_balance: float = 0.8
    learning_rate: float = 3e-4
    grad_clip: float = 100.0
    target_tau: float = 0.005
    replay_capacity: int = 100_000
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 1e-3
    eval_every_steps: int = 10_000
    eval_episodes: int = 10
    checkpoint_every_steps: int = 25_000
    log_every_steps: int = 1_000
    epsilon_start: float = 0.4
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    num_seeds: int = 5


@dataclass
class EnvCfg:
    scenario_path: str = "envs/scenarios/scenario_small.yaml"
    polling_interval_s: float = 5.0
    episode_length: int = 100


@dataclass
class Cfg:
    env: EnvCfg = field(default_factory=EnvCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    output_dir: str = "results/foe_dreamer"
    run_name: str = "default"


def load_cfg(path: Optional[str]) -> Cfg:
    if path is None:
        return Cfg()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Cfg()
    if "env" in raw:
        cfg.env = EnvCfg(**raw["env"])
    if "model" in raw:
        cfg.model = ModelCfg(**raw["model"])
    if "train" in raw:
        cfg.train = TrainCfg(**raw["train"])
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)
    cfg.run_name = raw.get("run_name", cfg.run_name)
    return cfg


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_modules(env, cfg: Cfg, device: torch.device):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    rssm = FactoredRSSM(
        action_size=action_dim,
        z_dim=cfg.model.z_dim,
        u_dim=cfg.model.u_dim,
        deter_size=cfg.model.rssm_node_size,
        node_size=cfg.model.hidden_dim,
        embedding_size=cfg.model.embed_dim,
        device=device,
    ).to(device)

    encoder = MLPEncoder(obs_dim, cfg.model.embed_dim, cfg.model.hidden_dim, cfg.model.num_layers).to(device)
    feat_dim = rssm.feature_size()
    ctrl_dim = rssm.controllable_size()
    decoder = MLPDecoder(feat_dim, obs_dim, cfg.model.hidden_dim, cfg.model.num_layers).to(device)
    reward_head = RewardHead(ctrl_dim, cfg.model.hidden_dim).to(device)
    discount_head = DiscountHead(feat_dim, cfg.model.hidden_dim).to(device)
    actor = DiscreteActor(feat_dim, action_dim, cfg.model.actor_hidden).to(device)
    critic = Critic(feat_dim, cfg.model.critic_hidden).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    target_critic.requires_grad_(False)

    opponent = OpponentModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        tau_dim=cfg.model.tau_dim,
        K_window=cfg.model.K_window,
        hidden_dim=cfg.model.hidden_dim,
        num_actions=8,        # opponent action vocabulary size
        num_targets=env.num_hosts,
    ).to(device)

    return {
        "rssm": rssm, "encoder": encoder, "decoder": decoder,
        "reward_head": reward_head, "discount_head": discount_head,
        "actor": actor, "critic": critic, "target_critic": target_critic,
        "opponent": opponent,
        "obs_dim": obs_dim, "action_dim": action_dim, "feat_dim": feat_dim,
    }


def build_optimizer(modules: Dict[str, Any], cfg: Cfg):
    params = trainable_params([
        modules["rssm"], modules["encoder"], modules["decoder"],
        modules["reward_head"], modules["discount_head"],
        modules["actor"], modules["critic"], modules["opponent"],
    ])
    return optim.Adam(params, lr=cfg.train.learning_rate)


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def train_step(modules, optimizer, replay, cfg: Cfg, device: torch.device):
    rssm = modules["rssm"]
    encoder = modules["encoder"]
    decoder = modules["decoder"]
    reward_head = modules["reward_head"]
    discount_head = modules["discount_head"]
    actor = modules["actor"]
    critic = modules["critic"]
    target_critic = modules["target_critic"]
    opponent = modules["opponent"]

    info_keys = ["attacker_action", "attacker_target"]
    batch = replay.sample(cfg.train.batch_size, info_keys=info_keys)

    obs = batch["obs"]
    action = batch["action"]
    reward = batch["reward"]
    nonterm = batch["nonterm"]
    nonterm_3d = nonterm.unsqueeze(-1)

    T, B, _ = obs.shape
    obs_embed = encoder(obs.reshape(T * B, -1)).reshape(T, B, -1)
    init_state = rssm.init_state(B, device)
    prior, posterior = rssm.rollout_observation(T, obs_embed, action, nonterm_3d, init_state)

    wm_total, wm_log = world_model_loss(
        rssm, decoder, reward_head, discount_head,
        prior, posterior, obs, reward, nonterm,
        beta_z=cfg.train.beta_z,
        beta_u=cfg.train.beta_u,
        free_nats=cfg.train.free_nats,
        kl_balance=cfg.train.kl_balance,
    )

    # Opponent prediction over the sliding window.
    opp_total, opp_log = opponent_prediction_loss(
        opponent, obs, action,
        batch["attacker_action"], batch["attacker_target"],
        K_window=cfg.model.K_window,
    )

    # Actor-critic over imagined rollouts.
    flat_post = rssm.seq_to_batch(posterior, B, T)
    actor_l, critic_l, ac_log = actor_critic_loss(
        actor, critic, target_critic, rssm, reward_head, discount_head,
        starting_state=flat_post,
        horizon=cfg.train.horizon,
        gamma=cfg.train.gamma,
        lam=cfg.train.lam,
        entropy_coef=cfg.train.entropy_coef,
    )

    total = (
        wm_total
        + cfg.train.beta_opponent * opp_total
        + cfg.train.alpha_policy * actor_l
        + critic_l
    )

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        list(trainable_params([
            rssm, encoder, decoder, reward_head, discount_head,
            actor, critic, opponent,
        ])),
        cfg.train.grad_clip,
    )
    optimizer.step()
    soft_update(target_critic, critic, cfg.train.target_tau)

    return {**wm_log, **opp_log, **ac_log, "loss/total": float(total.detach())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.total_steps is not None:
        cfg.train.total_steps = args.total_steps
    if args.scenario is not None:
        cfg.env.scenario_path = args.scenario

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump({"env": asdict(cfg.env), "model": asdict(cfg.model),
                   "train": asdict(cfg.train),
                   "run_name": cfg.run_name, "output_dir": cfg.output_dir}, f, indent=2)

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    env = make_env(cfg.env.scenario_path)
    eval_env = make_env(cfg.env.scenario_path)

    modules = build_modules(env, cfg, device)
    optimizer = build_optimizer(modules, cfg)

    replay = EpisodicReplay(
        capacity_episodes=max(64, cfg.train.replay_capacity // max(1, cfg.env.episode_length)),
        seq_len=cfg.train.seq_len,
        obs_dim=modules["obs_dim"],
        action_dim=modules["action_dim"],
        device=device,
    )

    rng = np.random.default_rng(cfg.train.seed)
    random_policy = RandomPolicy(action_dim=modules["action_dim"], seed=cfg.train.seed)
    env_steps = 0
    while env_steps < cfg.train.prefill_steps:
        ep, stats = collect_episode(
            env, random_policy, action_dim=modules["action_dim"],
            epsilon=0.0, rng=rng,
        )
        replay.add_episode(ep)
        env_steps += stats.episode_length
    _LOG.info("prefill done: %d steps in buffer", env_steps)

    csv_file = open(out_dir / "metrics.csv", "w", newline="")
    csv_writer: Optional[csv.DictWriter] = None

    train_step_idx = 0
    last_train_env_step = 0
    last_eval_env_step = 0
    last_ckpt_env_step = 0
    while env_steps < cfg.train.total_steps:
        frac = min(1.0, env_steps / max(1, cfg.train.epsilon_decay_steps))
        eps = cfg.train.epsilon_start + frac * (cfg.train.epsilon_end - cfg.train.epsilon_start)

        latent_policy = LatentPolicy(
            modules["rssm"], modules["encoder"], modules["actor"],
            action_dim=modules["action_dim"], device=device,
        )
        ep, stats = collect_episode(
            env, latent_policy, action_dim=modules["action_dim"],
            epsilon=eps, rng=rng,
        )
        replay.add_episode(ep)
        env_steps += stats.episode_length

        if env_steps - last_train_env_step >= cfg.train.train_iters * cfg.env.episode_length:
            for _ in range(cfg.train.train_iters):
                if not replay.has_min_steps(cfg.train.batch_size * cfg.train.seq_len):
                    break
                log = train_step(modules, optimizer, replay, cfg, device)
                train_step_idx += 1
                if train_step_idx % cfg.train.log_every_steps == 0:
                    row = {
                        "env_step": env_steps,
                        "train_step": train_step_idx,
                        "epsilon": eps,
                        "ep_return": stats.episode_return,
                        "ep_len": stats.episode_length,
                        **replay.stats(),
                        **log,
                    }
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    csv_file.flush()
                    _LOG.info("step=%d env=%d wm=%.3f foe=%.3f ac=%.3f",
                              train_step_idx, env_steps,
                              log.get("wm/total", 0.0),
                              log.get("foe/total", 0.0),
                              log.get("ac/actor_obj", 0.0))
            last_train_env_step = env_steps

        if env_steps - last_eval_env_step >= cfg.train.eval_every_steps:
            eval_policy = LatentPolicy(
                modules["rssm"], modules["encoder"], modules["actor"],
                action_dim=modules["action_dim"], device=device,
            )
            eval_log = evaluate_policy(
                eval_env, eval_policy,
                action_dim=modules["action_dim"],
                n_episodes=cfg.train.eval_episodes,
                seed=cfg.train.seed + env_steps,
            )
            _LOG.info("eval @ env=%d return=%.2f success=%.2f",
                      env_steps, eval_log["eval/return_mean"],
                      eval_log["eval/success_rate"])
            with open(out_dir / "eval.jsonl", "a") as f:
                f.write(json.dumps({"env_step": env_steps, **eval_log}) + "\n")
            last_eval_env_step = env_steps

        if env_steps - last_ckpt_env_step >= cfg.train.checkpoint_every_steps:
            ckpt_path = out_dir / f"ckpt_step{env_steps:09d}.pt"
            torch.save({
                "rssm": modules["rssm"].state_dict(),
                "encoder": modules["encoder"].state_dict(),
                "decoder": modules["decoder"].state_dict(),
                "reward_head": modules["reward_head"].state_dict(),
                "discount_head": modules["discount_head"].state_dict(),
                "actor": modules["actor"].state_dict(),
                "critic": modules["critic"].state_dict(),
                "target_critic": modules["target_critic"].state_dict(),
                "opponent": modules["opponent"].state_dict(),
                "env_step": env_steps,
                "config": {"env": asdict(cfg.env), "model": asdict(cfg.model),
                           "train": asdict(cfg.train)},
            }, ckpt_path)
            _LOG.info("checkpoint saved: %s", ckpt_path)
            last_ckpt_env_step = env_steps

    csv_file.close()
    _LOG.info("training complete: %d env steps, %d gradient steps",
              env_steps, train_step_idx)


if __name__ == "__main__":
    main()
