"""
Greedy evaluation entry point.

Loads a checkpoint produced by `interaction.train`, runs N episodes
greedily, and dumps a JSON summary plus a per-episode CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from envs import make_env
from models import (
    Critic, DiscountHead, DiscreteActor, FactoredRSSM, MLPDecoder, MLPEncoder,
    OpponentModel, RewardHead,
)
from .rollout import LatentPolicy, collect_episode


_LOG = logging.getLogger("foedreamer.eval")


def load_modules(checkpoint_path: str, env, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    rssm = FactoredRSSM(
        action_size=action_dim,
        z_dim=model_cfg["z_dim"],
        u_dim=model_cfg["u_dim"],
        deter_size=model_cfg["rssm_node_size"],
        node_size=model_cfg["hidden_dim"],
        embedding_size=model_cfg["embed_dim"],
        device=device,
    ).to(device)
    encoder = MLPEncoder(
        obs_dim, model_cfg["embed_dim"], model_cfg["hidden_dim"], model_cfg["num_layers"],
    ).to(device)
    feat_dim = rssm.feature_size()
    ctrl_dim = rssm.controllable_size()
    decoder = MLPDecoder(feat_dim, obs_dim, model_cfg["hidden_dim"], model_cfg["num_layers"]).to(device)
    reward_head = RewardHead(ctrl_dim, model_cfg["hidden_dim"]).to(device)
    discount_head = DiscountHead(feat_dim, model_cfg["hidden_dim"]).to(device)
    actor = DiscreteActor(feat_dim, action_dim, model_cfg["actor_hidden"]).to(device)
    critic = Critic(feat_dim, model_cfg["critic_hidden"]).to(device)
    opponent = OpponentModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        tau_dim=model_cfg["tau_dim"],
        K_window=model_cfg["K_window"],
        hidden_dim=model_cfg["hidden_dim"],
        num_actions=8,
        num_targets=env.num_hosts,
    ).to(device)

    rssm.load_state_dict(ckpt["rssm"])
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    reward_head.load_state_dict(ckpt["reward_head"])
    discount_head.load_state_dict(ckpt["discount_head"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    opponent.load_state_dict(ckpt["opponent"])

    rssm.eval(); encoder.eval(); actor.eval()
    return {
        "rssm": rssm, "encoder": encoder, "decoder": decoder,
        "reward_head": reward_head, "discount_head": discount_head,
        "actor": actor, "critic": critic, "opponent": opponent,
        "action_dim": action_dim,
        "env_step": ckpt.get("env_step", -1),
        "ckpt_config": cfg,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scenario", type=str, default="envs/scenarios/scenario_small.yaml")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results/eval")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.scenario)
    modules = load_modules(args.checkpoint, env, device)

    policy = LatentPolicy(
        modules["rssm"], modules["encoder"], modules["actor"],
        action_dim=modules["action_dim"], device=device,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    returns = []
    successes = []
    for i in range(args.n_episodes):
        ep, stats = collect_episode(
            env, policy, action_dim=modules["action_dim"],
            epsilon=0.0, rng=rng, seed=args.seed + i,
        )
        rows.append({
            "episode": i,
            "return": stats.episode_return,
            "length": stats.episode_length,
            "success": int(stats.success),
        })
        returns.append(stats.episode_return)
        successes.append(int(stats.success))
        _LOG.info("ep=%d  return=%.2f  len=%d  success=%d",
                  i, stats.episode_return, stats.episode_length, int(stats.success))

    summary = {
        "checkpoint": args.checkpoint,
        "scenario": args.scenario,
        "n_episodes": args.n_episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "env_step": modules["env_step"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _LOG.info("summary: %s", summary)


if __name__ == "__main__":
    main()
