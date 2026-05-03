"""
Episode rollout utilities.

Drives a Gym-API env with a torch policy, packing each step into the
episodic replay buffer's record format. Supports:

  - one-hot action representation (used by FOE-Dreamer's discrete actor)
  - epsilon-greedy exploration schedules
  - per-step logging hooks
  - opponent-action / opponent-target label collection from `info`
    (so the foe head's auxiliary loss can be trained from real
    interactions without an extra labeling pass)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .replay import Episode


@dataclass
class RolloutStats:
    episode_return: float
    episode_length: int
    success: bool
    info_summary: Dict[str, Any]


def one_hot(idx: int, dim: int) -> np.ndarray:
    a = np.zeros(dim, dtype=np.float32)
    a[idx] = 1.0
    return a


def collect_episode(
    env,
    policy_fn: Callable[[np.ndarray], int],
    *,
    action_dim: int,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[Episode, RolloutStats]:
    """Run one full episode and return the recorded trajectory + stats."""
    rng = rng or np.random.default_rng()
    obs, info = env.reset(seed=seed)
    episode = Episode()
    total_reward = 0.0
    success = False

    while True:
        if epsilon > 0.0 and rng.random() < epsilon:
            action_idx = int(rng.integers(0, action_dim))
        else:
            action_idx = int(policy_fn(obs))
        action_vec = one_hot(action_idx, action_dim)

        next_obs, reward, done, truncated, info = env.step(action_idx)

        # Carry per-step opponent labels for the foe head's auxiliary
        # loss. attacker_action and attacker_target are scalar ints
        # emitted by the env's step; def_host and def_action are the
        # defender's choice this tick.
        info_payload: Dict[str, Any] = {}
        for k in ("attacker_action", "attacker_target", "def_host", "def_action"):
            v = info.get(k)
            if v is None:
                continue
            if hasattr(v, "item"):
                info_payload[k] = int(v.item())
            else:
                info_payload[k] = v

        episode.append(obs, action_vec, reward, done, truncated, info_payload)
        total_reward += float(reward)

        if done or truncated:
            success = bool(info.get("exfil_done") is False and not done)
            break
        obs = next_obs

    return episode, RolloutStats(
        episode_return=total_reward,
        episode_length=len(episode),
        success=success,
        info_summary={"final_info": info_payload},
    )


def evaluate_policy(
    env,
    policy_fn: Callable[[np.ndarray], int],
    *,
    action_dim: int,
    n_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Greedy evaluation across n episodes; returns mean return / length / success."""
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    lengths: List[int] = []
    successes: List[bool] = []
    for i in range(n_episodes):
        ep_seed = None if seed is None else seed + i
        _, stats = collect_episode(
            env, policy_fn, action_dim=action_dim,
            epsilon=0.0, rng=rng, seed=ep_seed,
        )
        returns.append(stats.episode_return)
        lengths.append(stats.episode_length)
        successes.append(stats.success)
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/length_mean": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(successes)),
        "eval/n": float(len(returns)),
    }


# ---------------------------------------------------------------------------
# Greedy / latent policy adapters
# ---------------------------------------------------------------------------


class LatentPolicy:
    """
    Convenience wrapper that turns (RSSM, encoder, actor) into a callable
    policy_fn(obs) -> action_idx. Maintains a per-call latent state
    accumulator across an episode.
    """

    def __init__(
        self,
        rssm,
        encoder,
        actor,
        action_dim: int,
        device: torch.device,
    ):
        self.rssm = rssm
        self.encoder = encoder
        self.actor = actor
        self.action_dim = action_dim
        self.device = device
        self._state = None
        self._prev_action = None

    def reset(self) -> None:
        self._state = self.rssm.init_state(1, self.device)
        self._prev_action = torch.zeros(1, self.action_dim, device=self.device)

    def __call__(self, obs: np.ndarray) -> int:
        if self._state is None:
            self.reset()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            embed = self.encoder(obs_t)
            nonterm = torch.ones(1, 1, device=self.device)
            _, posterior = self.rssm.observe(
                embed, self._prev_action, nonterm, self._state
            )
            self._state = posterior
            feat = self.rssm.get_feature(posterior)
            action = self.actor.act_greedy(feat)
            self._prev_action = action
            return int(action.argmax(dim=-1).item())


class RandomPolicy:
    def __init__(self, action_dim: int, seed: Optional[int] = None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.action_dim))
