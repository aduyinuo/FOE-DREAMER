"""
Sequence-aware episodic replay buffer for world-model training.

Stores complete episodes; samples fixed-length windows. Each sample is
a stack of [seq_len, batch_size, ...] tensors covering observation,
action, reward, nonterminal, and (optionally) attacker action and
target labels for the foe head's auxiliary loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Episode:
    obs: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    done: List[bool] = field(default_factory=list)
    truncated: List[bool] = field(default_factory=list)
    info: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def append(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.action.append(np.asarray(action, dtype=np.float32))
        self.reward.append(float(reward))
        self.done.append(bool(done))
        self.truncated.append(bool(truncated))
        self.info.append(info)


class EpisodicReplay:
    """Stores complete episodes; samples fixed-length sub-sequences."""

    def __init__(
        self,
        capacity_episodes: int,
        seq_len: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity_episodes = capacity_episodes
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self._episodes: List[Episode] = []

    # -------- writing --------
    def add_episode(self, ep: Episode) -> None:
        if len(ep) < 2:
            return
        self._episodes.append(ep)
        if len(self._episodes) > self.capacity_episodes:
            self._episodes.pop(0)

    # -------- reading --------
    def __len__(self) -> int:
        return sum(len(e) for e in self._episodes)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def has_min_steps(self, n: int) -> bool:
        return len(self) >= n

    def sample(
        self, batch_size: int, info_keys: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        info_keys = info_keys or []
        if not self._episodes:
            raise RuntimeError("replay buffer is empty")

        obs_b = np.zeros((self.seq_len, batch_size, self.obs_dim), dtype=np.float32)
        act_b = np.zeros((self.seq_len, batch_size, self.action_dim), dtype=np.float32)
        rew_b = np.zeros((self.seq_len, batch_size), dtype=np.float32)
        nonterm_b = np.ones((self.seq_len, batch_size), dtype=np.float32)
        info_b: Dict[str, np.ndarray] = {
            k: np.full((self.seq_len, batch_size), -1, dtype=np.int64) for k in info_keys
        }

        rng = np.random.default_rng()
        for b in range(batch_size):
            ep = self._episodes[rng.integers(0, len(self._episodes))]
            if len(ep) <= self.seq_len:
                start = 0
                window = list(range(len(ep)))
                while len(window) < self.seq_len:
                    window.append(window[-1])
            else:
                start = int(rng.integers(0, len(ep) - self.seq_len + 1))
                window = list(range(start, start + self.seq_len))

            for t, idx in enumerate(window):
                obs_b[t, b] = ep.obs[idx]
                act_b[t, b] = ep.action[idx]
                rew_b[t, b] = ep.reward[idx]
                nonterm_b[t, b] = 0.0 if ep.done[idx] else 1.0
                for k in info_keys:
                    val = ep.info[idx].get(k, -1)
                    info_b[k][t, b] = int(val) if val is not None else -1

        out = {
            "obs": torch.from_numpy(obs_b).to(self.device),
            "action": torch.from_numpy(act_b).to(self.device),
            "reward": torch.from_numpy(rew_b).to(self.device),
            "nonterm": torch.from_numpy(nonterm_b).to(self.device),
        }
        for k in info_keys:
            out[k] = torch.from_numpy(info_b[k]).to(self.device)
        return out

    def stats(self) -> Dict[str, float]:
        lens = [len(e) for e in self._episodes]
        rews = [sum(e.reward) for e in self._episodes]
        return {
            "replay/num_eps": float(len(self._episodes)),
            "replay/total_steps": float(sum(lens)),
            "replay/mean_ep_len": float(np.mean(lens)) if lens else 0.0,
            "replay/mean_ep_return": float(np.mean(rews)) if rews else 0.0,
        }
