"""
Random baseline (§5.1 of the manuscript).

Samples uniformly over the (host, action) joint space at each polling
step. Shows what undirected activity buys against each attacker.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from envs import DEFENDER_ACTIONS


class RandomBaseline:
    """Uniform-random defender."""

    def __init__(self, num_hosts: int, seed: Optional[int] = None):
        self.num_hosts = num_hosts
        self.action_dim_per_host = len(DEFENDER_ACTIONS)
        self.total = num_hosts * self.action_dim_per_host
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.total))
