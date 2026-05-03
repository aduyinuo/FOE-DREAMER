"""
Sleep baseline (§5.1 of the manuscript).

Always selects the Null action. No-defense lower bound that shows
how much damage each attacker profile inflicts when nothing intervenes.
"""
from __future__ import annotations

import numpy as np

from envs import DEFENDER_ACTIONS


class SleepPolicy:
    """Always picks the Null action across all hosts."""

    def __init__(self, num_hosts: int):
        self.num_hosts = num_hosts
        self.null_idx = DEFENDER_ACTIONS.index("null")
        self.action_dim_per_host = len(DEFENDER_ACTIONS)

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> int:
        # Encode (host=0, action=null) as a single discrete action index.
        return 0 * self.action_dim_per_host + self.null_idx
