"""
U-rand user (§4.3 of the manuscript).

Memoryless: at every step each user samples an action independently.
The probability of acting at any given tick is `step_action_p`,
independent of every previous tick.
"""
from __future__ import annotations

from .base import User


class UrandUser(User):
    """Memoryless user with independent per-step action sampling."""

    def should_act_now(self) -> bool:
        return self.rng.random() < self.profile.step_action_p
