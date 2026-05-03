"""
U-burst user (§4.3 of the manuscript).

ON/OFF dynamics with heavy-tail ON durations: the user alternates
between active phases (during which it acts every step) and idle
phases (during which it does nothing). Active-phase durations are
drawn from a Pareto heavy-tail distribution, producing extended
periods of concentrated activity that overlap with attacker actions
in the alert stream.

The defender model trained against U-burst must learn to recognize
that a sustained burst of activity may be benign even when its
features overlap with attacker behavior.
"""
from __future__ import annotations

from .base import User


class UburstUser(User):
    """ON/OFF user with heavy-tail ON durations."""

    def _initial_state(self) -> dict:
        st = super()._initial_state()
        st["state"] = "off"
        st["state_ticks_remaining"] = 0
        return st

    def _draw_on_duration(self) -> int:
        alpha = self.profile.on_duration_alpha
        # Pareto draw shifted by minimum duration (in ticks).
        return int(self.profile.on_duration_min * (1.0 / (1.0 - self.rng.random())) ** (1.0 / alpha))

    def should_act_now(self) -> bool:
        if self.state["state"] == "on":
            self.state["state_ticks_remaining"] -= 1
            if self.state["state_ticks_remaining"] <= 0 or self.rng.random() < self.profile.on_to_off_p:
                self.state["state"] = "off"
            return True
        # off phase
        if self.rng.random() < self.profile.off_to_on_p:
            self.state["state"] = "on"
            self.state["state_ticks_remaining"] = self._draw_on_duration()
            return True
        return False
