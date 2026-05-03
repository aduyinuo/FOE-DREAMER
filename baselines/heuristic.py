"""
Expert-Inspired Heuristic baseline (§5.1 of the manuscript).

Encodes the standard cyber-deception playbook as a fixed policy:

  - At episode start: deploy a Cowrie SSH honeypot on one public-facing
    server, plant a honeyfile on a randomly chosen internal host.
  - During the episode: Isolate any host that triggers a decoy or raises
    an escalation alert. Reboot an isolated host once if no further
    alerts arrive within a quiet window.
  - Otherwise: Null.

The five-feature observation per host (alerts_count, honeypot_active,
host_up, trap_fired, last_event_age_s) is read directly from the obs
vector at fixed offsets.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from envs import DEFENDER_ACTIONS, HOST_FEATURE_DIM


# Indices into the per-host feature block.
F_ALERTS = 0
F_HONEYPOT = 1
F_HOST_UP = 2
F_TRAP_FIRED = 3
F_LAST_EVENT_AGE = 4


class ExpertHeuristic:
    """Rule-based defender derived from the standard deception playbook."""

    def __init__(
        self,
        num_hosts: int,
        public_host_indices: List[int],
        internal_host_indices: List[int],
        quiet_window_s: float = 30.0,
        seed: Optional[int] = None,
    ):
        self.num_hosts = num_hosts
        self.action_dim_per_host = len(DEFENDER_ACTIONS)
        self.public = list(public_host_indices)
        self.internal = list(internal_host_indices)
        self.quiet_window_s = quiet_window_s
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._isolated: dict = {}        # host_idx -> True if currently isolated
        self._isolated_at: dict = {}     # host_idx -> step when isolated
        self._honeypot_target: Optional[int] = None
        self._honeyfile_target: Optional[int] = None

    def _enc(self, host_idx: int, action_name: str) -> int:
        return host_idx * self.action_dim_per_host + DEFENDER_ACTIONS.index(action_name)

    def reset(self) -> None:
        self._step = 0
        self._isolated = {}
        self._isolated_at = {}
        self._honeypot_target = None
        self._honeyfile_target = None

    def __call__(self, obs: np.ndarray) -> int:
        self._step += 1

        # Episode-start setup actions.
        if self._honeypot_target is None and self.public:
            self._honeypot_target = int(self.rng.choice(self.public))
            return self._enc(self._honeypot_target, "deceive")
        if self._honeyfile_target is None and self.internal:
            self._honeyfile_target = int(self.rng.choice(self.internal))
            return self._enc(self._honeyfile_target, "deceive")

        # Inspect per-host signals.
        flat = obs.reshape(self.num_hosts, HOST_FEATURE_DIM)
        for h in range(self.num_hosts):
            trap_fired = flat[h, F_TRAP_FIRED] > 0.5
            recent_alerts = flat[h, F_ALERTS] > 0
            already_iso = self._isolated.get(h, False)

            if (trap_fired or recent_alerts) and not already_iso:
                self._isolated[h] = True
                self._isolated_at[h] = self._step
                return self._enc(h, "isolate")

            if already_iso:
                # Reboot once after a quiet window if alerts have stopped.
                quiet_steps = self._step - self._isolated_at[h]
                age = float(flat[h, F_LAST_EVENT_AGE])
                if quiet_steps >= 1 and age >= self.quiet_window_s:
                    self._isolated.pop(h, None)
                    self._isolated_at.pop(h, None)
                    return self._enc(h, "disrupt")

        # No action this tick.
        return self._enc(0, "null")
