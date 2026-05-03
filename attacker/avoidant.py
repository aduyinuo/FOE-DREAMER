"""
Avoidant attacker (§4.1 of the manuscript).

Conservative decision policy: any positive deception indicator triggers
immediate withdrawal. Operationalizes the systematic fingerprinting
techniques used by sophisticated APT-style adversaries; banner
inconsistencies, failed credential validations, or missing production
markers cause the attacker to abort.

Probes return binary outcomes (Suspicious / Not_Suspicious) computed
from attacker-observable state only. Withdrawal sets the host to
"abandoned" for the remainder of the episode; the attacker may resume
on a different host if available.
"""
from __future__ import annotations

from typing import Optional

from .base import Attacker, KILL_CHAIN


PROBE_KINDS = ("banner_check", "credential_validation", "post_exploit_marker")


class AvoidantAttacker(Attacker):
    """Probe-first attacker that withdraws on positive deception indicators."""

    def _initial_state(self) -> dict:
        st = super()._initial_state()
        st["abandoned_hosts"] = set()
        st["stage_idx"] = 0
        st["finished"] = False
        return st

    # ---- probe ----
    def _probe_suspicious(self, kind: str) -> bool:
        # Avoidant uses a low threshold — any deviation triggers withdrawal.
        return self.rng.random() < 0.55  # banner / cred / marker probe

    def pick_next_stage(self) -> Optional[str]:
        if self.state["finished"]:
            return None
        # Run a probe before each stage; if Suspicious, withdraw and idle.
        kind = PROBE_KINDS[self.state["stage_idx"] % len(PROBE_KINDS)]
        if self._probe_suspicious(kind):
            # Suspicious — abandon this attempt this tick.
            return None
        return KILL_CHAIN[self.state["stage_idx"]]

    def on_stage_success(self, stage: str) -> None:
        self.state["completed"].add(stage)
        self.state["stage_idx"] += 1
        if self.state["stage_idx"] >= len(KILL_CHAIN):
            self.state["finished"] = True
