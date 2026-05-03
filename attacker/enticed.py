"""
Enticed attacker (§4.1 of the manuscript).

Permissive decision policy that tolerates ambiguous deception
indicators when high-value opportunities are present. Reflects the
"smash-and-grab" tactics of opportunistic adversaries: automated
malware, insider threats, financially motivated actors. Failed probes
are not fatal; they merely shift priority to other targets.

The Enticed attacker walks the kill chain rapidly and aggressively
pivots when it captures credentials or escalation primitives.
"""
from __future__ import annotations

from typing import Optional

from .base import Attacker, KILL_CHAIN


class EnticedAttacker(Attacker):
    """Opportunistic attacker that ignores ambiguous deception signals."""

    def _initial_state(self) -> dict:
        st = super()._initial_state()
        st["stage_idx"] = 0
        st["finished"] = False
        return st

    def pick_next_stage(self) -> Optional[str]:
        if self.state["finished"]:
            return None
        return KILL_CHAIN[self.state["stage_idx"]]

    def on_stage_success(self, stage: str) -> None:
        self.state["completed"].add(stage)
        self.state["stage_idx"] += 1
        if self.state["stage_idx"] >= len(KILL_CHAIN):
            self.state["finished"] = True

    def on_stage_failure(self, stage: str) -> None:
        # Permissive: a failed stage doesn't cause withdrawal. Try again
        # next tick at the same stage; the per-stage probability fires
        # eventually.
        pass
