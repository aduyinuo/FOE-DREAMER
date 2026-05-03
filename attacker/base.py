"""
Base attacker class.

Defines the kill-chain stages, the runtime loop, and the profile
dataclass shared between the two concrete attacker policies. Subclasses
override `pick_next_stage` to implement different policies (deterministic
linear progression, stochastic stage selection, etc.).
"""
from __future__ import annotations

import logging
import os
import random
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_LOG = logging.getLogger("attacker.base")

# Canonical stage names. Subclasses choose which one to execute at each tick.
KILL_CHAIN = ("foothold", "credentials", "priv_esc", "lateral", "exfil")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class AttackerProfile:
    name: str = "default"
    foothold_p: float = 0.6
    credentials_p: float = 0.5
    priv_esc_p: float = 0.4
    lateral_p: float = 0.2
    exfil_p: float = 0.6
    inter_step_seconds: float = 5.0
    target_assets: List[str] = field(default_factory=lambda: ["/root/important_data.txt"])
    revisit_p: float = 0.0          # used by the stochastic policy
    detection_avoidance: float = 0.0  # extra sleep added when an alert was recently raised

    @classmethod
    def from_yaml(cls, path: str) -> "AttackerProfile":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


# ---------------------------------------------------------------------------
# Stage operations (shared)
# ---------------------------------------------------------------------------


def op_foothold(profile: AttackerProfile, rng: random.Random) -> bool:
    if rng.random() > profile.foothold_p:
        return False
    Path("/tmp/.foothold").touch(exist_ok=True)
    return True


def op_credentials(profile: AttackerProfile, rng: random.Random) -> bool:
    return rng.random() <= profile.credentials_p


def op_priv_esc(profile: AttackerProfile, rng: random.Random) -> bool:
    if rng.random() > profile.priv_esc_p:
        return False
    try:
        subprocess.run(["touch", "/usr/bin/script1"], check=False, timeout=5)
    except Exception:
        pass
    return True


def op_lateral(profile: AttackerProfile, rng: random.Random) -> bool:
    return rng.random() <= profile.lateral_p


def op_exfil(profile: AttackerProfile, rng: random.Random) -> bool:
    if rng.random() > profile.exfil_p:
        return False
    for path in profile.target_assets:
        try:
            with open(path, "rb") as f:
                _ = f.read()
        except Exception:
            pass
    return True


STAGE_OPS = {
    "foothold": op_foothold,
    "credentials": op_credentials,
    "priv_esc": op_priv_esc,
    "lateral": op_lateral,
    "exfil": op_exfil,
}


# ---------------------------------------------------------------------------
# Base runner
# ---------------------------------------------------------------------------


class Attacker(ABC):
    """Abstract attacker. Subclasses pick the next stage to execute."""

    def __init__(self, profile: AttackerProfile, seed: int = 0):
        self.profile = profile
        self.rng = random.Random(seed)
        self.state: Dict[str, Any] = self._initial_state()
        self._running = True
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

    # ---- subclass hooks ----
    @abstractmethod
    def pick_next_stage(self) -> Optional[str]:
        """Return the name of the next stage to execute, or None to idle."""

    def on_stage_success(self, stage: str) -> None:
        """Hook called after a successful stage; subclasses may update state."""

    def on_stage_failure(self, stage: str) -> None:
        """Hook called after a failed stage."""

    # ---- lifecycle ----
    def _initial_state(self) -> Dict[str, Any]:
        return {"completed": set(), "tick": 0}

    def _stop(self, signum, frame):  # noqa: ARG002
        _LOG.info("[%s] received signal %d, stopping", self.profile.name, signum)
        self._running = False

    def run(self) -> None:
        _LOG.info("[%s] attacker loop starting (pid=%d)",
                  self.profile.name, os.getpid())
        while self._running:
            stage = self.pick_next_stage()
            if stage is None:
                time.sleep(self.profile.inter_step_seconds)
                self.state["tick"] += 1
                continue
            ok = STAGE_OPS[stage](self.profile, self.rng)
            if ok:
                _LOG.info("[%s] stage[%s] success", self.profile.name, stage)
                self.on_stage_success(stage)
            else:
                _LOG.info("[%s] stage[%s] failure", self.profile.name, stage)
                self.on_stage_failure(stage)
            time.sleep(self.profile.inter_step_seconds)
            self.state["tick"] += 1
        _LOG.info("[%s] attacker loop exiting", self.profile.name)
