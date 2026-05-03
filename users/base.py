"""
Base user simulator (§4.3 of the manuscript).

Generates background user activity that the defender must learn to
distinguish from attacker behavior. Activities include browsing,
SSH administration, database queries, and file transfers, with
durations and gaps drawn from per-persona distributions.

The two personas U-rand and U-burst differ in their *timing* dynamics
rather than their op vocabulary. Each persona inherits from User,
which provides the op library and the runtime loop, and overrides
`should_act_now` to implement its sampling rule.
"""
from __future__ import annotations

import datetime
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


_LOG = logging.getLogger("users.base")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    name: str = "default"
    home: str = "/home/user"

    # Per-step probability of acting (used by U-rand).
    step_action_p: float = 0.4

    # ON/OFF dynamics (used by U-burst).
    on_to_off_p: float = 0.05            # transition probability per step while ON
    off_to_on_p: float = 0.10            # transition probability per step while OFF
    on_duration_alpha: float = 1.5       # heavy-tail alpha (Pareto) for ON duration
    on_duration_min: float = 5.0         # minimum ON duration (steps)

    # Per-step spacing in seconds (real-time wall-clock between actions).
    step_seconds: float = 5.0

    # Op vocabulary parameters.
    browse_session_min_requests: int = 5
    browse_session_max_requests: int = 10
    browse_request_seconds: float = 2.0
    ssh_session_min_seconds: float = 20.0
    ssh_session_max_seconds: float = 30.0
    db_query_min_seconds: float = 0.1
    db_query_max_seconds: float = 5.0
    file_transfer_min_seconds: float = 30.0
    file_transfer_max_seconds: float = 60.0

    peer_hosts: List[str] = field(default_factory=list)
    web_endpoints: List[str] = field(default_factory=list)
    db_targets: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "UserProfile":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


# ---------------------------------------------------------------------------
# Op library
# ---------------------------------------------------------------------------


def op_browse(profile: UserProfile, rng: random.Random) -> bool:
    """Browser session: 5-10 HTTP requests over 10-20 seconds."""
    if not profile.web_endpoints:
        return False
    n = rng.randint(profile.browse_session_min_requests,
                    profile.browse_session_max_requests)
    for _ in range(n):
        url = rng.choice(profile.web_endpoints)
        try:
            subprocess.run(
                ["curl", "-fsS", "-o", "/dev/null", "--max-time", "3", url],
                check=False, timeout=4,
            )
        except Exception:
            pass
        time.sleep(profile.browse_request_seconds * rng.uniform(0.5, 1.5))
    return True


def op_ssh_admin(profile: UserProfile, rng: random.Random) -> bool:
    """Long SSH session to a peer host (20-30s)."""
    if not profile.peer_hosts:
        return False
    host = rng.choice(profile.peer_hosts)
    duration = rng.uniform(profile.ssh_session_min_seconds, profile.ssh_session_max_seconds)
    try:
        subprocess.run(
            [
                "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=3",
                host, f"sleep {duration:.1f}; true",
            ],
            check=False, timeout=duration + 5,
        )
    except Exception:
        pass
    return True


def op_db_query(profile: UserProfile, rng: random.Random) -> bool:
    """Database query: sub-second to several seconds."""
    if not profile.db_targets:
        return False
    duration = rng.uniform(profile.db_query_min_seconds, profile.db_query_max_seconds)
    time.sleep(duration)
    return True


def op_file_transfer(profile: UserProfile, rng: random.Random) -> bool:
    """File transfer: 30-60s of activity."""
    home = Path(profile.home) / "transfers"
    home.mkdir(parents=True, exist_ok=True)
    duration = rng.uniform(profile.file_transfer_min_seconds, profile.file_transfer_max_seconds)
    end_at = time.time() + duration
    while time.time() < end_at:
        with open(home / f"chunk-{rng.randint(0, 999)}.bin", "ab") as f:
            f.write(os.urandom(4096))
        time.sleep(0.5)
    return True


OP_FNS = {
    "browse": op_browse,
    "ssh_admin": op_ssh_admin,
    "db_query": op_db_query,
    "file_transfer": op_file_transfer,
}


# ---------------------------------------------------------------------------
# Base user
# ---------------------------------------------------------------------------


class User(ABC):
    """Abstract user simulator. Subclasses define when to act."""

    def __init__(self, profile: UserProfile, seed: int = 0):
        self.profile = profile
        self.rng = random.Random(seed)
        self.state: Dict[str, Any] = self._initial_state()
        self._running = True
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

    def _initial_state(self) -> Dict[str, Any]:
        return {"tick": 0}

    def _stop(self, signum, frame):  # noqa: ARG002
        _LOG.info("[%s] received signal %d, stopping", self.profile.name, signum)
        self._running = False

    @abstractmethod
    def should_act_now(self) -> bool:
        """Return True if the user should run an op this tick."""

    def op_mix(self) -> Dict[str, float]:
        return {"browse": 0.4, "ssh_admin": 0.2, "db_query": 0.2, "file_transfer": 0.2}

    def _pick_op(self) -> str:
        mix = self.op_mix()
        ops = list(mix.keys())
        weights = [mix[k] for k in ops]
        return self.rng.choices(ops, weights=weights, k=1)[0]

    def run(self) -> None:
        _LOG.info("[%s] user simulator starting (pid=%d)",
                  self.profile.name, os.getpid())
        while self._running:
            if self.should_act_now():
                op_name = self._pick_op()
                op_fn = OP_FNS[op_name]
                ok = op_fn(self.profile, self.rng)
                _LOG.debug("[%s] op=%s ok=%s", self.profile.name, op_name, ok)
            self.state["tick"] += 1
            time.sleep(self.profile.step_seconds)
        _LOG.info("[%s] user simulator exiting", self.profile.name)
