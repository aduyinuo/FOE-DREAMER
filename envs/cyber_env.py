"""
Cyber-defense environment.

  - Scenario topology (hosts, services, vulnerabilities, attacker
    capabilities) is described by a YAML scenario spec under
    `scenarios/`.
  - On reset(), an Ansible playbook is invoked to (re)provision the
    scenario in the underlying compute substrate. The playbook is
    parameterized by the scenario YAML and an inventory file pointing
    at the substrate's API endpoint.
  - Observations are gathered through the C2 bridge: per-host telemetry
    plus an alert digest from the network sensors.
  - Defender actions are issued through the same C2 bridge; the bridge
    translates to per-host control RPCs.
  - At terminal / truncation, a teardown playbook is run.

The environment treats the substrate as opaque: anything that can
satisfy the inventory contract (an SSH-reachable controller and N
worker hosts with a known role assignment) works.

Concurrency: only one episode at a time per inventory.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from .c2_bridge import C2Bridge, DEFENDER_ACTIONS, HOST_FEATURE_DIM


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario YAML schema
# ---------------------------------------------------------------------------


@dataclass
class CyberHost:
    name: str
    role: str
    image: str
    flavor: str
    services: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class CyberScenario:
    name: str
    hosts: List[CyberHost]
    network: Dict[str, Any] = field(default_factory=dict)
    attacker_profile: str = "default"
    max_steps: int = 100
    inventory_path: str = "envs/inventory.yml"
    playbook_dir: str = "envs/playbooks"
    c2_host: str = "127.0.0.1"
    c2_port: int = 50051

    @classmethod
    def from_yaml(cls, path: str) -> "CyberScenario":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        hosts = [CyberHost(**h) for h in raw.pop("hosts")]
        return cls(hosts=hosts, **raw)


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------


class AnsibleProvisioner:
    """Runs Ansible playbooks against the substrate."""

    def __init__(self, scenario: CyberScenario):
        self.scenario = scenario
        self.playbook_dir = Path(scenario.playbook_dir)
        self.inventory_path = Path(scenario.inventory_path)

    def _run(self, playbook: str, extra_vars: Optional[Dict[str, Any]] = None) -> None:
        cmd = [
            "ansible-playbook",
            "-i", str(self.inventory_path),
            str(self.playbook_dir / playbook),
        ]
        if extra_vars:
            cmd += ["--extra-vars", json.dumps(extra_vars)]
        _LOG.info("ansible: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("ANSIBLE_HOST_KEY_CHECKING", "False")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            _LOG.error("ansible stdout: %s", result.stdout[-2000:])
            _LOG.error("ansible stderr: %s", result.stderr[-2000:])
            raise RuntimeError(f"playbook {playbook} failed (rc={result.returncode})")

    def deploy(self) -> None:
        scenario_blob = {
            "scenario_name": self.scenario.name,
            "hosts": [h.__dict__ for h in self.scenario.hosts],
            "attacker_profile": self.scenario.attacker_profile,
            "network": self.scenario.network,
        }
        self._run("deploy_scenario.yml", extra_vars=scenario_blob)

    def reset_state(self) -> None:
        """Reset scenario state without tearing down infrastructure."""
        self._run("reset_state.yml")

    def teardown(self) -> None:
        self._run("teardown.yml")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CyberEnv(gym.Env):
    """Cyber-defense env with Ansible-driven scenario boot."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str,
        keep_alive: bool = True,
    ):
        super().__init__()
        self.scenario = CyberScenario.from_yaml(scenario_path)
        self.keep_alive = keep_alive  # if True, deploy once and reuse across resets
        self.provisioner = AnsibleProvisioner(self.scenario)

        self.host_inventory = [h.name for h in self.scenario.hosts]
        self.num_hosts = len(self.host_inventory)

        self.action_space = spaces.Discrete(self.num_hosts * len(DEFENDER_ACTIONS))
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self.num_hosts * HOST_FEATURE_DIM,),
            dtype=np.float32,
        )

        self._step_idx = 0
        self._deployed = False
        self._bridge: Optional[C2Bridge] = None

    # ----------------- gym API -----------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        del seed, options
        if not self._deployed:
            self.provisioner.deploy()
            self._deployed = True
            self._bridge = C2Bridge(
                self.scenario.c2_host,
                self.scenario.c2_port,
                host_inventory=self.host_inventory,
            )
        else:
            self.provisioner.reset_state()
            if self._bridge is not None:
                self._bridge.reset()

        self._step_idx = 0
        obs = self._collect_observation()
        return obs, {"step_idx": 0, "scenario": self.scenario.name}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        host_idx = int(action) // len(DEFENDER_ACTIONS)
        action_idx = int(action) % len(DEFENDER_ACTIONS)
        action_name = DEFENDER_ACTIONS[action_idx]

        success, message = self._bridge.apply_action_index(host_idx, action_idx)

        # Five-second polling interval per §4.2.
        time.sleep(5.0)

        obs = self._collect_observation()
        reward = self._compute_reward(obs, success)

        self._step_idx += 1
        truncated = self._step_idx >= self.scenario.max_steps
        done = self._is_terminal()

        info = {
            "step_idx": self._step_idx,
            "def_host": host_idx,
            "def_action": action_name,
            "action_success": success,
            "action_message": message,
        }
        return obs, float(reward), done, truncated, info

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        if self._deployed and not self.keep_alive:
            self.provisioner.teardown()
            self._deployed = False

    # ----------------- helpers -----------------
    def _collect_observation(self) -> np.ndarray:
        return self._bridge.collect_observation().astype(np.float32)

    def _compute_reward(self, obs: np.ndarray, action_success: bool) -> float:
        # A per-step time penalty, a bonus for a successful defensive
        # action, and a penalty proportional to the number of hosts
        # whose alert vector indicates active compromise.
        flat = obs.reshape(self.num_hosts, HOST_FEATURE_DIM)
        active_alerts = flat[:, :4].sum(axis=1)  # first 4 features are activity flags
        compromise_penalty = -1.0 * float((active_alerts > 0).sum())
        action_bonus = 0.1 if action_success else 0.0
        return compromise_penalty + action_bonus - 0.01

    def _is_terminal(self) -> bool:
        # Terminal condition: any host with a confirmed-exfil signal.
        for host in self.host_inventory:
            hs = self._bridge.get_host_state(host)
            if hs.cron_modified and hs.suid_active and hs.suspicious_login_count > 5:
                return True
        return False
