"""Environment factories for FOE-Dreamer."""
from typing import Any

from .c2_bridge import C2Bridge, DEFENDER_ACTIONS, HOST_FEATURE_DIM
from .cyber_env import CyberEnv, CyberHost, CyberScenario

__all__ = [
    "C2Bridge", "DEFENDER_ACTIONS", "HOST_FEATURE_DIM",
    "CyberEnv", "CyberHost", "CyberScenario",
    "make_env",
]


def make_env(scenario_path: str, **kwargs: Any) -> CyberEnv:
    """Build the cyber-defense environment from a scenario YAML path."""
    return CyberEnv(
        scenario_path=scenario_path,
        keep_alive=kwargs.get("keep_alive", True),
    )
