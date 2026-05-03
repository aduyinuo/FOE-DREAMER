"""Baseline policies (§5.1 of the manuscript)."""
from .heuristic import ExpertHeuristic
from .iqn import IQNNetwork, quantile_huber_loss
from .rainbow import NoisyLinear, PrioritizedReplay, RainbowNetwork
from .random_policy import RandomBaseline
from .sleep import SleepPolicy

__all__ = [
    "ExpertHeuristic",
    "IQNNetwork", "quantile_huber_loss",
    "NoisyLinear", "PrioritizedReplay", "RainbowNetwork",
    "RandomBaseline",
    "SleepPolicy",
]
