"""
Rainbow baseline (§5.1 of the manuscript).

Implements the Rainbow ensemble: distributional Q-learning (C51) on
top of dueling network architecture, prioritized experience replay,
double Q-learning, n-step returns, and noisy nets for exploration.
The agent shares the same observation stream, action space, and
training budget as FOE-Dreamer.

Reference implementation; hyperparameters are documented in
Appendix C of the manuscript.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noisy linear layer (Fortunato et al., 2018)
# ---------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init * bound)
        self.bias_sigma.data.fill_(self.sigma_init * bound)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# Dueling distributional network (C51)
# ---------------------------------------------------------------------------


class RainbowNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        atoms: int = 51,
        hidden: int = 256,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.atoms = atoms
        self.action_dim = action_dim
        self.v_min, self.v_max = v_min, v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Dueling heads: value V(s) + advantage A(s, a) over atoms.
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, hidden), nn.ReLU(),
            NoisyLinear(hidden, atoms),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, hidden), nn.ReLU(),
            NoisyLinear(hidden, action_dim * atoms),
        )

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.feature(obs)
        v = self.value_stream(f).view(-1, 1, self.atoms)
        a = self.advantage_stream(f).view(-1, self.action_dim, self.atoms)
        q_atoms = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return (self.forward(obs) * self.support).sum(dim=-1)

    def act_greedy(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            return int(self.q_values(obs).argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Prioritized replay (sum-tree)
# ---------------------------------------------------------------------------


@dataclass
class PrioritizedReplay:
    capacity: int
    alpha: float = 0.5
    beta_start: float = 0.4
    beta_steps: int = 100_000

    storage: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = field(default_factory=list)
    priorities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pos: int = 0
    max_priority: float = 1.0

    def __post_init__(self):
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

    def push(self, obs, a, r, next_obs, done):
        if len(self.storage) < self.capacity:
            self.storage.append((obs, a, r, next_obs, done))
        else:
            self.storage[self.pos] = (obs, a, r, next_obs, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, step: int):
        if not self.storage:
            return None
        n = len(self.storage)
        probs = self.priorities[:n] ** self.alpha
        probs = probs / probs.sum()
        idx = np.random.choice(n, batch_size, p=probs)
        beta = min(1.0, self.beta_start + step / max(1, self.beta_steps) * (1.0 - self.beta_start))
        weights = (n * probs[idx]) ** (-beta)
        weights = weights / weights.max()
        batch = [self.storage[i] for i in idx]
        return batch, idx, weights

    def update_priorities(self, idx, priorities):
        for i, p in zip(idx, priorities):
            self.priorities[i] = float(p)
            self.max_priority = max(self.max_priority, float(p))
