"""
Opponent (foe) model for FOE-Dreamer (§4 / Approach of the manuscript).

Sliding-window LSTM encoder + decoder. The encoder runs over the last
K observations and defender actions to produce an embedding tau_t of
dimension `tau_dim`. The decoder reconstructs the opponent's action
and target one step ahead from tau_t.

The encoder embedding is consumed by the actor as an additional
conditioning vector and supplies the auxiliary opponent-prediction
loss that pulls the world model's posterior toward opponent-relevant
structure.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowEncoder(nn.Module):
    """LSTM-encoded sliding window over the defender's recent local history."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, tau_dim: int, K: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.K = K
        self.input_dim = obs_dim + action_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=False)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, tau_dim),
        )

    def forward(
        self,
        obs_window: torch.Tensor,        # [K, B, obs_dim]
        action_window: torch.Tensor,     # [K, B, action_dim]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.cat([obs_window, action_window], dim=-1)
        h, hidden = self.lstm(x, hidden)
        # Take the final hidden output as tau.
        tau = self.proj(h[-1])
        return tau, hidden


class OpponentDecoder(nn.Module):
    """Predicts opponent (action, target) one step ahead from tau."""

    def __init__(self, tau_dim: int, hidden_dim: int, num_actions: int, num_targets: int):
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(tau_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.target_head = nn.Sequential(
            nn.Linear(tau_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.action_head(tau), self.target_head(tau)


class OpponentModel(nn.Module):
    """Encoder + decoder bundle wired together with the manuscript's sliding-window."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        tau_dim: int,
        K_window: int,
        hidden_dim: int,
        num_actions: int,
        num_targets: int,
    ):
        super().__init__()
        self.encoder = SlidingWindowEncoder(obs_dim, action_dim, hidden_dim, tau_dim, K_window)
        self.decoder = OpponentDecoder(tau_dim, hidden_dim, num_actions, num_targets)
        self.tau_dim = tau_dim
        self.K_window = K_window

    def encode(
        self,
        obs_window: torch.Tensor,
        action_window: torch.Tensor,
    ) -> torch.Tensor:
        tau, _ = self.encoder(obs_window, action_window)
        return tau

    def predict(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(tau)


class OpponentWindow:
    """Per-episode rolling buffer that feeds the opponent encoder at deployment."""

    def __init__(self, K: int, obs_dim: int, action_dim: int):
        self.K = K
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs: Deque[torch.Tensor] = deque(maxlen=K)
        self.actions: Deque[torch.Tensor] = deque(maxlen=K)

    def push(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        self.obs.append(obs)
        self.actions.append(action)

    def is_full(self) -> bool:
        return len(self.obs) == self.K

    def stacked(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pad with zeros if not yet full.
        while len(self.obs) < self.K:
            self.obs.appendleft(torch.zeros(self.obs_dim, device=self.obs[0].device if self.obs else "cpu"))
            self.actions.appendleft(torch.zeros(self.action_dim, device=self.actions[0].device if self.actions else "cpu"))
        obs = torch.stack(list(self.obs), dim=0).unsqueeze(1)        # [K, 1, obs_dim]
        actions = torch.stack(list(self.actions), dim=0).unsqueeze(1) # [K, 1, action_dim]
        return obs, actions
