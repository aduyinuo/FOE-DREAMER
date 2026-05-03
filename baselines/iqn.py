"""
IQN baseline (§5.1 of the manuscript).

Implicit Quantile Networks (Dabney et al., 2018). Samples quantile
fractions tau ~ U(0, 1) at each forward pass and predicts a
quantile-conditioned Q-value. The Huber-quantile regression loss
trains the network to match the empirical return distribution.

Reference implementation; hyperparameters are documented in
Appendix C of the manuscript.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IQNNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 64,
        hidden: int = 256,
        n_cosines: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.n_cosines = n_cosines

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim), nn.ReLU(),
        )
        self.quantile_embed = nn.Linear(n_cosines, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def _cosine_embed(self, taus: torch.Tensor) -> torch.Tensor:
        # taus: [B, N]
        i = torch.arange(1, self.n_cosines + 1, device=taus.device).float()
        return torch.cos(math.pi * i * taus.unsqueeze(-1))   # [B, N, n_cosines]

    def forward(
        self, obs: torch.Tensor, n_quantiles: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        feat = self.feature(obs)                             # [B, embed]
        taus = torch.rand(B, n_quantiles, device=obs.device) # [B, N]
        cos = self._cosine_embed(taus)                       # [B, N, n_cosines]
        phi = F.relu(self.quantile_embed(cos))               # [B, N, embed]
        x = feat.unsqueeze(1) * phi                          # [B, N, embed]
        q = self.head(x.reshape(B * n_quantiles, -1))        # [B*N, A]
        q = q.reshape(B, n_quantiles, self.action_dim)
        return q, taus

    def q_values(self, obs: torch.Tensor, n_quantiles: int = 32) -> torch.Tensor:
        q, _ = self.forward(obs, n_quantiles)
        return q.mean(dim=1)

    def act_greedy(self, obs: torch.Tensor, n_quantiles: int = 32) -> int:
        with torch.no_grad():
            return int(self.q_values(obs, n_quantiles).argmax(dim=-1).item())


def quantile_huber_loss(
    pred_q: torch.Tensor,        # [B, N, A] selected over chosen action: [B, N]
    target_q: torch.Tensor,      # [B, N']
    taus: torch.Tensor,          # [B, N]
    kappa: float = 1.0,
) -> torch.Tensor:
    """Huber-quantile regression loss as in Dabney et al. (2018)."""
    td = target_q.unsqueeze(1) - pred_q.unsqueeze(2)  # [B, N, N']
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    indicator = (td < 0).float()
    weight = (taus.unsqueeze(2) - indicator).abs()
    return (weight * huber).sum(dim=2).mean()
