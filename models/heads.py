"""
Encoder, decoder, actor, critic, and reward heads for FOE-Dreamer.

These are generic MLP heads. The factored-observation cyber
environment ships a flat float32 vector observation, so MLP encoders
suffice. A graph-encoder variant is left as a drop-in replacement for the
encoder/decoder pair via the same forward signature.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.distributions as td
import torch.nn as nn


# ---------------------------------------------------------------------------
# Observation encoder / decoder
# ---------------------------------------------------------------------------


class MLPEncoder(nn.Module):
    """Maps observation -> embedding via a small MLP."""

    def __init__(self, obs_dim: int, embed_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDecoder(nn.Module):
    """Reconstructs observation from latent feature [z, h] as a Normal dist."""

    def __init__(self, feat_dim: int, obs_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, obs_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim

    def forward(self, feat: torch.Tensor) -> td.Distribution:
        mean = self.net(feat)
        dist = td.Normal(mean, 1.0)
        return td.Independent(dist, 1)


# ---------------------------------------------------------------------------
# Reward head
# ---------------------------------------------------------------------------


class RewardHead(nn.Module):
    """Predicts scalar reward as Normal(mean, 1.0) from latent feature."""

    def __init__(self, feat_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> td.Distribution:
        mean = self.net(feat).squeeze(-1)
        return td.Normal(mean, 1.0)


class DiscountHead(nn.Module):
    """Predicts terminal/non-terminal as Bernoulli from latent feature."""

    def __init__(self, feat_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> td.Distribution:
        logit = self.net(feat).squeeze(-1)
        return td.Bernoulli(logits=logit)


# ---------------------------------------------------------------------------
# Actor and critic
# ---------------------------------------------------------------------------


class DiscreteActor(nn.Module):
    """
    Categorical policy over a discrete action set, sampled with a
    straight-through Gumbel-softmax during imagination so gradients flow
    through the latent rollout. At deployment, sampling collapses to argmax.
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, td.Distribution]:
        logits = self.net(feat)
        dist = td.OneHotCategoricalStraightThrough(logits=logits)
        action = dist.rsample()
        return action, dist

    def act_greedy(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat)
        idx = logits.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, num_classes=self.action_dim).float()


class Critic(nn.Module):
    """Value head: feat -> scalar V(s)."""

    def __init__(self, feat_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> td.Distribution:
        mean = self.net(feat).squeeze(-1)
        return td.Normal(mean, 1.0)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak update: target <- tau*source + (1-tau)*target."""
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.copy_(p_s.data)


def trainable_params(modules: Iterable[nn.Module]):
    seen = set()
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield p
