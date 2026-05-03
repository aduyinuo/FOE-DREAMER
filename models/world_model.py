"""
Factored world model for FOE-Dreamer (§4 / Approach of the manuscript).

The latent state factors into two parallel stochastic components on top
of a shared deterministic recurrent state h_t:

  z_t : *controllable* latent. Posterior conditioned on (h_t, e_t,
        a_{t-1}); prior conditioned on (h_t, a_{t-1}). The defender's
        action drives z, so this is the part of the latent the policy
        controls.

  u_t : *exogenous* latent. Posterior conditioned on (h_t, e_t); prior
        conditioned on h_t alone. User and environment dynamics that
        the defender cannot influence live here, isolating their
        variance from the value function.

The recurrent state evolves as h_t = GRU(h_{t-1}, embed(z_{t-1}, u_{t-1},
a_{t-1})). Reward and discount are read out from (z, h); the decoder
reconstructs the observation from (z, u, h).
"""
from __future__ import annotations

from collections import namedtuple
from typing import List, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

# State containers ---------------------------------------------------------

FactoredState = namedtuple(
    "FactoredState",
    ["z_mean", "z_std", "z_stoch", "u_mean", "u_std", "u_stoch", "deter"],
)


def _seq_to_batch(x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    return x.reshape(batch_size * seq_len, *x.shape[2:])


def _batch_to_seq(x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    return x.reshape(seq_len, batch_size, *x.shape[1:])


# ---------------------------------------------------------------------------
# Factored RSSM
# ---------------------------------------------------------------------------


class FactoredRSSM(nn.Module):
    """Factored recurrent state-space model with controllable z and exogenous u."""

    def __init__(
        self,
        action_size: int,
        z_dim: int,
        u_dim: int,
        deter_size: int,
        node_size: int,
        embedding_size: int,
        device: torch.device,
        min_std: float = 0.1,
        act_fn=nn.ELU,
    ):
        super().__init__()
        self.action_size = action_size
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.deter_size = deter_size
        self.node_size = node_size
        self.embedding_size = embedding_size
        self.device = device
        self.min_std = min_std
        self.act_fn = act_fn

        # h_t = GRU(h_{t-1}, embed(z_{t-1}, u_{t-1}, a_{t-1}))
        self.fc_embed_state_action = nn.Sequential(
            nn.Linear(z_dim + u_dim + action_size, deter_size),
            act_fn(),
        )
        self.rnn = nn.GRUCell(deter_size, deter_size)

        # priors: action-conditioned for z, action-independent for u
        self.fc_prior_z = nn.Sequential(
            nn.Linear(deter_size + action_size, node_size),
            act_fn(),
            nn.Linear(node_size, 2 * z_dim),
        )
        self.fc_prior_u = nn.Sequential(
            nn.Linear(deter_size, node_size),
            act_fn(),
            nn.Linear(node_size, 2 * u_dim),
        )

        # posteriors: action-conditioned for z, action-independent for u
        self.fc_post_z = nn.Sequential(
            nn.Linear(deter_size + embedding_size + action_size, node_size),
            act_fn(),
            nn.Linear(node_size, 2 * z_dim),
        )
        self.fc_post_u = nn.Sequential(
            nn.Linear(deter_size + embedding_size, node_size),
            act_fn(),
            nn.Linear(node_size, 2 * u_dim),
        )

    # ---- helpers ----
    def _gauss(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        eps = torch.randn_like(mean)
        return mean, std, mean + eps * std

    def init_state(self, batch_size: int, device: torch.device) -> FactoredState:
        return FactoredState(
            z_mean=torch.zeros(batch_size, self.z_dim, device=device),
            z_std=torch.ones(batch_size, self.z_dim, device=device),
            z_stoch=torch.zeros(batch_size, self.z_dim, device=device),
            u_mean=torch.zeros(batch_size, self.u_dim, device=device),
            u_std=torch.ones(batch_size, self.u_dim, device=device),
            u_stoch=torch.zeros(batch_size, self.u_dim, device=device),
            deter=torch.zeros(batch_size, self.deter_size, device=device),
        )

    def feature_size(self) -> int:
        return self.z_dim + self.u_dim + self.deter_size

    def get_feature(self, s: FactoredState) -> torch.Tensor:
        return torch.cat([s.z_stoch, s.u_stoch, s.deter], dim=-1)

    def get_controllable_feature(self, s: FactoredState) -> torch.Tensor:
        # Reward / discount heads condition on (z, h) only.
        return torch.cat([s.z_stoch, s.deter], dim=-1)

    def controllable_size(self) -> int:
        return self.z_dim + self.deter_size

    def get_z_dist(self, s: FactoredState) -> td.Distribution:
        return td.Independent(td.Normal(s.z_mean, s.z_std), 1)

    def get_u_dist(self, s: FactoredState) -> td.Distribution:
        return td.Independent(td.Normal(s.u_mean, s.u_std), 1)

    # ---- transition ----
    def imagine(
        self, prev_action: torch.Tensor, prev: FactoredState, nonterm: float = 1.0
    ) -> FactoredState:
        sa = self.fc_embed_state_action(
            torch.cat([prev.z_stoch * nonterm, prev.u_stoch * nonterm, prev_action], dim=-1)
        )
        deter = self.rnn(sa, prev.deter * nonterm)
        z_mean, z_std, z_stoch = self._gauss(self.fc_prior_z(torch.cat([deter, prev_action], dim=-1)))
        u_mean, u_std, u_stoch = self._gauss(self.fc_prior_u(deter))
        return FactoredState(z_mean, z_std, z_stoch, u_mean, u_std, u_stoch, deter)

    def observe(
        self,
        obs_embed: torch.Tensor,
        prev_action: torch.Tensor,
        prev_nonterm: torch.Tensor,
        prev: FactoredState,
    ) -> Tuple[FactoredState, FactoredState]:
        prior = self.imagine(prev_action, prev, prev_nonterm)
        x_z = torch.cat([prior.deter, obs_embed, prev_action], dim=-1)
        x_u = torch.cat([prior.deter, obs_embed], dim=-1)
        z_mean, z_std, z_stoch = self._gauss(self.fc_post_z(x_z))
        u_mean, u_std, u_stoch = self._gauss(self.fc_post_u(x_u))
        post = FactoredState(z_mean, z_std, z_stoch, u_mean, u_std, u_stoch, prior.deter)
        return prior, post

    # ---- rollouts ----
    def rollout_observation(
        self,
        seq_len: int,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        nonterms: torch.Tensor,
        prev: FactoredState,
    ) -> Tuple[FactoredState, FactoredState]:
        priors_fields = {f: [] for f in FactoredState._fields}
        posts_fields = {f: [] for f in FactoredState._fields}
        for t in range(seq_len):
            prev_action = action[t] * nonterms[t]
            prior, post = self.observe(obs_embed[t], prev_action, nonterms[t], prev)
            for f in FactoredState._fields:
                priors_fields[f].append(getattr(prior, f))
                posts_fields[f].append(getattr(post, f))
            prev = post
        prior_seq = FactoredState(**{f: torch.stack(v, dim=0) for f, v in priors_fields.items()})
        post_seq = FactoredState(**{f: torch.stack(v, dim=0) for f, v in posts_fields.items()})
        return prior_seq, post_seq

    def rollout_imagination(
        self, horizon: int, actor: nn.Module, prev: FactoredState
    ) -> Tuple[FactoredState, torch.Tensor, torch.Tensor]:
        states = {f: [] for f in FactoredState._fields}
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        state = prev
        for _ in range(horizon):
            feat = self.get_feature(state).detach()
            action, dist = actor(feat)
            state = self.imagine(action, state)
            for f in FactoredState._fields:
                states[f].append(getattr(state, f))
            entropies.append(dist.entropy())
            log_probs.append(dist.log_prob(torch.round(action.detach())))
        state_seq = FactoredState(**{f: torch.stack(v, dim=0) for f, v in states.items()})
        return state_seq, torch.stack(log_probs, 0), torch.stack(entropies, 0)

    def seq_to_batch(self, s: FactoredState, batch_size: int, seq_len: int) -> FactoredState:
        return FactoredState(**{
            f: _seq_to_batch(getattr(s, f), batch_size, seq_len)
            for f in FactoredState._fields
        })


# Backwards-compatible aliases used by callers expecting the old names.
RSSM = FactoredRSSM
RSSMState = FactoredState
