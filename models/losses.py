"""
Loss functions for FOE-Dreamer.

Three blocks aligned with the manuscript:

  1. Factored ELBO: reconstruction + reward + discount + separate
     KL terms KL_z (beta_z) and KL_u (beta_u).
  2. Opponent-prediction loss: cross-entropy on (action, target) heads
     over the tau embedding produced by the sliding-window encoder.
  3. Actor-critic loss: lambda-returns over imagined trajectories.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from .world_model import FactoredRSSM, FactoredState


# ---------------------------------------------------------------------------
# Factored world-model ELBO
# ---------------------------------------------------------------------------


def _kl_balanced(prior_dist, post_dist, post_dist_detached, prior_dist_detached,
                 free_nats: float, kl_balance: float) -> torch.Tensor:
    kl_lhs = td.kl.kl_divergence(post_dist_detached, prior_dist)
    kl_rhs = td.kl.kl_divergence(post_dist, prior_dist_detached)
    kl_lhs = torch.clamp(kl_lhs, min=free_nats)
    kl_rhs = torch.clamp(kl_rhs, min=free_nats)
    return kl_balance * kl_lhs.mean() + (1.0 - kl_balance) * kl_rhs.mean()


def factored_kl(
    rssm: FactoredRSSM,
    prior: FactoredState,
    posterior: FactoredState,
    free_nats: float = 1.0,
    kl_balance: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (kl_z, kl_u) under KL balancing."""
    # z
    post_z = rssm.get_z_dist(posterior)
    prior_z = rssm.get_z_dist(prior)
    post_z_det = td.Independent(td.Normal(posterior.z_mean.detach(), posterior.z_std.detach()), 1)
    prior_z_det = td.Independent(td.Normal(prior.z_mean.detach(), prior.z_std.detach()), 1)
    kl_z = _kl_balanced(prior_z, post_z, post_z_det, prior_z_det, free_nats, kl_balance)
    # u
    post_u = rssm.get_u_dist(posterior)
    prior_u = rssm.get_u_dist(prior)
    post_u_det = td.Independent(td.Normal(posterior.u_mean.detach(), posterior.u_std.detach()), 1)
    prior_u_det = td.Independent(td.Normal(prior.u_mean.detach(), prior.u_std.detach()), 1)
    kl_u = _kl_balanced(prior_u, post_u, post_u_det, prior_u_det, free_nats, kl_balance)
    return kl_z, kl_u


def world_model_loss(
    rssm: FactoredRSSM,
    obs_decoder: nn.Module,
    reward_head: nn.Module,
    discount_head: Optional[nn.Module],
    prior: FactoredState,
    posterior: FactoredState,
    obs: torch.Tensor,
    reward: torch.Tensor,
    nonterm: torch.Tensor,
    beta_z: float = 1.0,
    beta_u: float = 0.5,
    discount_scale: float = 5.0,
    free_nats: float = 1.0,
    kl_balance: float = 0.8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    feat = rssm.get_feature(posterior)                   # decoder uses (z, u, h)
    ctrl_feat = rssm.get_controllable_feature(posterior) # reward head uses (z, h) only

    recon_dist = obs_decoder(feat)
    recon_loss = -recon_dist.log_prob(obs).mean()
    reward_dist = reward_head(ctrl_feat)
    reward_loss = -reward_dist.log_prob(reward).mean()

    if discount_head is not None:
        disc_dist = discount_head(feat)
        disc_loss = -disc_dist.log_prob(nonterm).mean() * discount_scale
    else:
        disc_loss = torch.zeros((), device=obs.device)

    kl_z, kl_u = factored_kl(rssm, prior, posterior, free_nats=free_nats, kl_balance=kl_balance)
    total = recon_loss + reward_loss + disc_loss + beta_z * kl_z + beta_u * kl_u
    log = {
        "wm/total": float(total.detach()),
        "wm/recon": float(recon_loss.detach()),
        "wm/reward": float(reward_loss.detach()),
        "wm/discount": float(disc_loss.detach()) if discount_head is not None else 0.0,
        "wm/kl_z": float(kl_z.detach()),
        "wm/kl_u": float(kl_u.detach()),
    }
    return total, log


# ---------------------------------------------------------------------------
# Opponent-prediction loss
# ---------------------------------------------------------------------------


def opponent_prediction_loss(
    opponent_model: nn.Module,                 # OpponentModel
    obs_seq: torch.Tensor,                     # [T, B, obs_dim]
    action_seq: torch.Tensor,                  # [T, B, action_dim]
    opp_actions: torch.Tensor,                 # [T, B] long
    opp_targets: torch.Tensor,                 # [T, B] long
    K_window: int,
    ignore_index: int = -1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Slides a window of length K over the sequence and predicts opponent
    (action, target) one step ahead from the tau embedding at each
    valid window position.
    """
    T, B, _ = obs_seq.shape
    if T < K_window + 1:
        zero = obs_seq.new_zeros(())
        return zero, {"foe/total": 0.0, "foe/action": 0.0, "foe/target": 0.0}

    a_logits_list, t_logits_list, a_targ_list, t_targ_list = [], [], [], []
    for t in range(K_window, T):
        obs_win = obs_seq[t - K_window:t]            # [K, B, obs]
        act_win = action_seq[t - K_window:t]         # [K, B, action]
        tau = opponent_model.encode(obs_win, act_win)  # [B, tau]
        a_logits, t_logits = opponent_model.predict(tau)
        a_logits_list.append(a_logits)
        t_logits_list.append(t_logits)
        a_targ_list.append(opp_actions[t])
        t_targ_list.append(opp_targets[t])

    a_logits = torch.cat(a_logits_list, dim=0)
    t_logits = torch.cat(t_logits_list, dim=0)
    a_targ = torch.cat(a_targ_list, dim=0)
    t_targ = torch.cat(t_targ_list, dim=0)

    la = F.cross_entropy(a_logits, a_targ, ignore_index=ignore_index)
    lt = F.cross_entropy(t_logits, t_targ, ignore_index=ignore_index)
    total = la + lt
    return total, {
        "foe/total": float(total.detach()),
        "foe/action": float(la.detach()),
        "foe/target": float(lt.detach()),
    }


# ---------------------------------------------------------------------------
# Actor-critic over imagined trajectories
# ---------------------------------------------------------------------------


def lambda_returns(
    rewards: torch.Tensor, values: torch.Tensor, discount: torch.Tensor,
    bootstrap: torch.Tensor, lam: float = 0.95,
) -> torch.Tensor:
    H = rewards.shape[0]
    next_value = bootstrap
    out: List[torch.Tensor] = []
    for t in reversed(range(H)):
        td_target = rewards[t] + discount[t] * (1.0 - lam) * values[t]
        next_value = td_target + discount[t] * lam * next_value
        out.append(next_value)
    out.reverse()
    return torch.stack(out, dim=0)


def actor_critic_loss(
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    rssm: FactoredRSSM,
    reward_head: nn.Module,
    discount_head: Optional[nn.Module],
    starting_state: FactoredState,
    horizon: int,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    states, log_probs, entropies = rssm.rollout_imagination(horizon, actor, starting_state)
    feat = rssm.get_feature(states)
    ctrl_feat = rssm.get_controllable_feature(states)
    rewards = reward_head(ctrl_feat).mean
    if discount_head is not None:
        disc = discount_head(feat).mean * gamma
    else:
        disc = torch.full_like(rewards, gamma)

    with torch.no_grad():
        target_values = target_critic(feat).mean

    bootstrap = target_values[-1]
    returns = lambda_returns(rewards[:-1], target_values[:-1], disc[:-1], bootstrap, lam=lam)

    actor_obj = (returns.detach() * log_probs[:-1]).mean()
    entropy_bonus = entropy_coef * entropies[:-1].mean()
    actor_loss = -(actor_obj + entropy_bonus)

    value_dist = critic(feat[:-1].detach())
    critic_loss = -value_dist.log_prob(returns.detach()).mean()
    log = {
        "ac/actor_obj": float(actor_obj.detach()),
        "ac/entropy": float(entropies[:-1].mean().detach()),
        "ac/return_mean": float(returns.mean().detach()),
        "ac/critic_loss": float(critic_loss.detach()),
        "ac/value_mean": float(target_values.mean().detach()),
    }
    return actor_loss, critic_loss, log
