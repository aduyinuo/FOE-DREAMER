"""FOE-Dreamer model components."""
from .world_model import FactoredRSSM, FactoredState, RSSM, RSSMState
from .opponent_model import (
    OpponentModel, OpponentWindow, SlidingWindowEncoder, OpponentDecoder,
)
from .heads import (
    Critic, DiscountHead, DiscreteActor, MLPDecoder, MLPEncoder, RewardHead,
    hard_update, soft_update, trainable_params,
)
from .losses import (
    actor_critic_loss, factored_kl, lambda_returns,
    opponent_prediction_loss, world_model_loss,
)

__all__ = [
    "FactoredRSSM", "FactoredState", "RSSM", "RSSMState",
    "OpponentModel", "OpponentWindow", "SlidingWindowEncoder", "OpponentDecoder",
    "Critic", "DiscountHead", "DiscreteActor", "MLPDecoder", "MLPEncoder",
    "RewardHead", "hard_update", "soft_update", "trainable_params",
    "actor_critic_loss", "factored_kl", "lambda_returns",
    "opponent_prediction_loss", "world_model_loss",
]
