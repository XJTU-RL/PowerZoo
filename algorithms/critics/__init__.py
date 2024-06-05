"""Critic registry."""
from algorithms.critics.v_critic import VCritic
from algorithms.critics.continuous_q_critic import ContinuousQCritic
from algorithms.critics.twin_continuous_q_critic import TwinContinuousQCritic
from algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from algorithms.critics.discrete_q_critic import DiscreteQCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hatrpo": VCritic,
    "haa2c": VCritic,
    "mappo": VCritic,
    "haddpg": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hasac": SoftTwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "matd3": TwinContinuousQCritic,
}
