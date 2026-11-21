"""Centralized action-value heads for COMA over heterogeneous discrete devices."""

import torch
from torch import nn
from torch.nn import functional as F

from models.base.mlp import MLPBase


class COMACritic(nn.Module):
    def __init__(self, model_args, state_dim, action_sizes):
        super().__init__()
        self.action_sizes = action_sizes
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    MLPBase(model_args, (state_dim + sum(action_sizes) - count,)),
                    nn.Linear(model_args["hidden_sizes"][-1], count),
                )
                for count in action_sizes
            ]
        )

    def forward(self, state, actions):
        one_hot = [
            F.one_hot(actions[:, i], count).to(state.dtype)
            for i, count in enumerate(self.action_sizes)
        ]
        return [
            head(torch.cat([state, *one_hot[:i], *one_hot[i + 1 :]], dim=1))
            for i, head in enumerate(self.heads)
        ]


def counterfactual_advantage(q_values, chosen, probabilities, available):
    """Compare the executed action to the actor's masked counterfactual baseline."""
    legal = available.to(dtype=torch.bool)
    if not legal.gather(1, chosen).all():
        raise ValueError("COMA received an unavailable action")
    weights = probabilities.masked_fill(~legal, 0)
    normalizer = weights.sum(dim=1, keepdim=True)
    if not (normalizer > 0).all():
        raise ValueError("COMA needs a positive probability for the available actions")
    baseline = (weights * q_values.masked_fill(~legal, 0)).sum(dim=1, keepdim=True)
    return q_values.gather(1, chosen) - baseline / normalizer


def lambda_targets(rewards, continuation, next_q, gamma, td_lambda, truncations=None):
    targets = torch.empty_like(next_q)
    future = next_q[-1]
    for t in reversed(range(rewards.shape[0])):
        future = rewards[t] + gamma * continuation[t] * (
            (1 - td_lambda) * next_q[t] + td_lambda * future
        )
        if truncations is not None:
            future = torch.where(truncations[t], rewards[t] + gamma * next_q[t], future)
        targets[t] = future
    return targets
