"""LDCPPO 网络：高斯 Actor 与 1+3 独立 Critic（论文表 II）。

Actor 隐藏 [128,128]，Gaussian 状态无关 log_std（init -0.5，论文
p.10）；Critic 隐藏 [128,128]。状态由编码器给出（PISTG/STG=80 维嵌入，
flat=平铺窗口 1992 维，对应论文 LDCPPO 基线）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, out_dim, activate_last=False):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    layers.append(nn.Linear(sizes[-1], out_dim))
    if activate_last:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(128, 128),
                 log_std_init: float = -0.5):
        super().__init__()
        self.mean_net = mlp([state_dim, *hidden], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def distribution(self, state: torch.Tensor):
        import torch.distributions as dist

        std = self.log_std.clamp(-3.0, 1.0).exp()
        return dist.Normal(self.mean_net(state), std)

    def forward(self, state):
        normal = self.distribution(state)
        action = normal.sample()
        return action, normal.log_prob(action).sum(-1)

    def evaluate(self, state, action):
        normal = self.distribution(state)
        return (
            normal.log_prob(action).sum(-1),
            normal.entropy().sum(-1),
        )


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden=(128, 128)):
        super().__init__()
        self.net = mlp([state_dim, *hidden], 1)

    def forward(self, state):
        return self.net(state).squeeze(-1)
