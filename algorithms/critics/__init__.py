# -*- coding: utf-8 -*-
"""
@File      : __init__.py
@Time      : 2025-04-08 17:45
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该文件是一个评论家（Critic）注册表，用于管理不同算法对应的评论家类。
- 从 `algorithms.critics` 模块导入了多个评论家类，包括 `VCritic`、`ContinuousQCritic` 等。
- `CRITIC_REGISTRY` 是一个字典，将算法名称映射到对应的评论家类：
  - "happo"、"hatrpo"、"haa2c"、"mappo" 对应 `VCritic`。
  - "haddpg"、"maddpg" 对应 `ContinuousQCritic`。
  - "hatd3"、"matd3" 对应 `TwinContinuousQCritic`。
  - "hasac" 对应 `SoftTwinContinuousQCritic`。
  - "had3qn" 对应 `DiscreteQCritic`。

此注册表可根据算法名称方便地获取对应的评论家类，为算法的使用提供了统一的接口。
"""
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
    "shom": VCritic,  # SHOM 使用相同的 V-Critic
    "haddpg": ContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hasac": SoftTwinContinuousQCritic,
    "had3qn": DiscreteQCritic,
    "maddpg": ContinuousQCritic,
    "matd3": TwinContinuousQCritic,
}
