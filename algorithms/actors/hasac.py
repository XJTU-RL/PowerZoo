# -*- coding: utf-8 -*-
"""
@File      : hasac.py
@Time      : 2025-04-08 17:46
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此文件实现了 HASAC 算法，用于处理不同类型的动作空间，主要包含一个类和多个方法。
- `HASAC` 类：继承自 `OffPolicyBase`，用于初始化和管理算法相关参数与模型。
  - `__init__` 方法：初始化类的属性，根据动作空间类型选择合适的策略模型，并创建优化器。
  - `get_actions` 方法：根据输入的观测值获取动作，支持随机或确定性动作。
  - `get_actions_with_logprobs` 方法：获取动作及对应的对数概率，支持不同动作空间类型。
  - `save` 方法：将 actor 模型保存到指定路径。
  - `restore` 方法：从指定路径恢复 actor 模型。

该算法依赖 `torch` 库进行深度学习计算，同时使用了自定义的模型和工具模块。
"""
"""HASAC algorithm."""
import torch
from models.policy_models.squashed_gaussian_policy import SquashedGaussianPolicy
from models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from utils.discrete_util import gumbel_softmax
from utils.envs_tools import check
from algorithms.actors.off_policy_base import OffPolicyBase


class HASAC(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        if act_space.__class__.__name__ == "Box":
            self.actor = SquashedGaussianPolicy(args, obs_space, act_space, device)
        else:
            self.actor = StochasticMlpPolicy(args, obs_space, act_space, device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.turn_off_grad()

    def get_actions(self, obs, available_actions=None, stochastic=True):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
        else:
            actions = self.actor(obs, available_actions, stochastic)
        return actions

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        """Get actions and logprobs of actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (batch_size, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            stochastic: (bool) stochastic actions or deterministic actions
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (batch_size, dim)
            logp_actions: (torch.Tensor) log probabilities of actions taken by this actor, shape is (batch_size, 1)
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions, logp_actions = self.actor(
                obs, stochastic=stochastic, with_logprob=True
            )
        elif self.action_type == "Discrete":
            logits = self.actor.get_logits(obs, available_actions)
            actions = gumbel_softmax(
                logits, hard=True, device=self.device
            )  # onehot actions
            # 使用log_softmax计算真正的log概率，而不是直接使用logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            logp_actions = torch.sum(actions * log_probs, dim=-1, keepdim=True)
        elif self.action_type == "MultiDiscrete":
            logits = self.actor.get_logits(obs, available_actions)
            actions = []
            logp_actions = []
            for logit in logits:
                action = gumbel_softmax(
                    logit, hard=True, device=self.device
                )  # onehot actions
                logp_action = torch.sum(action * logit, dim=-1, keepdim=True)
                actions.append(action)
                logp_actions.append(logp_action)
            actions = torch.cat(actions, dim=-1)
            logp_actions = torch.cat(logp_actions, dim=-1)
        return actions, logp_actions

    def save(self, save_dir, id):
        """Save the actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )

    def restore(self, model_dir, id):
        """Restore the actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
