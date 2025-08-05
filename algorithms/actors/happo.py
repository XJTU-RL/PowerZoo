# -*- coding: utf-8 -*-
"""
@File      : happo.py
@Time      : 2025-04-08 17:46
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此文件实现了 HAPPO（一种算法），用于更新和训练智能体的策略网络。
- 关键组件及职责：
  - HAPPO 类：继承自 OnPolicyBase，负责初始化 HAPPO 算法的参数，提供更新和训练功能。
  - __init__ 方法：初始化 HAPPO 算法，接收参数、观测空间、动作空间和设备信息。
  - update 方法：根据样本数据更新 actor 网络，计算策略损失、动作熵、梯度范数和重要性采样权重。
  - train 方法：使用小批量梯度下降法训练 actor 网络，返回训练信息。
- 工作流程：
  1. 在 __init__ 中初始化参数。
  2. 在 update 里从样本解压参数，计算动作对数概率和熵，更新 actor 网络。
  3. 在 train 中使用小批量数据生成器多次更新网络并记录训练信息。
- 依赖库：numpy、torch、torch.nn，以及 utils 模块和 algorithms 模块中的部分工具。
"""
"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from utils.envs_tools import check
from utils.models_tools import get_grad_norm
from algorithms.actors.on_policy_base import OnPolicyBase


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """初始化 HAPPO 算法。
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample #从样本中解压出来各个参数

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # 重塑以在一次前向传递中对所有步骤进行评估
        #dist_entropy表示动作的熵，训练完成后，这个值也趋近于0，表示智能体的动作逐渐趋于稳定
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        # HATRPO 文章中公式 4 的实现
        surr1 = imp_weights * adv_targ 
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)   # 截断 weight,将其限制在 [1-e,1+e]
            * adv_targ  
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss#每个actor都有一个L函数

        self.actor_optimizer.zero_grad() # 清空优化器梯度

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())#训练完成以后，梯度接近0，认为已经是最优参数了，这是nash均衡指标的一个体现

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """使用小批量 GD 执行训练更新。 使用梯度下降法训练 minibatch
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0 #新老策略的比
        # 检查 actor_buffer.active_masks 数组中的所有元素是否都为零。如果是这样，函数将提前返回，并且返回 train_info
        if np.all(actor_buffer.active_masks[:-1] == 0.0):                                        
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan #将所有等于 0 的值设置成 nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy) # 计算 std 除去 nan 值
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
