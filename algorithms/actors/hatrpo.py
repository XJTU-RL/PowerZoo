# -*- coding: utf-8 -*-
"""
@File      : hatrpo.py
@Time      : 2025-04-08 17:46
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此文件实现了 HATRPO（Hindsight Actor Trust Region Policy Optimization）算法，用于优化策略网络。
- 关键组件及职责：
  - `HATRPO` 类：继承自 `OnPolicyBase`，负责实现 HATRPO 算法。
    - `__init__` 方法：初始化 HATRPO 算法，接收参数、观测空间、动作空间和设备信息。
    - `update` 方法：更新演员网络，输入数据批次，输出 KL 散度、损失改进等信息。
    - `train` 方法：使用小批量梯度下降进行训练更新，输入演员缓冲区、优势和状态类型，输出训练信息。
- 工作流程：
  1. 初始化算法，设置 KL 阈值、线搜索步骤等参数。
  2. 在 `update` 方法中，计算动作概率、损失和梯度。
  3. 使用共轭梯度法求解步长方向。
  4. 进行回溯线搜索，更新参数并检查更新效果。
  5. 在 `train` 方法中，根据不同策略类型生成数据生成器，多次调用 `update` 方法更新网络。
- 依赖库：`numpy`、`torch`，以及自定义的工具模块。
"""
"""HATRPO algorithm."""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)
from utils.envs_tools import check
from utils.trpo_util import (
    flat_grad,
    flat_params,
    conjugate_gradient,
    fisher_vector_product,
    update_model,
    kl_divergence,
)
from algorithms.actors.on_policy_base import OnPolicyBase
from models.policy_models.stochastic_policy import StochasticPolicy


class HATRPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HATRPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        assert (
            act_space.__class__.__name__ != "MultiDiscrete"
        ), "only continuous and discrete action space is supported by HATRPO."
        super(HATRPO, self).__init__(args, obs_space, act_space, device)

        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]

    def update(self, sample):
        """Update actor networks.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            kl: (torch.Tensor) KL divergence between old and new policy.
            loss_improve: (np.float32) loss improvement.
            expected_improve: (np.ndarray) expected loss improvement.
            dist_entropy: (torch.Tensor) action entropies.
            ratio: (torch.Tensor) ratio between new and old policy.
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
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        ratio = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        if self.use_policy_active_masks:
            loss = (
                torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            loss = torch.sum(
                ratio * factor_batch * adv_targ, dim=-1, keepdim=True
            ).mean()

        loss_grad = torch.autograd.grad(
            loss, self.actor.parameters(), allow_unused=True
        )
        loss_grad = flat_grad(loss_grad)

        step_dir = conjugate_gradient(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            loss_grad.data,
            nsteps=10,
            device=self.device,
        )

        loss = loss.data.cpu().numpy()

        params = flat_params(self.actor)
        fvp = fisher_vector_product(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            step_dir,
        )
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        full_step = step_size * step_dir

        old_actor = StochasticPolicy(
            self.args, self.obs_space, self.act_space, self.device
        )
        update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()

        # Backtracking line search (https://en.wikipedia.org/wiki/Backtracking_line_search)
        flag = False
        fraction = 1
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            update_model(self.actor, new_params)
            action_log_probs, dist_entropy, _ = self.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )

            ratio = getattr(torch, self.action_aggregation)(
                torch.exp(action_log_probs - old_action_log_probs_batch),
                dim=-1,
                keepdim=True,
            )
            if self.use_policy_active_masks:
                new_loss = (
                    torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True)
                    * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                new_loss = torch.sum(
                    ratio * factor_batch * adv_targ, dim=-1, keepdim=True
                ).mean()

            new_loss = new_loss.data.cpu().numpy()
            loss_improve = new_loss - loss

            kl = kl_divergence(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                new_actor=self.actor,
                old_actor=old_actor,
            )
            kl = kl.mean()

            if (
                kl < self.kl_threshold
                and (loss_improve / expected_improve) > self.accept_ratio
                and loss_improve.item() > 0
            ):
                flag = True
                break
            expected_improve *= self.backtrack_coeff
            fraction *= self.backtrack_coeff

        if not flag:
            params = flat_params(old_actor)
            update_model(self.actor, params)
            logger.warning("HATRPO policy update rollback: surrogate loss did not improve after line search")

        return kl, loss_improve, expected_improve, dist_entropy, ratio

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, 1, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages, 1)
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, 1)

        for sample in data_generator:
            kl, loss_improve, expected_improve, dist_entropy, imp_weights = self.update(
                sample
            )

            train_info["kl"] += kl
            train_info["loss_improve"] += loss_improve.item()
            train_info["expected_improve"] += expected_improve
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["ratio"] += imp_weights.mean()

        num_updates = 1

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
