# -*- coding: utf-8 -*-
"""
@File      : on_policy_ma_runner.py
@Time      : 2025-04-08 17:40
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该文件是用于执行基于策略的多智能体（MA）算法的运行器脚本。
- 导入了 `numpy`、`torch` 库以及 `OnPolicyBaseRunner` 类。
- `OnPolicyMARunner` 类继承自 `OnPolicyBaseRunner`，用于执行基于策略的 MA 算法。
  - `train` 方法：实现了 MAPPO 算法的训练流程。
    - 计算优势函数，会根据 `value_normalizer` 是否存在采用不同计算方式。
    - 若状态类型为 "FP"，对优势函数进行归一化处理。
    - 根据 `share_param` 情况更新智能体的策略网络。
    - 更新批评家网络。
    - 最后返回策略网络和批评家网络的训练信息。
"""
"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
from runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param == "group":
            # 分组共享：每组用自己的代表网络在组内成员的 buffer 上更新；
            # 只调 actor[0] 会漏掉其余组的独立网络（审核复现的缺陷）。
            for representative, members in self._shared_groups():
                group_buffers = [self.actor_buffer[i] for i in members]
                info = self.actor[representative].share_param_train(
                    group_buffers, advantages.copy(), len(members), self.state_type
                )
                actor_train_infos.extend(info for _ in members)
        elif self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )
                actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
