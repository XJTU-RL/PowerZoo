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
            # 每份 active_masks 是 [T+1, N, 1]，stack 后带尾维 1；squeeze 掉
            # 才能与 [T, N, num_agents] 的 advantages 对齐做布尔掩蔽。
            advantages_copy[(active_masks_array[:-1] == 0.0).squeeze(axis=-1)] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        # 训练信息预分配并按实际 agent ID 回填：分组共享时同一网络的信息要
        # 落到它的每个成员的日志位上（组 [0,2]/[1] 应回填 [A,B,A]，而不是
        # 按组顺序拼接成 [A,A,B]）。
        actor_train_infos = [None] * self.num_agents
        if self.share_param == "group":
            # 分组共享：每组用自己的代表网络在组内成员的 buffer 上更新；
            # 只调 actor[0] 会漏掉其余组的独立网络（审核复现的缺陷）。
            for representative, members in self._shared_groups():
                group_buffers = [self.actor_buffer[i] for i in members]
                if self.state_type == "FP":
                    # FP 优势按 agent 维组织，须按成员索引切出组内优势；
                    # 传全队张量会让 MAPPO.share_param_train 用组内序号切到
                    # 其他 agent 的优势（非连续组 [0,2]/[1] 即串位）。
                    group_advantages = advantages[:, :, members].copy()
                else:
                    # EP 保持全队共享优势语义
                    group_advantages = advantages.copy()
                info = self.actor[representative].share_param_train(
                    group_buffers, group_advantages, len(members), self.state_type
                )
                for member in members:
                    actor_train_infos[member] = info
        elif self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            actor_train_infos = [actor_train_info] * self.num_agents
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
                actor_train_infos[agent_id] = actor_train_info

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
