# -*- coding: utf-8 -*-
"""
@File      : on_policy_ha_runner.py
@Time      : 2025-04-08 17:41
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此文件为基于策略的 HA 算法运行器，继承自 OnPolicyBaseRunner，核心功能是训练模型。
- 关键组件及职责：
  - OnPolicyHARunner 类：继承 OnPolicyBaseRunner，负责训练模型。
  - train 方法：具体实现训练逻辑。
- 工作流程：
  1. 初始化因子 factor。
  2. 计算优势值 advantages。
  3. 若状态类型为 FP，对优势值进行归一化。
  4. 若 useS 为 True，对信息进行汇总、排序。
  5. 根据 ordered 和 useS 确定 agent 顺序。
  6. 按顺序更新每个 agent 的 actor 网络，并更新因子。
  7. 更新 critic 网络。
  8. 返回 actor 和 critic 的训练信息。
- 依赖库：numpy、torch，工具模块 utils.trans_tools，基类模块 runners.on_policy_base_runner。
"""
"""Runner for on-policy PowerZoo algorithms."""
import numpy as np
import torch
from utils.trans_tools import _t2n
from runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""



    def train(self):
        """Train the model."""
        actor_train_infos = []

        # factor is used for considering updates made by previous agents 
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"], # 记录每个episode
                self.algo_args["train"]["n_rollout_threads"], # 记录每个用于收集环境信息的thread
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
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
        
        if hasattr(self, 'useS') and self.useS:
           result = {}
           for step_data in self.critic_buffer.infos.values():
        # 遍历每个步骤中的字典
               for item in step_data:
            # 遍历字典中的键值对
                   for key, value in item.items():
                # 分割键名，获取标签后面的数字
                       label, _ = key.split('.')
                    # 将标签后面的数字加入结果字典中
                       full_label = f"{label}.{_}"
                    # 将标签后面的数字加入结果字典中
                       result[full_label] = result.get(full_label, 0) + value
        #print(result)
        #print(self.get_ordered_agents_pairs)#初始编号信息，在此对各类信息进行汇总
        #print(self.get_agents_bus)
        
        # 只有当 get_agents_bus 存在时才进行智能体排序
        if hasattr(self, 'get_agents_bus') and self.get_agents_bus is not None:
            new_dict = {}
            # 遍历第一个字典
            for key, value in self.get_agents_bus.items():
                total_value = 0
                # 遍历第一个字典中的值
                for item in value:
                    # 如果值在第二个字典中，则累加对应的值
                    if item in result:
                        total_value += result[item]
                # 构建新的字典
                new_dict[key] = total_value

            # 计算智能体排序
            agent_order = self._get_agent_order(new_dict)
            print(f"{self._get_order_type()}_order: {agent_order}")
        else:
            # 当不使用智能体排序时，使用默认顺序
            agent_order = list(range(self.num_agents))
            print(f"default_order: {agent_order}")
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def _get_agent_order(self, new_dict):
        """计算智能体排序顺序。
        
        Args:
            new_dict: 包含智能体敏感度信息的字典
            
        Returns:
            list: 智能体排序列表
        """
        if self.ordered:
            if hasattr(self, 'useS') and self.useS:
                # 按敏感度排序
                sorted_keys = sorted(self.get_ordered_agents_pairs, key=lambda x: new_dict[x])
                sorted_values = [self.get_ordered_agents_pairs[key] for key in sorted_keys]
                
                if hasattr(self, 'big2small') and self.big2small:
                    return sorted_values  # 从大到小按照S排序
                else:
                    return sorted_values[::-1]  # 从小到大按照S排序
            else:
                # 固定顺序
                return list(range(self.num_agents))
        else:
            # 随机顺序
            return list(torch.randperm(self.num_agents).numpy())
    
    def _get_order_type(self):
        """获取排序类型描述。
        
        Returns:
            str: 排序类型描述
        """
        if self.ordered:
            if hasattr(self, 'useS') and self.useS:
                if hasattr(self, 'big2small') and self.big2small:
                    return "按敏感度从大到小排序"  # 根据敏感度值降序排列智能体
                else:
                    return "按敏感度从小到大排序"  # 根据敏感度值升序排列智能体
            else:
                return "固定顺序排序"  # 使用预定义的固定智能体顺序
        else:
            return "随机顺序排序"  # 随机打乱智能体顺序
