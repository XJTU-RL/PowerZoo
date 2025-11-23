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
    
    def _calculate_agent_order(self):
        """计算智能体的排序顺序。
        
        Returns:
            list: 智能体的排序列表
        """
        # 如果不使用敏感度排序，直接返回默认顺序
        if not (hasattr(self, 'useS') and self.useS):
            agent_order = list(range(self.num_agents))
            print(f"default_order: {agent_order}")
            return agent_order
            
        # 计算敏感度结果字典
        result = {}
        for step_data in self.critic_buffer.infos.values():
            for item in step_data:
                for key, value in item.items():
                    label, _ = key.split('.')
                    full_label = f"{label}.{_}"
                    result[full_label] = result.get(full_label, 0) + value
        
        # 如果没有get_agents_bus属性，返回默认顺序
        if not (hasattr(self, 'get_agents_bus') and self.get_agents_bus is not None):
            agent_order = list(range(self.num_agents))
            print(f"default_order: {agent_order}")
            return agent_order
            
        # 计算每个智能体的总敏感度
        new_dict = {}
        for key, value in self.get_agents_bus.items():
            total_value = sum(result.get(item, 0) for item in value)
            new_dict[key] = total_value

        # 根据敏感度计算智能体排序
        agent_order = self._get_agent_order(new_dict)
        print(f"{self._get_order_type()}_order: {agent_order}")

        # 最终验证agent_order的有效性
        if not isinstance(agent_order, list) or len(agent_order) != self.num_agents:
            print(f"错误：agent_order格式无效: {agent_order}，使用默认顺序")
            agent_order = list(range(self.num_agents))
            
        return agent_order

    def _get_buffer_attribute(self, agent_id, attr_name):
        """获取buffer属性，兼容异构和同构buffer

        Args:
            agent_id: 智能体ID
            attr_name: 属性名称（如'actions', 'available_actions'等）

        Returns:
            对应的属性值（混合动作空间会自动拼接）
        """
        buffer = self.actor_buffer[agent_id]
        attr = getattr(buffer, attr_name, None)

        # 处理混合动作空间的特殊情况
        if attr_name == 'actions' and isinstance(attr, dict):
            # 混合动作空间：拼接所有子动作
            actions_list = []
            for i in sorted(attr.keys()):
                actions_list.append(attr[i])
            # 在最后一个维度拼接
            return np.concatenate(actions_list, axis=-1)

        # 处理混合动作空间的available_actions
        if attr_name == 'available_actions' and isinstance(attr, dict):
            # 返回第一个非None的available_actions
            for i in sorted(attr.keys()):
                if attr[i] is not None:
                    return attr[i]
            return None

        # 非混合动作空间或其他属性，直接返回
        return attr

    def train(self):
        """Train the model."""
        actor_train_infos = []
        
        # HAPPO诊断：设置TensorBoard writer
        if hasattr(self, 'writer') and self.writer is not None:
            from utils import happo_diagnostics
            happo_diagnostics.set_writer(self.writer)
            # 获取全局步数
            if hasattr(self, 'total_num_steps'):
                happo_diagnostics.set_global_step(self.total_num_steps)

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
            
            # HAPPO诊断：保存归一化前的优势值
            advantages_before = advantages.copy()
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            
            # HAPPO诊断：记录优势归一化前后对比
            from utils import happo_diagnostics
            happo_diagnostics.log_advantages_normalization(
                advantages_raw=advantages_before,
                advantages_norm=advantages
            )
        

        # 计算agent排序
        agent_order = self._calculate_agent_order()
        
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            # 获取available_actions
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            # 获取actions数据（兼容异构buffer）
            actions_data = self._get_buffer_attribute(agent_id, 'actions')
            
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id].obs[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].obs.shape[2:]
                ),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(
                    -1, *self.actor_buffer[agent_id].rnn_states.shape[2:]
                ),
                actions_data.reshape(
                    -1, *actions_data.shape[2:]
                ),
                self.actor_buffer[agent_id].masks[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].masks.shape[2:]
                ),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].active_masks.shape[2:]
                ),
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
                self.actor_buffer[agent_id].obs[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].obs.shape[2:]
                ),
                self.actor_buffer[agent_id].rnn_states[0:1].reshape(
                    -1, *self.actor_buffer[agent_id].rnn_states.shape[2:]
                ),
                actions_data.reshape(
                    -1, *actions_data.shape[2:]
                ),
                self.actor_buffer[agent_id].masks[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].masks.shape[2:]
                ),
                available_actions,
                self.actor_buffer[agent_id].active_masks[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].active_masks.shape[2:]
                ),
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
        
        # 添加详细的tensorboard日志记录
        self._log_training_metrics(actor_train_infos, critic_train_info, advantages, factor)


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
                # 防御性编程：检查get_ordered_agents_pairs是否存在且有效
                # 注意：self.get_ordered_agents_pairs 是从 get_ordered_agents_pairs 函数返回的值
                if (hasattr(self, 'get_ordered_agents_pairs') and 
                    self.get_ordered_agents_pairs is not None and 
                    isinstance(self.get_ordered_agents_pairs, dict)):
                    
                    try:
                        # 按敏感度排序
                        sorted_keys = sorted(self.get_ordered_agents_pairs.keys(), key=lambda x: new_dict.get(x, 0))
                        sorted_values = [self.get_ordered_agents_pairs[key] for key in sorted_keys]
                        
                        # 验证所有agent_id都是整数
                        validated_values = []
                        for agent_id in sorted_values:
                            if isinstance(agent_id, int) and 0 <= agent_id < self.num_agents:
                                validated_values.append(agent_id)
                            else:
                                print(f"警告：检测到无效的agent_id: {agent_id}，跳过并使用默认顺序")
                                return list(range(self.num_agents))
                        
                        if hasattr(self, 'big2small') and self.big2small:
                            return validated_values  # 从大到小按照S排序
                        else:
                            return validated_values[::-1]  # 从小到大按照S排序
                            
                    except (KeyError, TypeError, AttributeError) as e:
                        print(f"警告：智能体排序过程中出现错误: {e}，使用默认顺序")
                        return list(range(self.num_agents))
                else:
                    print("警告：get_ordered_agents_pairs不可用，使用默认顺序")
                    return list(range(self.num_agents))
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
    
    def _log_training_metrics(self, actor_train_infos, critic_train_info, advantages, factor):
        """记录详细的训练指标到tensorboard
        
        Args:
            actor_train_infos: 各智能体的actor训练信息列表
            critic_train_info: critic训练信息
            advantages: 优势函数值
            factor: 更新因子
        """
        try:
            # 获取当前总步数
            total_steps = self.total_num_steps if hasattr(self, 'total_num_steps') else 0
            
            # 1. 记录每个智能体的详细指标
            for agent_id, info in enumerate(actor_train_infos):
                agent_prefix = f"agent_{agent_id}"
                
                # Actor损失相关
                if 'policy_loss' in info:
                    self.writer.add_scalar(f"{agent_prefix}/policy_loss", info['policy_loss'], total_steps)
                if 'dist_entropy' in info:
                    self.writer.add_scalar(f"{agent_prefix}/entropy", info['dist_entropy'], total_steps)
                if 'actor_grad_norm' in info:
                    self.writer.add_scalar(f"{agent_prefix}/actor_grad_norm", info['actor_grad_norm'], total_steps)
                if 'ratio' in info:
                    self.writer.add_scalar(f"{agent_prefix}/importance_ratio", info['ratio'], total_steps)
                if 'approx_kl' in info:
                    self.writer.add_scalar(f"{agent_prefix}/approx_kl", info['approx_kl'], total_steps)
                if 'clipfrac' in info:
                    self.writer.add_scalar(f"{agent_prefix}/clip_fraction", info['clipfrac'], total_steps)
                
                # 动作分布统计
                buffer = self.actor_buffer[agent_id]
                if hasattr(buffer, 'actions'):
                    actions = self._get_buffer_attribute(agent_id, 'actions')
                    if actions is not None:
                        # 记录动作分布
                        if hasattr(buffer, 'action_type'):
                            if buffer.action_type == 'discrete':
                                # 离散动作：记录动作频率
                                action_counts = np.bincount(actions.flatten().astype(int))
                                for action_idx, count in enumerate(action_counts):
                                    self.writer.add_scalar(
                                        f"{agent_prefix}/action_freq/action_{action_idx}", 
                                        count / len(actions.flatten()), 
                                        total_steps
                                    )
                            elif buffer.action_type == 'continuous':
                                # 连续动作：记录均值和标准差
                                self.writer.add_scalar(
                                    f"{agent_prefix}/action_mean", 
                                    np.mean(actions), 
                                    total_steps
                                )
                                self.writer.add_scalar(
                                    f"{agent_prefix}/action_std", 
                                    np.std(actions), 
                                    total_steps
                                )
                                # 对于PV智能体，记录两个控制维度
                                if actions.shape[-1] == 2:
                                    self.writer.add_scalar(
                                        f"{agent_prefix}/pv_active_power", 
                                        np.mean(actions[..., 0]), 
                                        total_steps
                                    )
                                    self.writer.add_scalar(
                                        f"{agent_prefix}/pv_power_factor", 
                                        np.mean(actions[..., 1]), 
                                        total_steps
                                    )
            
            # 2. 记录Critic相关指标
            if 'value_loss' in critic_train_info:
                self.writer.add_scalar("critic/value_loss", critic_train_info['value_loss'], total_steps)
            if 'critic_grad_norm' in critic_train_info:
                self.writer.add_scalar("critic/grad_norm", critic_train_info['critic_grad_norm'], total_steps)
            
            # 3. 记录优势函数统计
            self.writer.add_scalar("training/advantages_mean", np.mean(advantages), total_steps)
            self.writer.add_scalar("training/advantages_std", np.std(advantages), total_steps)
            self.writer.add_scalar("training/advantages_max", np.max(advantages), total_steps)
            self.writer.add_scalar("training/advantages_min", np.min(advantages), total_steps)
            
            # 4. 记录更新因子
            self.writer.add_scalar("training/factor_mean", np.mean(factor), total_steps)
            self.writer.add_scalar("training/factor_std", np.std(factor), total_steps)
            
            # 5. 记录价值函数预测质量
            if hasattr(self.critic_buffer, 'value_preds') and hasattr(self.critic_buffer, 'returns'):
                value_preds = self.critic_buffer.value_preds[:-1]
                returns = self.critic_buffer.returns[:-1]
                # 计算解释方差
                if self.value_normalizer is not None:
                    value_preds_denorm = self.value_normalizer.denormalize(value_preds)
                else:
                    value_preds_denorm = value_preds
                
                var_y = np.var(returns)
                explained_var = np.nan if var_y == 0 else 1 - np.var(returns - value_preds_denorm) / var_y
                self.writer.add_scalar("critic/explained_variance", explained_var, total_steps)
                
                # HAPPO诊断：解释方差详细分解
                from utils import happo_diagnostics
                happo_diagnostics.log_explained_variance(
                    y_true=returns,
                    y_pred=value_preds_denorm
                )
                self.writer.add_scalar("critic/value_pred_mean", np.mean(value_preds), total_steps)
                self.writer.add_scalar("critic/returns_mean", np.mean(returns), total_steps)
            
            # 6. 环境相关指标（如果有system_logger）
            if hasattr(self, 'system_logger') and self.system_logger is not None:
                realtime_metrics = self.system_logger.get_realtime_metrics()
                if realtime_metrics:
                    self.writer.add_scalar("env/recent_avg_reward", realtime_metrics.get('recent_avg_reward', 0), total_steps)
                    self.writer.add_scalar("env/voltage_violations", realtime_metrics.get('recent_voltage_violations', 0), total_steps)
            
            # 7. 记录学习率（如果可用）
            for agent_id in range(self.num_agents):
                if hasattr(self.actor[agent_id], 'optimizer'):
                    for param_group in self.actor[agent_id].optimizer.param_groups:
                        self.writer.add_scalar(f"agent_{agent_id}/learning_rate", param_group['lr'], total_steps)
                        break
            
            if hasattr(self.critic, 'optimizer'):
                for param_group in self.critic.optimizer.param_groups:
                    self.writer.add_scalar("critic/learning_rate", param_group['lr'], total_steps)
                    break
            
            # 8. 记录智能体排序信息（如果使用敏感度排序）
            if hasattr(self, 'useS') and self.useS and hasattr(self, 'get_agents_bus'):
                # 这里可以记录智能体的敏感度值
                pass
            
            # 确保数据写入
            self.writer.flush()
            
        except Exception as e:
            print(f"记录训练指标时出错: {e}")
            import traceback
            traceback.print_exc()
