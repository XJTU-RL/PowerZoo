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
        
        if self.useS==True:
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

        #print(new_dict)
           sorted_keys = sorted(self.get_ordered_agents_pairs, key=lambda x: new_dict[x])
        # 提取排序后的字典1中的值作为数组
           sorted_values = [self.get_ordered_agents_pairs[key] for key in sorted_keys]
        #print(sorted_values)
           reversed_array=sorted_values[::-1]
        #print(reversed_array)
        
        if self.ordered:
            # 说明：
            # 分四种顺序：
            # 1. 敏感度顺序排序，有big2small和small2big两种
            # 2. 固定顺序,第一种是所有更新使用一个固定顺序，第二种是每一轮更新都产生一个随机顺序
            # agent_order = list(range(self.num_agents)) #TODO:固定顺序,原始代码
            
            if self.useS==True:
               if self.big2small:# 从大到小按照S排序
                  agent_order=sorted_values
                  print("big2small_S_sorted_order: ",agent_order)
               else: # 小到大按照S排序
                  agent_order=reversed_array
                  print("small2big_S_sorted_order: ",agent_order)
            else: # 所有的顺序是一样的，随机固定顺序
                agent_order = list(range(self.num_agents))
                print("fixed_sorted_agent_order: ",agent_order)
        else: # 每轮更新都纯随机顺序，没有任何规则引导
            agent_order = list(torch.randperm(self.num_agents).numpy()) #TODO:随机顺序
            print("random_sorted_agent_order: ",agent_order)
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

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


