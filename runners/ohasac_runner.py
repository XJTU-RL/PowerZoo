# -*- coding: utf-8 -*-
"""OHASAC 运行器：HASAC + 次序网络（OrderAgent）。

来源与忠实性说明：
- 训练主体（critic/actor 更新、序贯按序更新、alpha、软更新）逐行取自
  PowerZoo 的 ``OffPolicyHARunner`` hasac 分支（该分支即论文 HARL 实现的
  物理重构版，非 lstm 路径下与论文数值等价）。
- 次序网络的前向/缓冲/评估点 PPO 更新逐行取自论文实现（P0 修复版）的
  ``off_policy_ha_runner.py`` + ``off_policy_base_runner.py``（升序排序——
  产出论文结果的在跑实现）。
- 本类通过**整覆写** ``train()``/``eval()`` 实现扩展，公共 runner 零改动，
  其他算法/案例不受影响。

与论文配置系统的适配：PowerZoo 算法配置固定八节，论文的 order_net 节
参数并入 ``algo`` 节（键名加 ``order_`` 前缀）：
``use_order_net / order_num_steps(9) / order_lr(3e-4) /
order_obs_index(9) / order_obs_scale(10)``。
"""

import numpy as np
import torch
import torch.optim as optim

from algorithms.actors.order_agent import OrderAgent
from runners.off_policy_ha_runner import OffPolicyHARunner


class OHASACRunner(OffPolicyHARunner):
    """HASAC + OrderAgent：按频差动态决定智能体更新次序。"""

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        algo_cfg = self.algo_args["algo"]
        self.use_order_net = algo_cfg.get("use_order_net", False)
        if self.use_order_net:
            self.order_obs_index = algo_cfg.get("order_obs_index", 9)
            self.order_num_steps = algo_cfg.get("order_num_steps", 9)
            self.order_net = OrderAgent(
                self.batch_size,
                self.num_agents,
                num_steps=self.order_num_steps,
                learning_rate=algo_cfg.get("order_lr", 3e-4),
                writer=self.writer if hasattr(self.writer, "add_scalar") else None,
                device=self.device,
            ).to(self.device)
            self.order_optimizer = optim.Adam(
                self.order_net.parameters(),
                lr=algo_cfg.get("order_lr", 3e-4),
                eps=1e-5,
            )
            horizon = self.algo_args["train"]["eval_interval"]
            self.order_obs = torch.zeros(horizon, self.batch_size).to(self.device)
            self.order_actions = torch.zeros(horizon, self.num_agents).to(self.device)
            self.order_logprobs = torch.zeros(horizon).to(self.device)
            self.order_rewards = torch.zeros(horizon).to(self.device)
            self.order_dones = torch.zeros(horizon).to(self.device)
            self.order_values = torch.zeros(horizon).to(self.device)
            self.order_step = 0

    # ------------------------------------------------------------------ #
    # 训练：hasac 分支 + 次序网络前向与按序更新（论文 off_policy_ha_runner） #
    # ------------------------------------------------------------------ #
    def train(self):
        """Train the model（OHASAC：critic → 次序网络前向 → 按序更新 actor）。"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
        ) = data
        # train critic
        self.critic.turn_on_grad()
        next_actions = []
        next_logp_actions = []
        for agent_id in range(self.num_agents):
            next_action, next_logp_action = self.actor[agent_id].get_actions_with_logprobs(
                sp_next_obs[agent_id],
                sp_next_available_actions[agent_id]
                if sp_next_available_actions is not None
                else None,
            )
            next_actions.append(next_action)
            next_logp_actions.append(next_logp_action)
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_valid_transition,
            sp_term,
            sp_next_share_obs,
            next_actions,
            next_logp_actions,
            sp_gamma,
            self.value_normalizer,
        )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)

        # 次序网络前向：频差维度的相邻差分 → 更新次序（论文核心）
        update_order = None
        if self.use_order_net:
            with torch.no_grad():
                order_obs1 = sp_share_obs[:, self.order_obs_index]
                order_obs2 = sp_next_share_obs[:, self.order_obs_index]
                scale = self.algo_args["algo"].get("order_obs_scale", 10)
                order_obs_temp = np.array((order_obs2 - order_obs1) * scale)
                order_obs = (
                    torch.from_numpy(order_obs_temp)
                    .to(self.device)
                    .to(torch.float32)
                    .reshape(1, len(order_obs_temp))
                )
                order_action, order_logprob, _, order_value = (
                    self.order_net.get_action_and_value(order_obs)
                )
                sorted_tensor, sorted_indices = torch.sort(order_action, descending=False)
                update_order = sorted_indices[0].tolist()
                # 缓冲边界显式化（P0-3）：论文时序下 order_step 不会越过缓冲长度
                buf_len = self.order_obs.shape[0]
                if self.order_step < buf_len:
                    self.order_obs[self.order_step] = order_obs
                    self.order_actions[self.order_step] = order_action
                    self.order_logprobs[self.order_step] = order_logprob
                    self.order_values[self.order_step] = order_value
                    self.order_dones[self.order_step] = self.order_step >= buf_len - 1
                self.order_step += 1

        if self.total_it % self.policy_freq == 0:
            # train actors
            actions = []
            logp_actions = []
            with torch.no_grad():
                for agent_id in range(self.num_agents):
                    action, logp_action = self.actor[agent_id].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    actions.append(action)
                    logp_actions.append(logp_action)
            # 更新次序决策（论文）：fixed > order_net > 随机
            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            elif self.use_order_net and update_order is not None:
                agent_order = update_order
            else:
                agent_order = list(np.random.permutation(self.num_agents))
            for agent_id in agent_order:
                self.actor[agent_id].turn_on_grad()
                actions[agent_id], logp_actions[agent_id] = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id]
                    if sp_available_actions is not None
                    else None,
                )
                if self.state_type == "EP":
                    logp_action = logp_actions[agent_id]
                    actions_t = torch.cat(actions, dim=-1)
                elif self.state_type == "FP":
                    logp_action = torch.tile(logp_actions[agent_id], (self.num_agents, 1))
                    actions_t = torch.tile(torch.cat(actions, dim=-1), (self.num_agents, 1))
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                if self.algo_args["algo"]["use_policy_active_masks"]:
                    if self.state_type == "EP":
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha[agent_id] * logp_action)
                                * sp_valid_transition[agent_id]
                            )
                            / sp_valid_transition[agent_id].sum()
                        )
                    elif self.state_type == "FP":
                        valid_transition = torch.tile(
                            sp_valid_transition[agent_id], (self.num_agents, 1)
                        )
                        actor_loss = (
                            -torch.sum(
                                (value_pred - self.alpha[agent_id] * logp_action)
                                * valid_transition
                            )
                            / valid_transition.sum()
                        )
                else:
                    actor_loss = -torch.mean(value_pred - self.alpha[agent_id] * logp_action)
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
                # train this agent's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    log_prob = (
                        logp_actions[agent_id].detach() + self.target_entropy[agent_id]
                    )
                    alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                    self.alpha_optimizer[agent_id].zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer[agent_id].step()
                    self.alpha[agent_id] = torch.exp(self.log_alpha[agent_id].detach())
                actions[agent_id], _ = self.actor[agent_id].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    sp_available_actions[agent_id]
                    if sp_available_actions is not None
                    else None,
                )
            # HASAC actor soft update
            for agent_id in range(self.num_agents):
                if hasattr(self.actor[agent_id], "soft_update"):
                    self.actor[agent_id].soft_update()
            # train critic's alpha
            if self.algo_args["algo"]["auto_alpha"]:
                self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            self.critic.soft_update()

    # ------------------------------------------------------------------ #
    # 评估：底本为 PowerZoo OffPolicyBaseRunner.eval；追加次序网络训练点     #
    # ------------------------------------------------------------------ #
    def eval(self, step):
        """评估并（启用时）训练次序网络——论文把 order net 更新挂在评估分支。"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        episode_lens = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int64
        )

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        while True:
            eval_actions = self.get_actions(
                eval_obs, available_actions=eval_available_actions, add_random=False
            )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)

                print(
                    f"Eval average episode reward is {eval_avg_rew}, "
                    f"eval average episode length is {eval_avg_len}.\n"
                )
                self.log_file.write(
                    ",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n"
                )
                self.log_file.flush()
                self.writer.add_scalar("eval_average_episode_rewards", eval_avg_rew, step)
                self.writer.add_scalar("eval_average_episode_length", eval_avg_len, step)
                break

        # ---- 次序网络训练（论文 off_policy_base_runner 评估分支） ----
        if self.use_order_net:
            horizon = self.order_obs.shape[0]
            self.order_rewards = torch.full((horizon,), float(eval_avg_rew)).to(self.device)
            self.order_net.train(
                self.order_obs,
                self.order_actions,
                self.order_logprobs,
                self.order_values,
                self.order_rewards,
                self.order_dones,
                step,
                self.order_optimizer,
            )
            with torch.no_grad():
                self.order_obs = torch.zeros_like(self.order_obs)
                self.order_actions = torch.zeros_like(self.order_actions)
                self.order_logprobs = torch.zeros_like(self.order_logprobs)
                self.order_rewards = torch.zeros_like(self.order_rewards)
                self.order_dones = torch.zeros_like(self.order_dones)
                self.order_values = torch.zeros_like(self.order_values)
                self.order_step = 0
