# -*- coding: utf-8 -*-
"""OHASAC 次序网络（OrderAgent）。

来源：OHASAC 论文实现（papers/ohasac/code，P0 修复版 2026-11-11）的
``harl/algorithms/actors/ppo_hasac.py``，原文件为 CleanRL 风格 PPO 脚本，
已按 P0-1/2/3 完成去命令行化与切片显式化后移植至此。

职责：输入频率偏差特征（共享观测中 gsp 频差维度的相邻时刻差分），
输出各智能体的次序得分；升序排序即为本轮 actor 更新次序（OHASAC 的 "O"）。
每个评估点由 runner 用缓冲的 (obs, 得分, logp, value) 与 eval 回合总奖励
做一次 num_steps 步的 PPO 更新（GAE + clip + value clip）。
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class OrderAgent(nn.Module):
    """估计异构智能体更新次序的 PPO 风格网络。

    Args:
        obs_shape / action_shape: 输入/输出维（runner 传入 batch_size 与 num_agents）。
        num_steps: 每次更新消费的时间步数（off-policy 侧 9，on-policy 侧 5）。
        exp_tag / seed_tag: tensorboard run 名组成（保持论文 run_name 格式）。
        writer: 可注入的 SummaryWriter；缺省在 ``runs_dir`` 下自建（论文原行为）。
    """

    def __init__(
        self,
        obs_shape,
        action_shape,
        num_steps=9,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        update_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.0,
        vf_coef=0.5,
        target_kl=None,
        norm_adv=True,
        exp_tag="ppo_hasac",
        seed_tag=1,
        runs_dir="runs",
        writer=None,
        device=None,
    ):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(int(np.array(obs_shape).prod()), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(int(np.array(obs_shape).prod()), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, int(np.prod(action_shape))), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, int(np.prod(action_shape))))

        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.batch_size = num_steps

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        run_name = f"lfc__{exp_tag}__{seed_tag}__{int(time.time())}"
        if writer is not None:
            self.writer = writer
        else:
            self.writer = SummaryWriter(os.path.abspath(os.path.join(runs_dir, run_name)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """返回 (次序得分, logprob, entropy, value)。"""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def train(
        self,
        obs_batch,
        action_batch,
        logprob_batch,
        value_batch,
        reward_batch,
        dones_batch,
        global_step,
        optimizer,
    ):
        """用最近 num_steps 步缓冲做一次 PPO 更新（由 runner 在评估点调用）。

        入参均为长度 eval_interval 的缓冲，仅消费前 num_steps 步（论文行为）。
        """
        obs = torch.as_tensor(obs_batch[: self.num_steps, :]).to(self.device)
        actions = torch.as_tensor(action_batch[: self.num_steps, :]).to(self.device)
        logprobs = torch.as_tensor(logprob_batch[: self.num_steps]).to(self.device)
        rewards = torch.as_tensor(reward_batch[-self.num_steps :]).to(self.device)
        dones = torch.as_tensor(dones_batch[: self.num_steps]).to(self.device)
        values = torch.as_tensor(value_batch[: self.num_steps]).to(self.device)
        next_obs = torch.as_tensor(obs_batch[-1]).to(self.device)
        next_done = torch.as_tensor(dones_batch[-1]).to(self.device)

        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_batch.shape[1:])
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_batch.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.batch_size):
                mb_inds = b_inds[start : start + self.batch_size]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
        self.writer.add_scalar(
            "charts/episodic_return", float(torch.sum(rewards).item()), global_step
        )
