# -*- coding: utf-8 -*-
"""OHASAC 运行器：HASAC + 次序网络（OrderAgent）。

忠实性说明：
- 训练主体（critic/actor/alpha/软更新）完全复用基类 ``OffPolicyHARunner``
  的 hasac 分支（本文件不覆写 ``train()``），次序决策经基类 ``_agent_order``
  钩子注入——论文 ``off_policy_ha_runner`` 的升序排序实现。
- 次序网络训练挂在评估分支（论文 ``off_policy_base_runner`` 原语义），
  奖励信号 = eval 回合平均总奖励。
- 分奖励指标（论文 f1~f7 对应的 info 分量）按论文 tag 记录 tensorboard，
  支撑训练曲线对拍（论文 CSV 即这些 tag 的导出）。
- ``render()`` 移植论文 lfc 分支：回合结束后导出频率曲线与分量奖励 CSV
  （论文 fig9-14 的数据源）。
"""

import os

import numpy as np
import torch
import torch.optim as optim

from algorithms.actors.order_agent import OrderAgent
from runners.off_policy_ha_runner import OffPolicyHARunner

# 分奖励 info 键 → 论文 tensorboard tag（论文 runner 479-486 行的 tag 串，
# training_curve/<tag>/ 目录即对拍口径）
_COMPONENT_TAGS = {
    "f_deviation_reward": "train_average_f_deviation_rewards",
    "total_violation_reward": "train_average_total_violation_rewards",
    "allocate_err_reward": "train_average_allocate_err_rewards",
    "bat_allocate_reward": "train_average_bat_allocate_rewards",
    "bat_output_reward": "train_average_bat_output_smooth_rewards",
    "pv_bat_smooth_reward": "train_average_pv_bat_smooth_rewards",
    "bat_discharge_reward": "train_average_bat_discharge_rewards",
}


class OHASACRunner(OffPolicyHARunner):
    """HASAC + OrderAgent：按频差动态决定智能体更新次序。"""

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        algo_cfg = self.algo_args["algo"]
        self.use_order_net = algo_cfg.get("use_order_net", False)
        if self.use_order_net and not algo_args["eval"]["use_eval"]:
            raise ValueError(
                "ohasac 启用次序网络时必须开启评估（eval.use_eval=true）——"
                "次序网络的 PPO 更新与分量指标落盘均挂在评估分支（论文原语义）"
            )
        # PowerZoo 基类未缓存 batch_size（论文版有），次序网络构造需要它
        self.batch_size = algo_cfg["batch_size"]
        if self.use_order_net:
            self.order_obs_index = algo_cfg.get("order_obs_index", 9)
            self.order_net = OrderAgent(
                self.batch_size,
                self.num_agents,
                num_steps=algo_cfg.get("order_num_steps", 9),
                learning_rate=algo_cfg.get("order_lr", 3e-4),
                writer=(
                    getattr(self, "writer", None)
                    if hasattr(getattr(self, "writer", None), "add_scalar")
                    else None
                ),
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
        self._init_component_buffers()

    # ------------------------------------------------------------------ #
    # 检查点：次序网络与优化器随模型一并保存/恢复（基类只管 actor/critic）   #
    # ------------------------------------------------------------------ #
    def save(self):
        super().save()
        if self.use_order_net:
            torch.save(
                {
                    "order_net": self.order_net.state_dict(),
                    "order_optimizer": self.order_optimizer.state_dict(),
                },
                os.path.join(self.save_dir, "order_net.pt"),
            )

    def restore(self):
        super().restore()
        order_ckpt = os.path.join(self.algo_args["train"]["model_dir"], "order_net.pt")
        if self.use_order_net and os.path.exists(order_ckpt):
            state = torch.load(order_ckpt, map_location=self.device)
            self.order_net.load_state_dict(state["order_net"])
            self.order_optimizer.load_state_dict(state["order_optimizer"])

    # ------------------------------------------------------------------ #
    # 次序决策钩子（基类 train 的 hasac 分支调用）                          #
    # ------------------------------------------------------------------ #
    def _agent_order(self, share_obs=None, next_share_obs=None):
        """次序网络前向：频差维度相邻差分 → 升序即更新次序（论文实现）。"""
        if not self.use_order_net or share_obs is None:
            return super()._agent_order(share_obs, next_share_obs)
        with torch.no_grad():
            order_obs1 = share_obs[:, self.order_obs_index]
            order_obs2 = next_share_obs[:, self.order_obs_index]
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
            _, sorted_indices = torch.sort(order_action, descending=False)
            # 缓冲边界显式化（P0-3）：论文时序下 order_step 不会越过缓冲长度
            buf_len = self.order_obs.shape[0]
            if self.order_step < buf_len:
                self.order_obs[self.order_step] = order_obs
                self.order_actions[self.order_step] = order_action
                self.order_logprobs[self.order_step] = order_logprob
                self.order_values[self.order_step] = order_value
                self.order_dones[self.order_step] = self.order_step >= buf_len - 1
            self.order_step += 1
            return sorted_indices[0].tolist()

    # ------------------------------------------------------------------ #
    # 评估：基类跑回合并返回均值 → 次序网络 PPO 更新 + 分量指标落盘         #
    # ------------------------------------------------------------------ #
    def eval(self, step):
        eval_avg_rew = super().eval(step)
        if self.use_order_net:
            horizon = self.order_obs.shape[0]
            self.order_rewards = torch.full((horizon,), float(eval_avg_rew)).to(
                self.device
            )
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
        self._log_component_rewards(step)

    # ------------------------------------------------------------------ #
    # 分奖励指标（论文 tag，供训练曲线对拍）                                #
    # ------------------------------------------------------------------ #
    def _init_component_buffers(self):
        n = self.algo_args["train"]["n_rollout_threads"]
        self._ep_total = np.zeros(n)
        self._ep_components = {key: np.zeros(n) for key in _COMPONENT_TAGS}
        self._done_totals = []
        self._done_components = {key: [] for key in _COMPONENT_TAGS}

    def insert(self, data):
        super().insert(data)
        infos, rewards, dones = data[6], data[4], data[5]
        dones_env = np.all(np.asarray(dones), axis=1)
        for thread in range(len(infos)):
            info0 = infos[thread][0]
            self._ep_total[thread] += float(np.mean(rewards[thread]))
            for key in _COMPONENT_TAGS:
                value = info0.get(key)
                if value is not None:
                    self._ep_components[key][thread] += float(np.mean(value))
            if dones_env[thread]:
                self._done_totals.append(self._ep_total[thread])
                for key in _COMPONENT_TAGS:
                    self._done_components[key].append(self._ep_components[key][thread])
                self._ep_total[thread] = 0.0
                for key in _COMPONENT_TAGS:
                    self._ep_components[key][thread] = 0.0

    def _log_component_rewards(self, step):
        if self._done_totals:
            self.writer.add_scalar(
                "train_average_episode_rewards", np.mean(self._done_totals), step
            )
            self._done_totals = []
        for key, tag in _COMPONENT_TAGS.items():
            if self._done_components[key]:
                self.writer.add_scalar(tag, np.mean(self._done_components[key]), step)
                self._done_components[key] = []

    # ------------------------------------------------------------------ #
    # 渲染：论文 lfc 分支移植——回合结束导出频率/分量奖励 CSV                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def render(self):
        """渲染一个回合并导出论文口径的 CSV（model_dir/render/ 下）。"""
        print("start rendering (LFC)")
        env = self.envs  # render 模式下是裸 LFCEnv 包装器
        model_dir = self.algo_args["train"]["model_dir"] or "./results/render_out"
        render_dir = os.path.join(model_dir, "render")
        os.makedirs(render_dir, exist_ok=True)

        component_logs = {key: [] for key in _COMPONENT_TAGS}
        active_list = []
        obs_list, _, _ = env.reset()
        rewards = 0.0
        while True:
            actions = []
            for agent_id, obs in enumerate(obs_list):
                out = self.actor[agent_id].get_actions(
                    np.asarray(obs)[None], stochastic=False
                )
                # HASAC.get_actions 非 lstm 路径返回单张量（lstm 路径为
                # (action, rnn_states) 元组）；渲染回放需 CPU 转移兼容 CUDA
                action = out[0] if isinstance(out, tuple) else out
                actions.append(
                    np.asarray(action.detach().cpu()).reshape(1)
                )
            actions = np.stack(actions)
            obs_list, _, step_rewards, step_dones, infos, _ = env.step(actions)
            info0 = infos[0]
            rewards += float(step_rewards[0][0])
            active_list.append(float(np.mean(info0["active_output"])))
            for key in _COMPONENT_TAGS:
                value = info0.get(key)
                component_logs[key].append(
                    float(np.mean(value)) if value is not None else 0.0
                )
            if bool(np.all(step_dones)):
                print(f"total reward of this episode: {rewards}")
                break

        # 频率曲线（论文 out_t / out_f_load1）
        t, f = env.render()
        stem = f"{self.args['algo']}_lfc"
        np.savetxt(
            os.path.join(render_dir, f"{stem}_freq.csv"),
            np.column_stack((t, f)),
            delimiter=",",
            fmt="%.14f",
            header="time,freq at bus 8",
            comments="",
        )
        steps = np.arange(1, len(active_list) + 1)
        columns = [steps, active_list] + [component_logs[k] for k in _COMPONENT_TAGS]
        np.savetxt(
            os.path.join(render_dir, f"{stem}_reward.csv"),
            np.column_stack(columns),
            delimiter=",",
            fmt="%.14f",
            header="time,active_output," + ",".join(_COMPONENT_TAGS),
            comments="",
        )
        print(f"render 导出完成：{render_dir}")
