# -*- coding: utf-8 -*-
"""
District Dispatch Logger

区域调度环境日志记录器，兼容 PowerZoo 的 base_logger 接口。
Runner 以 tuple 形式传递 data，本 logger 解包后提取指标。

记录指标:
- 每个台区: 电压统计、PV出力、储能SOC、净负荷
- 系统级: 总网损、总功率交换、电压越限率
- 经济指标: 运行成本、碳排放量
- 训练指标: 奖励分量分解
"""

import logging
import os
import time
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger('DistrictDispatchLogger')


class DistrictDispatchLogger:
	"""区域调度环境日志记录器

	兼容 PowerZoo 的 base_logger 接口:
	- init(episodes)
	- episode_init(episode)
	- per_step(data)  # data 是 tuple
	- episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
	- eval_init(), eval_per_step(), eval_thread_done(), eval_log()
	- close()
	"""

	def __init__(
		self,
		args: Dict[str, Any],
		algo_args: Dict[str, Any],
		env_args: Dict[str, Any],
		num_agents: int,
		writter: Any = None,
		run_dir: Optional[str] = None,
		log_interval: int = 1,
	):
		"""初始化日志记录器

		Args:
			args: 主参数
			algo_args: 算法参数
			env_args: 环境参数
			num_agents: 智能体数量（= 台区数）
			writter: TensorBoard SummaryWriter
			run_dir: 日志输出目录
			log_interval: 日志打印间隔（episode数）
		"""
		self.args = args
		self.algo_args = algo_args
		self.env_args = env_args
		self.num_agents = num_agents
		self.writter = writter
		self.run_dir = run_dir or "logs/district_dispatch"
		self.log_interval = log_interval

		os.makedirs(self.run_dir, exist_ok=True)

		# 训练状态
		self.episode_count = 0
		self.episode = 0
		self.total_steps = 0

		# 奖励追踪（base_logger 兼容）
		self.train_episode_rewards = None
		self.done_episodes_rewards = []

		# 当前 episode 指标
		self.current_episode_rewards = defaultdict(float)
		self.current_episode_metrics = defaultdict(list)

		# 历史指标
		self.episode_rewards_history = defaultdict(list)
		self.episode_metrics = defaultdict(list)

		# 日志文件
		log_file_path = os.path.join(self.run_dir, 'progress.txt')
		self.log_file = open(log_file_path, 'w')

		logger.info(f"DistrictDispatchLogger initialized: {self.run_dir}")

	def init(self, episodes: int):
		"""初始化训练运行

		Args:
			episodes: 总 episode 数
		"""
		self.start = time.time()
		self.episodes = episodes
		self.total_episodes = episodes
		self.episode_count = 0

		n_threads = self.algo_args["train"]["n_rollout_threads"]
		self.train_episode_rewards = np.zeros(n_threads)
		self.done_episodes_rewards = []

		self.log_file.write(f"{'=' * 80}\n")
		self.log_file.write(f"District Dispatch Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
		self.log_file.write(f"Total episodes: {episodes}, Agents: {self.num_agents}\n")
		self.log_file.write(f"Environment: {self.args['env']}, Algorithm: {self.args['algo']}\n")
		self.log_file.write(f"{'=' * 80}\n\n")
		self.log_file.flush()

	def episode_init(self, episode: int):
		"""新 episode 初始化

		Args:
			episode: 当前 episode 编号
		"""
		self.episode = episode
		self.episode_step_count = 0
		self.current_episode_rewards = defaultdict(float)
		self.current_episode_metrics = defaultdict(list)

	def per_step(self, data):
		"""每步日志记录

		Args:
			data: Runner 传递的 tuple:
				(obs, share_obs, rewards, dones, infos, available_actions,
				 values, actions, action_log_probs, rnn_states, rnn_states_critic)
		"""
		(
			obs,
			share_obs,
			rewards,
			dones,
			infos,
			available_actions,
			values,
			actions,
			action_log_probs,
			rnn_states,
			rnn_states_critic,
		) = data

		self.episode_step_count += 1
		self.total_steps += 1

		# base_logger 兼容：per-thread 奖励追踪
		dones_env = np.all(dones, axis=1)
		reward_env = np.mean(rewards, axis=1).flatten()
		self.train_episode_rewards += reward_env
		for t in range(self.algo_args["train"]["n_rollout_threads"]):
			if dones_env[t]:
				self.done_episodes_rewards.append(self.train_episode_rewards[t])
				self.train_episode_rewards[t] = 0

		# per-agent 奖励追踪
		if isinstance(rewards, np.ndarray) and rewards.ndim >= 2:
			for i in range(min(rewards.shape[1], self.num_agents)):
				r = float(np.mean(rewards[:, i]))
				self.current_episode_rewards[i] += r

		# 从 infos 提取区域调度指标
		self._extract_dispatch_metrics(infos)

	def _extract_dispatch_metrics(self, infos):
		"""从环境 info 中提取区域调度指标"""
		if infos is None:
			return

		try:
			# infos[0] 是第一个线程的 info
			info_list = infos[0] if len(infos) > 0 else []
			for agent_id, agent_info in enumerate(info_list):
				if not isinstance(agent_info, dict):
					continue

				# 电压越限
				if 'reward/voltage_compliance_raw' in agent_info:
					self.current_episode_metrics[f'district_{agent_id}/voltage_penalty'].append(
						agent_info['reward/voltage_compliance_raw']
					)

				# 经济调度
				if 'reward/economic_dispatch_raw' in agent_info:
					self.current_episode_metrics[f'district_{agent_id}/economic_reward'].append(
						agent_info['reward/economic_dispatch_raw']
					)

				# 网损
				if 'reward/loss_minimization_raw' in agent_info:
					self.current_episode_metrics['system/loss_penalty'].append(
						agent_info['reward/loss_minimization_raw']
					)

				# 总奖励
				if 'reward/total' in agent_info:
					self.current_episode_metrics[f'district_{agent_id}/total_reward'].append(
						agent_info['reward/total']
					)
		except (IndexError, TypeError, KeyError):
			pass

	def episode_log(
		self,
		actor_train_infos: Dict[str, Any],
		critic_train_info: Dict[str, Any],
		actor_buffer: Any,
		critic_buffer: Any,
	):
		"""Episode 总结日志

		Args:
			actor_train_infos: Actor 训练信息
			critic_train_info: Critic 训练信息
			actor_buffer: Actor 缓冲区
			critic_buffer: Critic 缓冲区
		"""
		self.episode_count += 1
		total_num_steps = (
			self.episode
			* self.algo_args["train"]["episode_length"]
			* self.algo_args["train"]["n_rollout_threads"]
		)
		end = time.time()
		fps = int(total_num_steps / max(1, end - self.start))

		# 控制台输出（base_logger 兼容）
		print(
			"环境: {} 算法 {} 实验 {} updates {}/{} episodes, "
			"总时间步 {}/{}, FPS {}.".format(
				self.args["env"],
				self.args["algo"],
				self.args["exp_name"],
				self.episode,
				self.episodes,
				total_num_steps,
				self.algo_args["train"]["num_env_steps"],
				fps,
			)
		)

		# 平均步奖励
		avg_step_reward = critic_buffer.get_mean_rewards()
		critic_train_info["average_step_rewards"] = avg_step_reward
		print(f"Average step reward is {avg_step_reward}.")

		# 存储 per-agent episode 奖励
		for agent_id, total_reward in self.current_episode_rewards.items():
			self.episode_rewards_history[f'district_{agent_id}'].append(total_reward)

		# TensorBoard 写入
		if self.writter is not None:
			# Actor 训练信息
			for agent_id in range(self.num_agents):
				for k, v in actor_train_infos[agent_id].items():
					agent_k = f"district_{agent_id}/{k}"
					self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

			# Critic 训练信息
			for k, v in critic_train_info.items():
				critic_k = f"critic/{k}"
				self.writter.add_scalars(critic_k, {critic_k: v}, total_num_steps)

			# 区域调度指标
			for metric_name, values in self.current_episode_metrics.items():
				if values:
					self.writter.add_scalar(
						f"dispatch/{metric_name}",
						np.mean(values),
						total_num_steps,
					)

			# Per-agent 奖励
			for agent_id in range(self.num_agents):
				r = self.current_episode_rewards.get(agent_id, 0)
				self.writter.add_scalar(
					f"district_{agent_id}/episode_reward",
					r,
					total_num_steps,
				)

		# Done episodes 奖励日志
		if len(self.done_episodes_rewards) > 0:
			aver_episode_rewards = np.mean(self.done_episodes_rewards)
			print(f"Some episodes done, average episode reward is {aver_episode_rewards}.\n")
			if self.writter is not None:
				self.writter.add_scalars(
					"train_episode_rewards",
					{"aver_rewards": aver_episode_rewards},
					total_num_steps,
				)
			self.log_file.write(
				f"[DONE] Episode {self.episode}, Steps {total_num_steps}, "
				f"Avg Reward: {aver_episode_rewards:.4f}, FPS: {fps}\n"
			)
			self.log_file.flush()
			self.done_episodes_rewards = []
		else:
			self.log_file.write(
				f"[TRAIN] Episode {self.episode}, Steps {total_num_steps}, "
				f"Avg Step Reward: {avg_step_reward:.4f}, FPS: {fps}\n"
			)
			self.log_file.flush()

		# 打印台区调度摘要
		if self.episode_count % self.log_interval == 0:
			self._print_dispatch_summary()

	def _print_dispatch_summary(self):
		"""打印区域调度 episode 摘要"""
		print(f"\n--- District Dispatch Summary (Episode {self.episode}) ---")
		for agent_id in range(self.num_agents):
			r = self.current_episode_rewards.get(agent_id, 0)
			print(f"  District {agent_id}: total_reward={r:.4f}")
		print("---\n")

	# ===== 评估接口 =====

	def eval_init(self):
		"""初始化评估"""
		self.eval_episode_rewards = defaultdict(float)
		self.eval_episode_metrics = defaultdict(list)
		self.eval_threads_done = 0

	def eval_per_step(self, eval_data):
		"""评估每步日志

		Args:
			eval_data: (eval_obs, eval_share_obs, eval_rewards, eval_dones,
						eval_infos, eval_available_actions)
		"""
		eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_data

		if isinstance(eval_rewards, np.ndarray) and eval_rewards.ndim >= 2:
			for i in range(min(eval_rewards.shape[1], self.num_agents)):
				r = float(np.mean(eval_rewards[:, i]))
				self.eval_episode_rewards[i] += r

	def eval_thread_done(self, tid):
		"""评估线程完成"""
		self.eval_threads_done += 1

	def eval_log(self, eval_episode: int):
		"""评估总结日志

		Args:
			eval_episode: 评估 episode 编号
		"""
		total_num_steps = (
			self.episode
			* self.algo_args["train"]["episode_length"]
			* self.algo_args["train"]["n_rollout_threads"]
		)

		print(f"\n=== Eval Summary (Episode {eval_episode}) ===")
		total_eval_reward = 0.0
		for agent_id in range(self.num_agents):
			r = self.eval_episode_rewards.get(agent_id, 0)
			total_eval_reward += r
			print(f"  District {agent_id}: eval_reward={r:.4f}")
		avg_eval = total_eval_reward / max(self.num_agents, 1)
		print(f"  Average: {avg_eval:.4f}")
		print("===\n")

		if self.writter is not None:
			self.writter.add_scalar("eval/average_reward", avg_eval, total_num_steps)
			for agent_id in range(self.num_agents):
				r = self.eval_episode_rewards.get(agent_id, 0)
				self.writter.add_scalar(
					f"eval/district_{agent_id}_reward", r, total_num_steps
				)

		self.log_file.write(
			f"[EVAL] Episode {eval_episode}, Avg Eval Reward: {avg_eval:.4f}\n"
		)
		self.log_file.flush()

	def close(self):
		"""关闭日志记录器，释放资源"""
		if hasattr(self, 'log_file') and self.log_file:
			self.log_file.close()
		logger.info("DistrictDispatchLogger closed")
