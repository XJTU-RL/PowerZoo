# -*- coding: utf-8 -*-
"""
Stackelberg Game Logger

This module provides logging functionality for the Stackelberg game environment,
compatible with PowerZoo's logging infrastructure (base_logger interface).

IMPORTANT: The runner passes `data` as a tuple, not a dict.
This logger unpacks the tuple to extract rewards, dones, infos, etc.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger('StackelbergLogger')


class StackelbergLogger:
	"""
	Logger for Stackelberg game environment.

	Tracks and logs:
	- Episode rewards for UC and consumers
	- Nash gap metrics
	- System performance (voltage, losses, carbon)
	- Training progress

	Compatible with PowerZoo's base_logger interface.
	The runner calls per_step(data) where data is a tuple:
		(obs, share_obs, rewards, dones, infos, available_actions,
		 values, actions, action_log_probs, rnn_states, rnn_states_critic)
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
		"""
		Initialize Stackelberg logger.

		Args:
			args: Main arguments
			algo_args: Algorithm arguments
			env_args: Environment arguments
			num_agents: Number of agents
			writter: TensorBoard writer (optional)
			run_dir: Directory for logs
			log_interval: Logging interval
		"""
		self.args = args
		self.algo_args = algo_args
		self.env_args = env_args
		self.num_agents = num_agents
		self.writter = writter
		self.run_dir = run_dir or "logs/stackelberg"
		self.log_interval = log_interval

		# Create log directory
		os.makedirs(self.run_dir, exist_ok=True)

		# Episode tracking
		self.episode_count = 0
		self.episode = 0
		self.total_steps = 0

		# Metrics storage
		self.episode_rewards_history = defaultdict(list)
		self.episode_metrics = defaultdict(list)
		self.step_metrics = defaultdict(list)

		# Stackelberg-specific metrics
		self.nash_gaps = []
		self.uc_utilities = []
		self.consumer_utilities = []

		# System metrics
		self.system_metrics = {
			'voltage_violations': [],
			'power_losses': [],
			'carbon_emissions': [],
			'dr_participation': []
		}

		# Progress log file (compatible with base_logger)
		log_file_path = os.path.join(self.run_dir, 'progress.txt')
		self.log_file = open(log_file_path, 'w')

		logger.info(f"StackelbergLogger initialized: {self.run_dir}")

	def init(self, episodes: int):
		"""
		Initialize logger for training run.
		Must match base_logger.init() interface.

		Args:
			episodes: Total number of episodes
		"""
		self.start = time.time()
		self.episodes = episodes
		self.total_episodes = episodes
		self.episode_count = 0

		# base_logger compatible: per-thread reward tracking
		self.train_episode_rewards = np.zeros(
			self.algo_args["train"]["n_rollout_threads"]
		)
		self.done_episodes_rewards = []

		self.log_file.write(f"{'=' * 80}\n")
		self.log_file.write(f"Stackelberg Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
		self.log_file.write(f"Total episodes: {episodes}, Total steps: {self.algo_args['train']['num_env_steps']}\n")
		self.log_file.write(f"Environment: {self.args['env']}, Algorithm: {self.args['algo']}\n")
		self.log_file.write(f"{'=' * 80}\n\n")
		self.log_file.flush()

		logger.info(f"Training initialized for {episodes} episodes")

	def episode_init(self, episode: int):
		"""
		Initialize for new episode.
		Must match base_logger.episode_init() interface.

		Args:
			episode: Episode number
		"""
		self.episode = episode
		self.current_episode = episode
		self.episode_step_count = 0
		self.current_episode_rewards = defaultdict(float)
		self.current_episode_metrics = defaultdict(list)

	def per_step(self, data):
		"""
		Log data per step.

		Args:
			data: Tuple from runner containing:
				(obs, share_obs, rewards, dones, infos, available_actions,
				 values, actions, action_log_probs, rnn_states, rnn_states_critic)
		"""
		# Unpack tuple -- same convention as base_logger.per_step
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

		# --- base_logger compatible reward tracking ---
		dones_env = np.all(dones, axis=1)
		reward_env = np.mean(rewards, axis=1).flatten()
		self.train_episode_rewards += reward_env
		for t in range(self.algo_args["train"]["n_rollout_threads"]):
			if dones_env[t]:
				self.done_episodes_rewards.append(self.train_episode_rewards[t])
				self.train_episode_rewards[t] = 0

		# --- Stackelberg-specific: per-agent reward tracking ---
		# rewards shape: (n_threads, n_agents, 1)
		if isinstance(rewards, np.ndarray) and rewards.ndim >= 2:
			for i in range(rewards.shape[1]):
				r = float(np.mean(rewards[:, i]))
				self.current_episode_rewards[i] += r

		# --- Extract system metrics from infos ---
		if infos is not None:
			try:
				# infos[0] is the first thread's info list
				info_list = infos[0] if len(infos) > 0 else []
				uc_info = info_list[0] if isinstance(info_list, (list, np.ndarray)) and len(info_list) > 0 else {}
				if isinstance(uc_info, dict):
					if 'voltage_violations' in uc_info:
						self.current_episode_metrics['voltage_violations'].append(
							uc_info['voltage_violations']
						)
					if 'power_loss_ratio' in uc_info:
						self.current_episode_metrics['power_losses'].append(
							uc_info['power_loss_ratio']
						)
			except (IndexError, TypeError, KeyError):
				pass

		# Calculate Nash gap if rewards available
		if isinstance(rewards, np.ndarray) and rewards.size > 0:
			self._calculate_nash_gap(rewards)

	def _calculate_nash_gap(self, rewards):
		"""
		Calculate Nash gap between UC and consumers.

		Args:
			rewards: numpy array shape (n_threads, n_agents, 1) or dict/list
		"""
		try:
			if isinstance(rewards, np.ndarray):
				# rewards shape: (n_threads, n_agents, 1), take mean across threads
				mean_rewards = np.mean(rewards, axis=0).flatten()  # (n_agents,)
				if len(mean_rewards) > 1:
					uc_reward = float(mean_rewards[0])
					consumer_rewards = mean_rewards[1:].tolist()
				else:
					return
			elif isinstance(rewards, dict):
				uc_reward = float(rewards.get(0, 0))
				consumer_rewards = [float(r) for aid, r in rewards.items() if aid > 0]
			elif isinstance(rewards, list) and len(rewards) > 0:
				uc_reward = float(rewards[0][0]) if isinstance(rewards[0], (list, np.ndarray)) else float(rewards[0])
				consumer_rewards = [
					float(r[0]) if isinstance(r, (list, np.ndarray)) else float(r)
					for r in rewards[1:]
				]
			else:
				return

			if consumer_rewards:
				avg_consumer = np.mean(consumer_rewards)
				nash_gap = abs(uc_reward - avg_consumer)
				self.current_episode_metrics['nash_gap'].append(nash_gap)
		except (IndexError, TypeError, ValueError):
			pass

	def episode_log(
		self,
		actor_train_infos: Dict[str, Any],
		critic_train_info: Dict[str, Any],
		actor_buffer: Any,
		critic_buffer: Any,
	):
		"""
		Log episode summary.
		Must match base_logger.episode_log() interface.

		Args:
			actor_train_infos: Actor training info
			critic_train_info: Critic training info
			actor_buffer: Actor replay buffer
			critic_buffer: Critic replay buffer
		"""
		self.episode_count += 1
		total_num_steps = (
			self.episode
			* self.algo_args["train"]["episode_length"]
			* self.algo_args["train"]["n_rollout_threads"]
		)
		end = time.time()

		# base_logger compatible console output
		fps = int(total_num_steps / max(1, end - self.start))
		print(
			"环境： {} 算法 {} 实验名称 {} updates {}/{} episodes, "
			"总时间步数 {}/{}, FPS {}.".format(
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

		# Average step reward from critic buffer
		avg_step_reward = critic_buffer.get_mean_rewards()
		critic_train_info["average_step_rewards"] = avg_step_reward
		print(f"Average step reward is {avg_step_reward}.")

		# Store Stackelberg-specific episode rewards history
		for agent_id, total_reward in self.current_episode_rewards.items():
			self.episode_rewards_history[f'agent_{agent_id}'].append(total_reward)

		# Calculate episode averages for Stackelberg metrics
		for metric_name, values in self.current_episode_metrics.items():
			if values:
				self.episode_metrics[f'{metric_name}_mean'].append(np.mean(values))

		# Log to TensorBoard if available
		if self.writter is not None:
			# Actor train info
			for agent_id in range(self.num_agents):
				for k, v in actor_train_infos[agent_id].items():
					agent_k = f"agent{agent_id}/{k}"
					self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
			# Critic train info
			for k, v in critic_train_info.items():
				critic_k = f"critic/{k}"
				self.writter.add_scalars(critic_k, {critic_k: v}, total_num_steps)

			# Stackelberg-specific: Nash gap
			if 'nash_gap' in self.current_episode_metrics:
				self.writter.add_scalar(
					'metrics/nash_gap',
					np.mean(self.current_episode_metrics['nash_gap']),
					total_num_steps,
				)

		# Done episodes reward logging
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
				f"[COMPLETED] Episode {self.episode}, Steps {total_num_steps}, "
				f"Avg Episode Reward: {aver_episode_rewards:.4f}, FPS: {fps}\n"
			)
			self.log_file.flush()
			self.done_episodes_rewards = []
		else:
			self.log_file.write(
				f"[TRAINING] Episode {self.episode}, Steps {total_num_steps}, "
				f"Avg Step Reward: {avg_step_reward:.4f}, FPS: {fps}\n"
			)
			self.log_file.flush()

		# Log Stackelberg-specific summary
		if self.episode_count % self.log_interval == 0:
			self._print_stackelberg_summary()

	def _print_stackelberg_summary(self):
		"""Print Stackelberg-specific episode summary to console."""
		uc_reward = self.current_episode_rewards.get(0, 0)
		consumer_rewards = [
			r for aid, r in self.current_episode_rewards.items() if aid != 0
		]
		avg_consumer = np.mean(consumer_rewards) if consumer_rewards else 0

		nash_gaps = self.current_episode_metrics.get('nash_gap', [])
		avg_nash_gap = np.mean(nash_gaps) if nash_gaps else 0

		logger.info(
			f"Stackelberg Episode {self.episode_count}: "
			f"UC={uc_reward:.2f}, "
			f"Consumer={avg_consumer:.2f}, "
			f"Nash Gap={avg_nash_gap:.4f}, "
			f"Steps={self.episode_step_count}"
		)

	def eval_init(self):
		"""
		Initialize evaluation mode.
		Must match base_logger.eval_init() interface.
		"""
		self.total_num_steps = (
			self.episode
			* self.algo_args["train"]["episode_length"]
			* self.algo_args["train"]["n_rollout_threads"]
		)
		self.eval_episode_rewards = []
		self.one_episode_rewards = []
		for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
			self.one_episode_rewards.append([])
			self.eval_episode_rewards.append([])

	def eval_per_step(self, eval_data):
		"""
		Log evaluation step data.
		Must match base_logger.eval_per_step() interface.

		Args:
			eval_data: Tuple from runner containing:
				(eval_obs, eval_share_obs, eval_rewards, eval_dones,
				 eval_infos, eval_available_actions)
		"""
		(
			eval_obs,
			eval_share_obs,
			eval_rewards,
			eval_dones,
			eval_infos,
			eval_available_actions,
		) = eval_data
		for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
			self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
		self.eval_infos = eval_infos

	def eval_thread_done(self, tid: int):
		"""
		Handle completion of one evaluation thread/episode.
		Must match base_logger.eval_thread_done() interface.

		Args:
			tid: Thread ID that finished its episode
		"""
		self.eval_episode_rewards[tid].append(
			np.sum(self.one_episode_rewards[tid], axis=0)
		)
		self.one_episode_rewards[tid] = []

	def eval_log(self, eval_episode: int):
		"""
		Log evaluation summary.
		Must match base_logger.eval_log() interface.

		Args:
			eval_episode: Evaluation episode number
		"""
		self.eval_episode_rewards = np.concatenate(
			[rewards for rewards in self.eval_episode_rewards if rewards]
		)
		eval_avg_rew = np.mean(self.eval_episode_rewards)
		eval_max_rew = np.max(self.eval_episode_rewards)
		eval_min_rew = np.min(self.eval_episode_rewards)

		print(
			"Evaluation average episode reward is {:.4f} (max: {:.4f}, min: {:.4f}).\n".format(
				eval_avg_rew, eval_max_rew, eval_min_rew
			)
		)

		if self.writter is not None:
			eval_env_infos = {
				"eval_average_episode_rewards": self.eval_episode_rewards,
				"eval_max_episode_rewards": [eval_max_rew],
			}
			for k, v in eval_env_infos.items():
				if len(v) > 0:
					self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)

		self.log_file.write(
			f"[EVAL] Episode {eval_episode}, Steps {self.total_num_steps}, "
			f"Avg Reward: {eval_avg_rew:.4f}, Max: {eval_max_rew:.4f}, "
			f"Min: {eval_min_rew:.4f}, Eval Episodes: {len(self.eval_episode_rewards)}\n"
		)
		self.log_file.flush()

		logger.info(
			f"Eval Episode {eval_episode}: "
			f"Avg={eval_avg_rew:.2f}, Max={eval_max_rew:.2f}, Min={eval_min_rew:.2f}"
		)

	def close(self):
		"""Save logs and close."""
		# Save metrics to file
		metrics_file = os.path.join(self.run_dir, 'metrics.json')
		metrics_data = {
			'episode_rewards': {k: [float(x) for x in v] for k, v in self.episode_rewards_history.items()},
			'episode_metrics': {k: [float(x) for x in v] for k, v in self.episode_metrics.items()},
			'total_episodes': self.episode_count,
			'total_steps': self.total_steps,
			'timestamp': datetime.now().isoformat()
		}

		with open(metrics_file, 'w') as f:
			json.dump(metrics_data, f, indent=2)

		if hasattr(self, 'log_file') and not self.log_file.closed:
			self.log_file.close()

		logger.info(f"Metrics saved to {metrics_file}")
