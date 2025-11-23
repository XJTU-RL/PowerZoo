# -*- coding: utf-8 -*-
"""
Stackelberg Game Logger

This module provides logging functionality for the Stackelberg game environment,
compatible with PowerZoo's logging infrastructure.
"""

import os
import json
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

	Compatible with PowerZoo's logger interface.
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
		self.total_steps = 0

		# Metrics storage
		self.episode_rewards = defaultdict(list)
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

		logger.info(f"StackelbergLogger initialized: {self.run_dir}")

	def init(self, episodes: int):
		"""
		Initialize logger for training run.

		Args:
			episodes: Total number of episodes
		"""
		self.total_episodes = episodes
		self.episode_count = 0
		logger.info(f"Training initialized for {episodes} episodes")

	def episode_init(self, episode: int):
		"""
		Initialize for new episode.

		Args:
			episode: Episode number
		"""
		self.current_episode = episode
		self.episode_step_count = 0
		self.current_episode_rewards = defaultdict(float)
		self.current_episode_metrics = defaultdict(list)

	def per_step(self, data: Dict[str, Any]):
		"""
		Log data per step.

		Args:
			data: Dictionary containing step data
		"""
		self.episode_step_count += 1
		self.total_steps += 1

		# Extract rewards
		rewards = data.get('rewards', {})
		if isinstance(rewards, dict):
			for agent_id, reward in rewards.items():
				self.current_episode_rewards[agent_id] += reward
		elif isinstance(rewards, (list, np.ndarray)):
			for i, reward in enumerate(rewards):
				r = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
				self.current_episode_rewards[i] += r

		# Extract system metrics
		infos = data.get('infos', [])
		if infos and len(infos) > 0:
			uc_info = infos[0] if isinstance(infos, list) else infos.get(0, {})
			if isinstance(uc_info, dict):
				if 'voltage_violations' in uc_info:
					self.current_episode_metrics['voltage_violations'].append(
						uc_info['voltage_violations']
					)
				if 'power_loss_ratio' in uc_info:
					self.current_episode_metrics['power_losses'].append(
						uc_info['power_loss_ratio']
					)

		# Calculate Nash gap if rewards available
		if rewards:
			self._calculate_nash_gap(rewards)

	def _calculate_nash_gap(self, rewards):
		"""Calculate Nash gap between UC and consumers."""
		if isinstance(rewards, dict):
			uc_reward = rewards.get(0, 0)
			consumer_rewards = [r for aid, r in rewards.items() if aid > 0]
		elif isinstance(rewards, (list, np.ndarray)):
			if len(rewards) > 0:
				uc_reward = rewards[0][0] if isinstance(rewards[0], (list, np.ndarray)) else rewards[0]
				consumer_rewards = [
					r[0] if isinstance(r, (list, np.ndarray)) else r
					for r in rewards[1:]
				]
			else:
				return

		if consumer_rewards:
			avg_consumer = np.mean(consumer_rewards)
			nash_gap = abs(uc_reward - avg_consumer)
			self.current_episode_metrics['nash_gap'].append(nash_gap)

	def episode_log(
		self,
		actor_train_infos: Dict[str, Any],
		critic_train_info: Dict[str, Any],
		actor_buffer: Any,
		critic_buffer: Any,
	):
		"""
		Log episode summary.

		Args:
			actor_train_infos: Actor training info
			critic_train_info: Critic training info
			actor_buffer: Actor replay buffer
			critic_buffer: Critic replay buffer
		"""
		self.episode_count += 1

		# Store episode rewards
		for agent_id, total_reward in self.current_episode_rewards.items():
			self.episode_rewards[f'agent_{agent_id}'].append(total_reward)

		# Calculate episode averages
		for metric_name, values in self.current_episode_metrics.items():
			if values:
				self.episode_metrics[f'{metric_name}_mean'].append(np.mean(values))

		# Log to TensorBoard if available
		if self.writter is not None:
			# UC reward
			if 'agent_0' in self.current_episode_rewards:
				self.writter.add_scalar(
					'rewards/uc_reward',
					self.current_episode_rewards['agent_0'],
					self.episode_count
				)

			# Average consumer reward
			consumer_rewards = [
				r for aid, r in self.current_episode_rewards.items()
				if isinstance(aid, str) and aid.startswith('agent_') and aid != 'agent_0'
			]
			if consumer_rewards:
				self.writter.add_scalar(
					'rewards/avg_consumer_reward',
					np.mean(consumer_rewards),
					self.episode_count
				)

			# Nash gap
			if 'nash_gap' in self.current_episode_metrics:
				self.writter.add_scalar(
					'metrics/nash_gap',
					np.mean(self.current_episode_metrics['nash_gap']),
					self.episode_count
				)

		# Log to console at intervals
		if self.episode_count % self.log_interval == 0:
			self._print_episode_summary()

	def _print_episode_summary(self):
		"""Print episode summary to console."""
		uc_reward = self.current_episode_rewards.get(0, 0)
		consumer_rewards = [
			r for aid, r in self.current_episode_rewards.items() if aid != 0
		]
		avg_consumer = np.mean(consumer_rewards) if consumer_rewards else 0

		nash_gaps = self.current_episode_metrics.get('nash_gap', [])
		avg_nash_gap = np.mean(nash_gaps) if nash_gaps else 0

		logger.info(
			f"Episode {self.episode_count}: "
			f"UC={uc_reward:.2f}, "
			f"Consumer={avg_consumer:.2f}, "
			f"Nash Gap={avg_nash_gap:.4f}, "
			f"Steps={self.episode_step_count}"
		)

	def eval_init(self):
		"""Initialize evaluation mode."""
		self.eval_rewards = defaultdict(list)
		self.eval_metrics = defaultdict(list)

	def eval_per_step(self, eval_data: Dict[str, Any]):
		"""Log evaluation step data."""
		rewards = eval_data.get('rewards', {})
		if isinstance(rewards, dict):
			for agent_id, reward in rewards.items():
				self.eval_rewards[agent_id].append(reward)
		elif isinstance(rewards, (list, np.ndarray)):
			for i, reward in enumerate(rewards):
				r = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
				self.eval_rewards[i].append(r)

	def eval_log(self, eval_episode: int):
		"""
		Log evaluation summary.

		Args:
			eval_episode: Evaluation episode number
		"""
		# Calculate totals
		uc_total = sum(self.eval_rewards.get(0, [0]))
		consumer_totals = [
			sum(self.eval_rewards.get(i, [0]))
			for i in range(1, self.num_agents)
		]
		avg_consumer = np.mean(consumer_totals) if consumer_totals else 0

		logger.info(
			f"Eval Episode {eval_episode}: "
			f"UC={uc_total:.2f}, "
			f"Avg Consumer={avg_consumer:.2f}"
		)

		if self.writter is not None:
			self.writter.add_scalar('eval/uc_reward', uc_total, eval_episode)
			self.writter.add_scalar('eval/avg_consumer_reward', avg_consumer, eval_episode)

	def close(self):
		"""Save logs and close."""
		# Save metrics to file
		metrics_file = os.path.join(self.run_dir, 'metrics.json')
		metrics_data = {
			'episode_rewards': {k: list(v) for k, v in self.episode_rewards.items()},
			'episode_metrics': {k: list(v) for k, v in self.episode_metrics.items()},
			'total_episodes': self.episode_count,
			'total_steps': self.total_steps,
			'timestamp': datetime.now().isoformat()
		}

		with open(metrics_file, 'w') as f:
			json.dump(metrics_data, f, indent=2)

		logger.info(f"Metrics saved to {metrics_file}")
