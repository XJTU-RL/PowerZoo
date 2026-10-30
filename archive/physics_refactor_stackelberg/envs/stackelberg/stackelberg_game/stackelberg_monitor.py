# -*- coding: utf-8 -*-
"""
Stackelberg Game Monitor

This module provides comprehensive monitoring and logging for the
Stackelberg game environment, tracking convergence, Nash gaps, and
system performance metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import logging


class StackelbergMonitor:
	"""
	Monitor for tracking Stackelberg game metrics and convergence.
	
	Tracks:
	- Nash gap between UC and consumer utilities
	- Policy convergence metrics
	- System performance (losses, voltage, carbon)
	- Agent behavior patterns
	"""
	
	def __init__(self, 
				 log_dir: str = "logs/stackelberg",
				 experiment_name: Optional[str] = None,
				 config: Optional[Dict[str, Any]] = None):
		"""
		Initialize monitor.
		
		Args:
			log_dir: Directory for saving logs
			experiment_name: Name of experiment
			config: Configuration dictionary
		"""
		self.config = config or {}
		self.logger = logging.getLogger('StackelbergMonitor')
		
		# Create log directory
		self.log_dir = log_dir
		os.makedirs(log_dir, exist_ok=True)
		
		# Experiment naming
		if experiment_name is None:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			experiment_name = f"stackelberg_exp_{timestamp}"
		self.experiment_name = experiment_name
		
		# Create experiment directory
		self.exp_dir = os.path.join(log_dir, experiment_name)
		os.makedirs(self.exp_dir, exist_ok=True)
		
		# Metrics to track
		self.metrics = defaultdict(list)
		self.episode_metrics = defaultdict(list)
		self.convergence_metrics = defaultdict(list)
		
		# Tracking configuration
		self.track_convergence = config.get('track_convergence', True)
		self.track_nash_gap = config.get('track_nash_gap', True)
		self.track_carbon = config.get('track_carbon', True)
		self.save_interval = config.get('save_interval', 100)
		self.plot_interval = config.get('plot_interval', 50)
		
		# Episode tracking
		self.current_episode = 0
		self.total_steps = 0
		
		# Action and reward buffers
		self.action_buffer = defaultdict(list)
		self.reward_buffer = defaultdict(list)
		
		# Nash equilibrium tracking
		self.nash_gaps = []
		self.uc_utilities = []
		self.consumer_utilities = []
		
		# Convergence tracking
		self.policy_distances = defaultdict(list)
		self.value_changes = defaultdict(list)
		
		# System metrics
		self.system_metrics = {
			'voltage_violations': [],
			'power_losses': [],
			'carbon_emissions': [],
			'der_curtailment': [],
			'dr_participation': []
		}
		
		# Save configuration
		self.save_config()
		
	def save_config(self):
		"""Save monitor configuration."""
		config_path = os.path.join(self.exp_dir, "monitor_config.json")
		with open(config_path, 'w') as f:
			json.dump(self.config, f, indent=2)
	
	def log_step(self, 
				 step: int,
				 observations: Dict[int, np.ndarray],
				 actions: Dict[int, np.ndarray],
				 rewards: Dict[int, float],
				 infos: Dict[int, Dict[str, Any]]):
		"""
		Log data from one environment step.
		
		Args:
			step: Current step number
			observations: Agent observations
			actions: Agent actions
			rewards: Agent rewards
			infos: Additional information
		"""
		self.total_steps += 1
		
		# Store actions and rewards
		for agent_id, action in actions.items():
			self.action_buffer[agent_id].append(action.copy())
		
		for agent_id, reward in rewards.items():
			self.reward_buffer[agent_id].append(reward)
		
		# Extract system metrics from infos
		uc_info = infos.get(0, {})
		if 'voltage_violations' in uc_info:
			self.system_metrics['voltage_violations'].append(
				uc_info['voltage_violations']
			)
		if 'power_loss_ratio' in uc_info:
			self.system_metrics['power_losses'].append(
				uc_info['power_loss_ratio']
			)
		if 'carbon_intensity' in uc_info and 'total_load' in uc_info:
			carbon_emission = uc_info['carbon_intensity'] * uc_info['total_load']
			self.system_metrics['carbon_emissions'].append(carbon_emission)
		
		# Log Nash gap if tracking
		if self.track_nash_gap and len(rewards) > 1:
			self._log_nash_gap(rewards)
		
		# Log at intervals
		if self.total_steps % self.save_interval == 0:
			self.save_metrics()
		
		if self.total_steps % self.plot_interval == 0:
			self.plot_metrics()
	
	def log_episode(self, 
					episode: int,
					episode_rewards: Dict[int, float],
					episode_info: Optional[Dict[str, Any]] = None):
		"""
		Log episode-level metrics.
		
		Args:
			episode: Episode number
			episode_rewards: Total rewards for each agent
			episode_info: Additional episode information
		"""
		self.current_episode = episode
		
		# Store episode rewards
		for agent_id, total_reward in episode_rewards.items():
			self.episode_metrics[f'agent_{agent_id}_reward'].append(total_reward)
		
		# Calculate and store Nash gap
		if self.track_nash_gap:
			uc_reward = episode_rewards.get(0, 0)
			consumer_rewards = [r for aid, r in episode_rewards.items() if aid > 0]
			if consumer_rewards:
				avg_consumer_reward = np.mean(consumer_rewards)
				nash_gap = abs(uc_reward - avg_consumer_reward)
				self.episode_metrics['nash_gap'].append(nash_gap)
		
		# Store system metrics averages
		for metric_name, values in self.system_metrics.items():
			if values:
				self.episode_metrics[f'avg_{metric_name}'].append(np.mean(values))
		
		# Clear buffers
		self.action_buffer.clear()
		self.reward_buffer.clear()
		for metric_list in self.system_metrics.values():
			metric_list.clear()
		
		# Log episode info
		if episode_info:
			for key, value in episode_info.items():
				self.episode_metrics[f'episode_{key}'].append(value)
		
		self.logger.info(
			f"Episode {episode}: "
			f"UC reward={episode_rewards.get(0, 0):.2f}, "
			f"Avg consumer reward={np.mean(consumer_rewards):.2f}, "
			f"Nash gap={nash_gap:.2f}"
		)
	
	def log_convergence(self,
						agent_id: int,
						policy_distance: float,
						value_change: float):
		"""
		Log convergence metrics for an agent.
		
		Args:
			agent_id: Agent ID
			policy_distance: Distance between old and new policy
			value_change: Change in value function
		"""
		if not self.track_convergence:
			return
		
		self.policy_distances[agent_id].append(policy_distance)
		self.value_changes[agent_id].append(value_change)
		
		# Check convergence
		if len(self.policy_distances[agent_id]) > 10:
			recent_distances = self.policy_distances[agent_id][-10:]
			if all(d < 0.01 for d in recent_distances):
				self.logger.info(f"Agent {agent_id} policy converged")
	
	def _log_nash_gap(self, rewards: Dict[int, float]):
		"""Calculate and log Nash gap."""
		uc_reward = rewards.get(0, 0)
		consumer_rewards = [r for aid, r in rewards.items() if aid > 0]
		
		if consumer_rewards:
			avg_consumer_reward = np.mean(consumer_rewards)
			nash_gap = abs(uc_reward - avg_consumer_reward)
			self.nash_gaps.append(nash_gap)
			self.uc_utilities.append(uc_reward)
			self.consumer_utilities.append(avg_consumer_reward)
	
	def get_summary_statistics(self) -> Dict[str, Any]:
		"""Get summary statistics for current experiment."""
		summary = {
			'experiment_name': self.experiment_name,
			'total_episodes': self.current_episode,
			'total_steps': self.total_steps,
			'current_metrics': {}
		}
		
		# Add recent metrics
		for metric_name, values in self.episode_metrics.items():
			if values:
				summary['current_metrics'][metric_name] = {
					'mean': np.mean(values[-10:]),
					'std': np.std(values[-10:]),
					'min': np.min(values[-10:]),
					'max': np.max(values[-10:])
				}
		
		# Add convergence info
		if self.track_convergence:
			converged_agents = []
			for agent_id, distances in self.policy_distances.items():
				if len(distances) > 10:
					recent = distances[-10:]
					if all(d < 0.01 for d in recent):
						converged_agents.append(agent_id)
			summary['converged_agents'] = converged_agents
		
		return summary
	
	def save_metrics(self):
		"""Save all metrics to files."""
		# Save episode metrics
		if self.episode_metrics:
			df_episodes = pd.DataFrame(self.episode_metrics)
			df_episodes.to_csv(
				os.path.join(self.exp_dir, "episode_metrics.csv"),
				index=False
			)
		
		# Save convergence metrics
		if self.track_convergence and self.policy_distances:
			convergence_data = {
				f'agent_{aid}_policy_dist': dists 
				for aid, dists in self.policy_distances.items()
			}
			convergence_data.update({
				f'agent_{aid}_value_change': changes
				for aid, changes in self.value_changes.items()
			})
			df_convergence = pd.DataFrame(
				dict([(k, pd.Series(v)) for k, v in convergence_data.items()])
			)
			df_convergence.to_csv(
				os.path.join(self.exp_dir, "convergence_metrics.csv"),
				index=False
			)
		
		# Save Nash gap data
		if self.track_nash_gap and self.nash_gaps:
			nash_data = {
				'nash_gap': self.nash_gaps,
				'uc_utility': self.uc_utilities,
				'avg_consumer_utility': self.consumer_utilities
			}
			df_nash = pd.DataFrame(
				dict([(k, pd.Series(v)) for k, v in nash_data.items()])
			)
			df_nash.to_csv(
				os.path.join(self.exp_dir, "nash_gap_metrics.csv"),
				index=False
			)
		
		# Save summary
		summary = self.get_summary_statistics()
		with open(os.path.join(self.exp_dir, "summary.json"), 'w') as f:
			json.dump(summary, f, indent=2)
		
		self.logger.info(f"Metrics saved to {self.exp_dir}")
	
	def plot_metrics(self):
		"""Plot metrics (requires matplotlib)."""
		try:
			import matplotlib.pyplot as plt
			import matplotlib.gridspec as gridspec
			
			# Create figure
			fig = plt.figure(figsize=(15, 10))
			gs = gridspec.GridSpec(3, 2, figure=fig)
			
			# Plot 1: Episode rewards
			ax1 = fig.add_subplot(gs[0, 0])
			if 'agent_0_reward' in self.episode_metrics:
				episodes = range(len(self.episode_metrics['agent_0_reward']))
				ax1.plot(episodes, self.episode_metrics['agent_0_reward'], 
						label='UC', linewidth=2)
				
				# Plot average consumer reward
				consumer_rewards = []
				for i in range(len(episodes)):
					rewards = []
					for key in self.episode_metrics:
						if key.startswith('agent_') and key.endswith('_reward'):
							agent_id = int(key.split('_')[1])
							if agent_id > 0:
								rewards.append(self.episode_metrics[key][i])
					if rewards:
						consumer_rewards.append(np.mean(rewards))
				
				if consumer_rewards:
					ax1.plot(episodes, consumer_rewards, 
							label='Avg Consumer', linewidth=2)
				
				ax1.set_xlabel('Episode')
				ax1.set_ylabel('Reward')
				ax1.set_title('Agent Rewards')
				ax1.legend()
				ax1.grid(True, alpha=0.3)
			
			# Plot 2: Nash gap
			ax2 = fig.add_subplot(gs[0, 1])
			if 'nash_gap' in self.episode_metrics:
				episodes = range(len(self.episode_metrics['nash_gap']))
				ax2.plot(episodes, self.episode_metrics['nash_gap'], 
						'r-', linewidth=2)
				ax2.set_xlabel('Episode')
				ax2.set_ylabel('Nash Gap')
				ax2.set_title('Stackelberg-Nash Gap')
				ax2.grid(True, alpha=0.3)
			
			# Plot 3: System metrics
			ax3 = fig.add_subplot(gs[1, 0])
			if 'avg_voltage_violations' in self.episode_metrics:
				episodes = range(len(self.episode_metrics['avg_voltage_violations']))
				ax3.plot(episodes, self.episode_metrics['avg_voltage_violations'], 
						'b-', label='Voltage Violations')
				ax3_twin = ax3.twinx()
				if 'avg_power_losses' in self.episode_metrics:
					ax3_twin.plot(episodes, self.episode_metrics['avg_power_losses'], 
								'g-', label='Power Losses')
				ax3.set_xlabel('Episode')
				ax3.set_ylabel('Voltage Violations', color='b')
				ax3_twin.set_ylabel('Power Loss Ratio', color='g')
				ax3.set_title('System Performance')
				ax3.grid(True, alpha=0.3)
			
			# Plot 4: Carbon emissions
			ax4 = fig.add_subplot(gs[1, 1])
			if 'avg_carbon_emissions' in self.episode_metrics:
				episodes = range(len(self.episode_metrics['avg_carbon_emissions']))
				ax4.plot(episodes, self.episode_metrics['avg_carbon_emissions'], 
						'k-', linewidth=2)
				ax4.set_xlabel('Episode')
				ax4.set_ylabel('Carbon Emissions (kg CO2)')
				ax4.set_title('Carbon Emissions')
				ax4.grid(True, alpha=0.3)
			
			# Plot 5: Policy convergence
			ax5 = fig.add_subplot(gs[2, :])
			if self.policy_distances:
				for agent_id, distances in self.policy_distances.items():
					if distances:
						steps = range(len(distances))
						ax5.semilogy(steps, distances, 
									label=f'Agent {agent_id}', alpha=0.7)
				ax5.set_xlabel('Update Steps')
				ax5.set_ylabel('Policy Distance (log scale)')
				ax5.set_title('Policy Convergence')
				ax5.legend()
				ax5.grid(True, alpha=0.3)
			
			# Save figure
			plt.tight_layout()
			plt.savefig(
				os.path.join(self.exp_dir, "metrics_plot.png"),
				dpi=300, bbox_inches='tight'
			)
			plt.close()
			
			self.logger.info("Metrics plotted")
			
		except ImportError:
			self.logger.warning("Matplotlib not available for plotting")
	
	def save_monitoring_data(self):
		"""Save all monitoring data."""
		self.save_metrics()
		self.plot_metrics()
		
		# Save final summary
		summary = self.get_summary_statistics()
		summary['timestamp'] = datetime.now().isoformat()
		
		with open(os.path.join(self.exp_dir, "final_summary.json"), 'w') as f:
			json.dump(summary, f, indent=2)
		
		self.logger.info(f"All monitoring data saved to {self.exp_dir}")
	
	def close(self):
		"""Clean up and save final data."""
		self.save_monitoring_data()