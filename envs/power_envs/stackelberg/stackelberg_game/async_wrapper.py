# -*- coding: utf-8 -*-
"""
Asynchronous Multi-Agent Wrapper for Stackelberg Game

This module implements asynchronous execution for the Stackelberg game,
where UC acts first as leader and consumers respond as followers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

from envs.power_envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.power_envs.stackelberg.stackelberg_game.stackelberg_monitor import StackelbergMonitor


class AsyncMultiAgentWrapper:
	"""
	Wrapper for asynchronous multi-agent execution in Stackelberg game.
	
	Implements the temporal structure where:
	1. UC (leader) makes decisions first
	2. Consumers (followers) observe UC actions and respond
	3. System state updates based on combined actions
	"""
	
	def __init__(self, base_env, config: Optional[Dict[str, Any]] = None):
		"""
		Initialize async wrapper.
		
		Args:
			base_env: Base Stackelberg environment
			config: Configuration dictionary
		"""
		self.env = base_env
		self.config = config or {}
		self.logger = logging.getLogger('AsyncStackelberg')
		
		# Async execution parameters
		self.consumer_delay = config.get('consumer_delay', 1)
		self.action_persistence = config.get('action_persistence', 1)
		self.enable_prediction = config.get('enable_action_prediction', False)
		self.partial_observability = config.get('partial_observability', False)
		self.observation_noise = config.get('observation_noise', 0.0)
		
		# Phase tracking
		self.current_phase = 'reset'  # 'uc_decision', 'consumer_response', 'system_update'
		self.phase_step = 0
		
		# Action buffers
		self.pending_uc_action = None
		self.pending_consumer_actions = {}
		self.executed_actions = defaultdict(lambda: None)
		
		# State tracking
		self.last_observations = {}
		self.action_history = []
		
		# Monitoring
		self.monitor = config.get('monitor', None)
		
	def reset(self, load_profile_idx: Optional[int] = None) -> Dict[int, np.ndarray]:
		"""
		Reset environment and return initial observations.
		
		Args:
			load_profile_idx: Optional load profile index
			
		Returns:
			Initial observations for all agents
		"""
		# Reset base environment
		observations = self.env.reset(load_profile_idx)
		
		# Reset wrapper state
		self.current_phase = 'uc_decision'
		self.phase_step = 0
		self.pending_uc_action = None
		self.pending_consumer_actions.clear()
		self.executed_actions.clear()
		self.action_history.clear()
		
		# Add noise if partial observability is enabled
		if self.partial_observability and self.observation_noise > 0:
			observations = self._add_observation_noise(observations)
		
		# Store observations
		self.last_observations = observations.copy()
		
		return observations
	
	def step(self, actions: Dict[int, np.ndarray]) -> Tuple[
		Dict[int, np.ndarray], Dict[int, float], bool, Dict[int, Dict[str, Any]]
	]:
		"""
		Execute one step of the async Stackelberg game.
		
		Args:
			actions: Dictionary mapping agent_id to action
			
		Returns:
			observations: Next observations
			rewards: Agent rewards
			done: Episode termination flag
			infos: Additional information
		"""
		# Separate UC and consumer actions
		uc_action = actions.get(self.env.uc_agent_id, None)
		consumer_actions = {
			aid: act for aid, act in actions.items() 
			if aid in self.env.consumer_agent_ids
		}
		
		# Execute based on current phase
		if self.current_phase == 'uc_decision':
			return self._execute_uc_phase(uc_action)
		elif self.current_phase == 'consumer_response':
			return self._execute_consumer_phase(consumer_actions)
		else:
			raise ValueError(f"Invalid phase: {self.current_phase}")
	
	def _execute_uc_phase(self, uc_action: Optional[np.ndarray]) -> Tuple[
		Dict[int, np.ndarray], Dict[int, float], bool, Dict[int, Dict[str, Any]]
	]:
		"""Execute UC decision phase."""
		if uc_action is None:
			raise ValueError("UC action required in UC decision phase")
		
		# Store UC action
		self.pending_uc_action = uc_action
		
		# Execute UC action in environment
		uc_reward, uc_info = self.env.step_uc(uc_action)
		
		# Get current observations (UC action is now visible to consumers)
		observations = self.env.get_observations()
		
		# Add observation noise if enabled
		if self.partial_observability and self.observation_noise > 0:
			observations = self._add_observation_noise(observations)
		
		# Prepare rewards (only UC gets reward in this phase)
		rewards = {self.env.uc_agent_id: uc_reward}
		for aid in self.env.consumer_agent_ids:
			rewards[aid] = 0.0
		
		# Prepare infos
		infos = {self.env.uc_agent_id: uc_info}
		for aid in self.env.consumer_agent_ids:
			infos[aid] = {'phase': 'waiting_for_uc'}
		
		# Update phase
		self.current_phase = 'consumer_response'
		self.phase_step += 1
		
		# Store observations
		self.last_observations = observations.copy()
		
		# Episode not done yet
		done = False
		
		return observations, rewards, done, infos
	
	def _execute_consumer_phase(self, consumer_actions: Dict[int, np.ndarray]) -> Tuple[
		Dict[int, np.ndarray], Dict[int, float], bool, Dict[int, Dict[str, Any]]
	]:
		"""Execute consumer response phase."""
		# Validate all consumers have actions
		missing_consumers = set(self.env.consumer_agent_ids) - set(consumer_actions.keys())
		if missing_consumers:
			# Use previous actions or default for missing consumers
			for cid in missing_consumers:
				if cid in self.executed_actions:
					consumer_actions[cid] = self.executed_actions[cid]
				else:
					# Default action (no change)
					consumer_actions[cid] = np.zeros_like(self.env.action_spaces[cid].sample())
		
		# Store consumer actions
		self.pending_consumer_actions = consumer_actions
		
		# Execute combined step in environment
		rewards, infos, done = self.env.step_consumers(consumer_actions)
		
		# Get next observations
		if not done:
			observations = self.env.get_observations()
			
			# Add noise if enabled
			if self.partial_observability and self.observation_noise > 0:
				observations = self._add_observation_noise(observations)
		else:
			# Use last observations if episode is done
			observations = self.last_observations
		
		# Store executed actions
		self.executed_actions[self.env.uc_agent_id] = self.pending_uc_action
		for cid, action in consumer_actions.items():
			self.executed_actions[cid] = action
		
		# Record action history
		self.action_history.append({
			'uc_action': self.pending_uc_action.copy(),
			'consumer_actions': {k: v.copy() for k, v in consumer_actions.items()},
			'step': self.phase_step
		})
		
		# Update phase for next step
		if not done:
			self.current_phase = 'uc_decision'
			self.phase_step += 1
			self.last_observations = observations.copy()
		
		return observations, rewards, done, infos
	
	def _add_observation_noise(self, observations: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
		"""Add Gaussian noise to observations for partial observability."""
		noisy_obs = {}
		
		for agent_id, obs in observations.items():
			noise = np.random.normal(0, self.observation_noise, obs.shape)
			noisy_obs[agent_id] = obs + noise
		
		return noisy_obs
	
	def get_agent_types(self) -> Dict[int, str]:
		"""Get agent type mapping."""
		return self.env.agent_types
	
	def get_phase_info(self) -> Dict[str, Any]:
		"""Get current phase information."""
		return {
			'current_phase': self.current_phase,
			'phase_step': self.phase_step,
			'pending_uc_action': self.pending_uc_action is not None,
			'pending_consumer_count': len(self.pending_consumer_actions)
		}
	
	def get_action_history(self) -> List[Dict[str, Any]]:
		"""Get action history for current episode."""
		return self.action_history.copy()
	
	def predict_consumer_response(self, uc_action: np.ndarray) -> Dict[int, np.ndarray]:
		"""
		Predict consumer responses to UC action.
		
		This can be used by UC for anticipatory planning.
		
		Args:
			uc_action: Proposed UC action
			
		Returns:
			Predicted consumer actions
		"""
		if not self.enable_prediction:
			return {}
		
		# Simple prediction based on historical responses
		# In practice, this would use learned models
		predicted_actions = {}
		
		for cid in self.env.consumer_agent_ids:
			# Placeholder: predict based on price signal
			price_signal = uc_action[0] if len(uc_action) > 0 else 1.0
			
			# Higher price -> more load reduction
			load_reduction = -0.1 * (price_signal - 1.0)
			load_reduction = np.clip(load_reduction, -0.3, 0.1)
			
			# Create predicted action
			action_dim = self.env.action_spaces[cid].shape[0]
			predicted_action = np.zeros(action_dim)
			predicted_action[0] = load_reduction
			
			predicted_actions[cid] = predicted_action
		
		return predicted_actions
	
	def set_monitor(self, monitor):
		"""Set monitoring object."""
		self.monitor = monitor
	
	def close(self):
		"""Clean up resources."""
		if hasattr(self.env, 'close'):
			self.env.close()
	
	def __getattr__(self, name):
		"""Forward unknown attributes to base environment."""
		return getattr(self.env, name)