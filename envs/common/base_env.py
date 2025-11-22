"""
Base Multi-Agent Environment
============================

This module provides an abstract base class for all multi-agent environments
in the PowerZoo project. It defines the standard interface that all environments
must implement to ensure compatibility with HAPPO and other MARL algorithms.

Features:
---------
- Standardized step/reset return formats
- Available actions computation
- Space validation
- Common utility methods
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BaseMultiAgentEnv(ABC):
	"""
	Abstract base class for multi-agent reinforcement learning environments.

	All PowerZoo environments should inherit from this class to ensure
	compatibility with the training infrastructure and MARL algorithms.

	Attributes:
		n_agents: Number of agents in the environment
		observation_space: List of observation spaces for each agent
		action_space: List of action spaces for each agent
		discrete: Whether actions are discrete
		episode_limit: Maximum number of steps per episode

	Standard Return Formats:
	------------------------
	reset() returns:
		Tuple[obs_list, state, available_actions]
		- obs_list: List of np.ndarray, one per agent
		- state: np.ndarray, global state
		- available_actions: List of List[int], action masks

	step() returns:
		Tuple[obs_list, state, rewards, dones, infos, available_actions]
		- obs_list: List of np.ndarray, one per agent
		- state: np.ndarray, global state
		- rewards: np.ndarray of shape (n_agents, 1)
		- dones: List[bool], one per agent
		- infos: List[Dict], one per agent
		- available_actions: List of List[int], action masks
	"""

	def __init__(self):
		"""Initialize the base environment."""
		self._n_agents: int = 0
		self._observation_space: List = []
		self._action_space: List = []
		self._discrete: bool = True
		self._episode_limit: int = 100

	@property
	def n_agents(self) -> int:
		"""Return the number of agents."""
		return self._n_agents

	@n_agents.setter
	def n_agents(self, value: int):
		"""Set the number of agents."""
		self._n_agents = value

	@property
	def observation_space(self) -> List:
		"""Return observation spaces for all agents."""
		return self._observation_space

	@observation_space.setter
	def observation_space(self, value: List):
		"""Set observation spaces."""
		self._observation_space = value

	@property
	def action_space(self) -> List:
		"""Return action spaces for all agents."""
		return self._action_space

	@action_space.setter
	def action_space(self, value: List):
		"""Set action spaces."""
		self._action_space = value

	@property
	def discrete(self) -> bool:
		"""Return whether actions are discrete."""
		return self._discrete

	@discrete.setter
	def discrete(self, value: bool):
		"""Set discrete flag."""
		self._discrete = value

	@property
	def episode_limit(self) -> int:
		"""Return the episode limit."""
		return self._episode_limit

	@episode_limit.setter
	def episode_limit(self, value: int):
		"""Set episode limit."""
		self._episode_limit = value

	@abstractmethod
	def reset(self, **kwargs) -> Tuple[List[np.ndarray], np.ndarray, List[List[int]]]:
		"""
		Reset the environment to initial state.

		Returns:
			Tuple containing:
			- observations: List of observations for each agent
			- state: Global state
			- available_actions: Available action masks for each agent
		"""
		raise NotImplementedError

	@abstractmethod
	def step(self, actions: Union[List, np.ndarray]) -> Tuple[
		List[np.ndarray],  # observations
		np.ndarray,  # state
		np.ndarray,  # rewards
		List[bool],  # dones
		List[Dict],  # infos
		List[List[int]]  # available_actions
	]:
		"""
		Execute one step in the environment.

		Args:
			actions: Actions for all agents

		Returns:
			Tuple containing:
			- observations: List of observations for each agent
			- state: Global state
			- rewards: Rewards array of shape (n_agents, 1)
			- dones: Done flags for each agent
			- infos: Info dicts for each agent
			- available_actions: Available action masks for each agent
		"""
		raise NotImplementedError

	@abstractmethod
	def close(self) -> None:
		"""Clean up environment resources."""
		raise NotImplementedError

	def get_avail_actions(self) -> List[List[int]]:
		"""
		Get available actions for all agents.

		Returns:
			List of available action masks for each agent.
			For discrete spaces, returns list of 1s with length = n_actions.
			For continuous spaces, returns [1] indicating all actions available.
		"""
		available_actions = []
		for i in range(self.n_agents):
			space = self.action_space[i]
			if hasattr(space, 'n'):
				# Discrete action space
				available_actions.append([1] * space.n)
			else:
				# Continuous action space
				available_actions.append([1])
		return available_actions

	def get_obs_size(self) -> int:
		"""
		Get observation size (for first agent, assuming homogeneous).

		Returns:
			Observation dimension
		"""
		if not self.observation_space:
			return 0
		space = self.observation_space[0]
		if hasattr(space, 'shape'):
			return space.shape[0]
		elif hasattr(space, 'n'):
			return space.n
		return 0

	def get_state_size(self) -> int:
		"""
		Get global state size.

		Default implementation: concatenation of all agent observations.
		Override for custom state representations.

		Returns:
			State dimension
		"""
		total = 0
		for space in self.observation_space:
			if hasattr(space, 'shape'):
				total += space.shape[0]
			elif hasattr(space, 'n'):
				total += space.n
		return total

	def get_total_actions(self) -> int:
		"""
		Get total number of actions (for discrete action spaces).

		Returns:
			Number of actions, or -1 for continuous spaces
		"""
		if not self.action_space:
			return -1
		space = self.action_space[0]
		if hasattr(space, 'n'):
			return space.n
		return -1

	def get_env_info(self) -> Dict[str, Any]:
		"""
		Get environment information for training infrastructure.

		Returns:
			Dictionary containing environment metadata
		"""
		return {
			"n_agents": self.n_agents,
			"n_actions": self.get_total_actions(),
			"state_shape": self.get_state_size(),
			"obs_shape": self.get_obs_size(),
			"episode_limit": self.episode_limit,
		}

	def seed(self, seed: Optional[int] = None) -> None:
		"""
		Set random seed for reproducibility.

		Args:
			seed: Random seed value
		"""
		if seed is not None:
			np.random.seed(seed)

	def render(self, mode: str = 'human') -> Optional[np.ndarray]:
		"""
		Render the environment.

		Args:
			mode: Render mode ('human', 'rgb_array', etc.)

		Returns:
			Optional array for rgb_array mode
		"""
		pass  # Default: no rendering

	def validate_spaces(self) -> bool:
		"""
		Validate that observation and action spaces are properly configured.

		Returns:
			True if spaces are valid
		"""
		if len(self.observation_space) != self.n_agents:
			return False
		if len(self.action_space) != self.n_agents:
			return False
		return True

	def _format_rewards(self, rewards: Union[List, np.ndarray]) -> np.ndarray:
		"""
		Format rewards to standard (n_agents, 1) shape.

		Args:
			rewards: Rewards in various formats

		Returns:
			Rewards array of shape (n_agents, 1)
		"""
		rewards = np.asarray(rewards)
		if rewards.ndim == 1:
			return rewards.reshape(-1, 1)
		return rewards

	def _format_observations(
		self,
		obs_dict: Dict[int, np.ndarray]
	) -> Tuple[List[np.ndarray], np.ndarray]:
		"""
		Format observations from dict to standard format.

		Args:
			obs_dict: Dictionary mapping agent_id to observation

		Returns:
			Tuple of (obs_list, global_state)
		"""
		obs_list = [
			obs_dict.get(i, np.zeros(self.get_obs_size()))
			for i in range(self.n_agents)
		]
		state = np.concatenate([o.flatten() for o in obs_list])
		return obs_list, state
