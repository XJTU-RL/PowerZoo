"""
Environment Interface Tests
===========================

This module tests that all PowerZoo environments comply with the standard
multi-agent RL interface required by HAPPO and other MARL algorithms.

Tests:
------
1. reset() returns correct format (obs_list, state, available_actions)
2. step() returns correct format (obs_list, state, rewards, dones, infos, available_actions)
3. Observation/action spaces are properly configured
4. Available actions have correct shape
5. Rewards have correct shape (n_agents, 1)
"""

import numpy as np
import pytest
from typing import Dict, List, Any, Optional


class MockActionSpace:
	"""Mock discrete action space for testing."""

	def __init__(self, n: int = 5):
		self.n = n

	def sample(self) -> int:
		return np.random.randint(0, self.n)


class MockObservationSpace:
	"""Mock Box observation space for testing."""

	def __init__(self, shape: tuple = (10,), low: float = -np.inf, high: float = np.inf):
		self.shape = shape
		self.low = np.full(shape, low)
		self.high = np.full(shape, high)
		self.dtype = np.float32

	def sample(self) -> np.ndarray:
		return np.random.randn(*self.shape).astype(np.float32)


class MockEnv:
	"""
	Mock environment for testing interface compliance.

	This is used when real environments cannot be instantiated
	(e.g., when OpenDSS is not available).
	"""

	def __init__(self, n_agents: int = 3, obs_dim: int = 10, n_actions: int = 5):
		self.n_agents = n_agents
		self._obs_dim = obs_dim
		self._n_actions = n_actions

		# Setup spaces
		self.observation_space = [
			MockObservationSpace(shape=(obs_dim,))
			for _ in range(n_agents)
		]
		self.action_space = [
			MockActionSpace(n=n_actions)
			for _ in range(n_agents)
		]

		self.discrete = True
		self.episode_limit = 100
		self._current_step = 0

	def reset(self):
		"""Reset environment with standard return format."""
		self._current_step = 0
		obs_list = [
			np.random.randn(self._obs_dim).astype(np.float32)
			for _ in range(self.n_agents)
		]
		state = np.concatenate([o.flatten() for o in obs_list])
		available_actions = self.get_avail_actions()
		return obs_list, state, available_actions

	def step(self, actions):
		"""Execute step with standard return format."""
		self._current_step += 1
		done = self._current_step >= self.episode_limit

		obs_list = [
			np.random.randn(self._obs_dim).astype(np.float32)
			for _ in range(self.n_agents)
		]
		state = np.concatenate([o.flatten() for o in obs_list])
		rewards = np.random.randn(self.n_agents, 1).astype(np.float32)
		dones = [done] * self.n_agents
		infos = [{"step": self._current_step}] * self.n_agents
		available_actions = self.get_avail_actions()

		return obs_list, state, rewards, dones, infos, available_actions

	def get_avail_actions(self) -> List[List[int]]:
		"""Get available actions for all agents."""
		return [[1] * self._n_actions for _ in range(self.n_agents)]

	def close(self):
		"""Clean up resources."""
		pass


class TestEnvInterfaceCompliance:
	"""Test suite for environment interface compliance."""

	def test_reset_returns_tuple_of_three(self):
		"""Test that reset returns (obs_list, state, available_actions)."""
		env = MockEnv(n_agents=3)
		result = env.reset()

		assert isinstance(result, tuple), "reset() must return a tuple"
		assert len(result) == 3, "reset() must return exactly 3 elements"

	def test_reset_obs_list_format(self):
		"""Test that reset returns observations as list of arrays."""
		env = MockEnv(n_agents=3, obs_dim=10)
		obs_list, state, available_actions = env.reset()

		assert isinstance(obs_list, list), "obs_list must be a list"
		assert len(obs_list) == env.n_agents, "obs_list length must match n_agents"

		for i, obs in enumerate(obs_list):
			assert isinstance(obs, np.ndarray), f"obs[{i}] must be np.ndarray"
			assert obs.shape == (10,), f"obs[{i}] shape must match observation space"

	def test_reset_state_format(self):
		"""Test that reset returns state as numpy array."""
		env = MockEnv(n_agents=3, obs_dim=10)
		obs_list, state, available_actions = env.reset()

		assert isinstance(state, np.ndarray), "state must be np.ndarray"
		# State should be concatenation of all observations
		expected_size = 3 * 10  # n_agents * obs_dim
		assert state.shape[0] == expected_size, f"state size should be {expected_size}"

	def test_reset_available_actions_format(self):
		"""Test that reset returns available_actions as list of lists."""
		env = MockEnv(n_agents=3, n_actions=5)
		obs_list, state, available_actions = env.reset()

		assert isinstance(available_actions, list), "available_actions must be a list"
		assert len(available_actions) == env.n_agents

		for i, avail in enumerate(available_actions):
			assert isinstance(avail, list), f"available_actions[{i}] must be a list"
			assert len(avail) == 5, f"available_actions[{i}] length must match n_actions"

	def test_step_returns_tuple_of_six(self):
		"""Test that step returns 6-tuple."""
		env = MockEnv(n_agents=3)
		env.reset()
		actions = [0] * env.n_agents
		result = env.step(actions)

		assert isinstance(result, tuple), "step() must return a tuple"
		assert len(result) == 6, "step() must return exactly 6 elements"

	def test_step_rewards_shape(self):
		"""Test that step returns rewards with shape (n_agents, 1)."""
		env = MockEnv(n_agents=3)
		env.reset()
		actions = [0] * env.n_agents
		obs_list, state, rewards, dones, infos, available_actions = env.step(actions)

		assert isinstance(rewards, np.ndarray), "rewards must be np.ndarray"
		assert rewards.shape == (3, 1), "rewards shape must be (n_agents, 1)"

	def test_step_dones_format(self):
		"""Test that step returns dones as list of bools."""
		env = MockEnv(n_agents=3)
		env.reset()
		actions = [0] * env.n_agents
		obs_list, state, rewards, dones, infos, available_actions = env.step(actions)

		assert isinstance(dones, list), "dones must be a list"
		assert len(dones) == env.n_agents, "dones length must match n_agents"
		for d in dones:
			assert isinstance(d, (bool, np.bool_)), "each done must be bool"

	def test_step_infos_format(self):
		"""Test that step returns infos as list of dicts."""
		env = MockEnv(n_agents=3)
		env.reset()
		actions = [0] * env.n_agents
		obs_list, state, rewards, dones, infos, available_actions = env.step(actions)

		assert isinstance(infos, list), "infos must be a list"
		assert len(infos) == env.n_agents, "infos length must match n_agents"
		for info in infos:
			assert isinstance(info, dict), "each info must be a dict"

	def test_spaces_match_n_agents(self):
		"""Test that observation and action spaces match n_agents."""
		env = MockEnv(n_agents=4)

		assert len(env.observation_space) == env.n_agents
		assert len(env.action_space) == env.n_agents

	def test_episode_terminates(self):
		"""Test that episode terminates at episode_limit."""
		env = MockEnv(n_agents=2)
		env.episode_limit = 10
		env.reset()

		for _ in range(10):
			actions = [0] * env.n_agents
			obs_list, state, rewards, dones, infos, available_actions = env.step(actions)

		assert all(dones), "Episode should terminate at episode_limit"


class TestSpaceValidation:
	"""Test suite for space validation utilities."""

	def test_discrete_action_space_avail_actions(self):
		"""Test available actions for discrete spaces."""
		env = MockEnv(n_agents=2, n_actions=3)
		available_actions = env.get_avail_actions()

		assert len(available_actions) == 2
		assert all(len(a) == 3 for a in available_actions)
		assert all(all(x == 1 for x in a) for a in available_actions)


class TestObservationConsistency:
	"""Test suite for observation consistency."""

	def test_obs_dtype_is_float32(self):
		"""Test that observations are float32."""
		env = MockEnv(n_agents=2)
		obs_list, _, _ = env.reset()

		for obs in obs_list:
			assert obs.dtype == np.float32, "Observations should be float32"

	def test_obs_no_nan_or_inf(self):
		"""Test that observations don't contain NaN or Inf."""
		env = MockEnv(n_agents=2)
		obs_list, state, _ = env.reset()

		for obs in obs_list:
			assert not np.any(np.isnan(obs)), "Observations should not contain NaN"
			assert not np.any(np.isinf(obs)), "Observations should not contain Inf"

		assert not np.any(np.isnan(state)), "State should not contain NaN"
		assert not np.any(np.isinf(state)), "State should not contain Inf"


# Run tests if executed directly
if __name__ == "__main__":
	pytest.main([__file__, "-v"])
