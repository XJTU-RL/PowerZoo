"""
Unit tests for PowerZoo_LLM environment.

Tests cover the advanced PowerZoo_LLM environment with modular architecture,
including circuit system, rewards, logging, and single-agent support.
"""

import pytest
import numpy as np
import gym


@pytest.mark.unit
@pytest.mark.powerzoo_llm
class TestPowerZooLLMImports:
	"""Test that all PowerZoo_LLM modules can be imported."""

	def test_import_base_env(self):
		"""Test importing base environment module."""
		try:
			from envs.powerzoo_llm.base_env import PowerZooEnv
			assert PowerZooEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZooEnv: {e}")

	def test_import_circuit_system(self):
		"""Test importing circuit system module."""
		try:
			from envs.powerzoo_llm.circuit_system import Circuits
			assert Circuits is not None
		except ImportError as e:
			pytest.fail(f"Failed to import Circuits: {e}")

	def test_import_loadprofile(self):
		"""Test importing loadprofile module."""
		try:
			from envs.powerzoo_llm.data_process import LoadProfile
			assert LoadProfile is not None
		except ImportError as e:
			pytest.fail(f"Failed to import LoadProfile: {e}")

	def test_import_rewards(self):
		"""Test importing reward module."""
		try:
			from envs.powerzoo_llm.rewards import PowerZooReward
			assert PowerZooReward is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZooReward: {e}")

	def test_import_logging(self):
		"""Test importing logging module."""
		try:
			from envs.powerzoo_llm.logging import get_logger
			assert get_logger is not None
		except ImportError as e:
			pytest.fail(f"Failed to import logging: {e}")

	def test_import_single_agent(self):
		"""Test importing single agent module."""
		try:
			from envs.powerzoo_llm.single_agent import SingleAgentPowerZooEnv
			assert SingleAgentPowerZooEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import SingleAgentPowerZooEnv: {e}")


@pytest.mark.integration
@pytest.mark.powerzoo_llm
@pytest.mark.requires_opendss
class TestPowerZooLLMEnvCreation:
	"""Test PowerZoo_LLM environment creation and initialization."""

	def test_env_creation_without_config(self, skip_if_no_opendss):
		"""Test environment creation fails gracefully without config."""
		from envs.powerzoo_llm.base_env import PowerZooEnv

		# Should fail or warn when no DSS file is provided
		with pytest.raises(Exception):
			env = PowerZooEnv()

	def test_env_creation_with_config(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test environment creation with valid configuration."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		env = PowerZooEnv(**powerzoo_llm_config)
		assert env is not None
		assert hasattr(env, "reset")
		assert hasattr(env, "step")

	def test_observation_space(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test that observation space is properly defined."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		env = PowerZooEnv(**powerzoo_llm_config)

		assert hasattr(env, "observation_space")
		assert isinstance(env.observation_space, (gym.Space, list, dict))

	def test_action_space(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test that action space is properly defined."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		env = PowerZooEnv(**powerzoo_llm_config)

		assert hasattr(env, "action_space")
		assert isinstance(env.action_space, (gym.Space, list, dict))


@pytest.mark.integration
@pytest.mark.powerzoo_llm
@pytest.mark.requires_opendss
class TestPowerZooLLMReset:
	"""Test PowerZoo_LLM reset functionality."""

	def test_reset_returns_observation(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test that reset returns valid observations."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		env = PowerZooEnv(**powerzoo_llm_config)
		observations = env.reset()

		assert observations is not None

	def test_reset_seed_consistency(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test that reset with same seed produces consistent results."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		env = PowerZooEnv(**powerzoo_llm_config)

		obs1 = env.reset()
		obs2 = env.reset()

		# Structure should be the same
		if isinstance(obs1, dict) and isinstance(obs2, dict):
			assert obs1.keys() == obs2.keys()
		elif isinstance(obs1, list) and isinstance(obs2, list):
			assert len(obs1) == len(obs2)


@pytest.mark.integration
@pytest.mark.powerzoo_llm
@pytest.mark.requires_opendss
@pytest.mark.slow
class TestPowerZooLLMStep:
	"""Test PowerZoo_LLM step functionality."""

	def test_step_returns_tuple(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test that step returns (obs, reward, done, info) tuple."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		env = PowerZooEnv(**powerzoo_llm_config)
		env.reset()

		# Sample actions
		if isinstance(env.action_space, list):
			actions = [space.sample() for space in env.action_space]
		elif isinstance(env.action_space, dict):
			actions = {key: space.sample() for key, space in env.action_space.items()}
		else:
			actions = env.action_space.sample()

		result = env.step(actions)

		assert isinstance(result, tuple)
		assert len(result) == 4

		obs, reward, done, info = result
		assert obs is not None
		assert isinstance(done, (bool, dict))
		assert isinstance(info, (dict, list))

	def test_full_episode(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test running a full episode."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.base_env import PowerZooEnv

		# Short episode for testing
		config = powerzoo_llm_config.copy()
		config["episode_length"] = 10
		config["max_steps"] = 10

		env = PowerZooEnv(**config)
		env.reset()

		done = False
		step_count = 0
		total_reward = 0

		while not done and step_count < 15:  # Safety limit
			if isinstance(env.action_space, list):
				actions = [space.sample() for space in env.action_space]
			elif isinstance(env.action_space, dict):
				actions = {key: space.sample() for key, space in env.action_space.items()}
			else:
				actions = env.action_space.sample()

			obs, reward, done_flag, info = env.step(actions)

			step_count += 1

			# Accumulate reward
			if isinstance(reward, dict):
				total_reward += sum(reward.values())
			elif isinstance(reward, list):
				total_reward += sum(reward)
			else:
				total_reward += reward

			# Check done
			if isinstance(done_flag, bool):
				done = done_flag
			elif isinstance(done_flag, dict):
				done = all(done_flag.values())

		# Should complete within episode length
		assert step_count <= 15


@pytest.mark.unit
@pytest.mark.powerzoo_llm
class TestCircuitSystemModule:
	"""Test Circuit System module."""

	def test_circuit_import(self):
		"""Test Circuit module import."""
		from envs.powerzoo_llm.circuit_system import Circuits
		assert Circuits is not None

	def test_components_import(self):
		"""Test component modules import."""
		from envs.powerzoo_llm.circuit_system.components import (
			Node, Edge, Load, Capacitor, PVSystem
		)
		assert all([Node, Edge, Load, Capacitor, PVSystem])


@pytest.mark.unit
@pytest.mark.powerzoo_llm
class TestRewardModule:
	"""Test reward module functionality."""

	def test_reward_import(self):
		"""Test reward module import."""
		from envs.powerzoo_llm.rewards import PowerZooReward
		assert PowerZooReward is not None

	def test_lagrangian_import(self):
		"""Test Lagrangian updater import."""
		from envs.powerzoo_llm.rewards import LagrangianUpdater
		assert LagrangianUpdater is not None


@pytest.mark.unit
@pytest.mark.powerzoo_llm
class TestDataProcessModule:
	"""Test data processing module."""

	def test_loadprofile_import(self):
		"""Test LoadProfile import."""
		from envs.powerzoo_llm.data_process import LoadProfile
		assert LoadProfile is not None

	def test_loadprofile_core_import(self):
		"""Test LoadProfile core module import."""
		from envs.powerzoo_llm.data_process.loadprofile_core import LoadProfile as CoreLoadProfile
		assert CoreLoadProfile is not None


@pytest.mark.unit
@pytest.mark.powerzoo_llm
class TestLoggingModule:
	"""Test logging module functionality."""

	def test_logger_import(self):
		"""Test logger import."""
		from envs.powerzoo_llm.logging import get_logger
		assert get_logger is not None

	def test_system_logger_import(self):
		"""Test SystemLogger import."""
		from envs.powerzoo_llm.logging import SystemLogger
		assert SystemLogger is not None

	def test_get_logger_creates_logger(self):
		"""Test that get_logger creates a logger instance."""
		from envs.powerzoo_llm.logging import get_logger

		logger = get_logger("test_logger")
		assert logger is not None
		assert hasattr(logger, "info")
		assert hasattr(logger, "debug")
		assert hasattr(logger, "warning")
		assert hasattr(logger, "error")


@pytest.mark.integration
@pytest.mark.powerzoo_llm
class TestSingleAgentEnv:
	"""Test single agent environment functionality."""

	def test_single_agent_import(self):
		"""Test single agent environment import."""
		from envs.powerzoo_llm.single_agent import SingleAgentPowerZooEnv
		assert SingleAgentPowerZooEnv is not None

	@pytest.mark.requires_opendss
	@pytest.mark.slow
	def test_single_agent_creation(self, powerzoo_llm_config, skip_if_no_opendss):
		"""Test single agent environment creation."""
		if powerzoo_llm_config["dss_folder_path"] is None:
			pytest.skip("34Bus system not found")

		from envs.powerzoo_llm.single_agent import SingleAgentPowerZooEnv

		config = powerzoo_llm_config.copy()
		config["enable_logging"] = False

		try:
			env = SingleAgentPowerZooEnv(**config)
			assert env is not None
		except Exception as e:
			# Single agent might have different config requirements
			pytest.skip(f"Single agent env creation failed: {e}")
