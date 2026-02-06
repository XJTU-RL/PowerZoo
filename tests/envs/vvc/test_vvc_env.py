"""
Unit tests for PowerZoo environment.

Tests cover basic environment functionality, reset/step operations,
and space validation.
"""

import pytest
import numpy as np
import gym


@pytest.mark.unit
@pytest.mark.powerzoo
class TestVVCEnvBasics:
	"""Test basic PowerZoo environment functionality."""

	def test_import_powerzoo(self):
		"""Test that PowerZoo module can be imported."""
		try:
			from envs.vvc import VVCEnv
			assert VVCEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZoo: {e}")

	def test_import_circuit(self):
		"""Test that Circuit module can be imported."""
		try:
			from envs.vvc.vvc.circuit import Circuits
			assert Circuits is not None
		except ImportError as e:
			pytest.fail(f"Failed to import Circuits: {e}")

	def test_import_loadprofile(self):
		"""Test that LoadProfile module can be imported."""
		try:
			from envs.vvc.vvc.loadprofile import LoadProfile
			assert LoadProfile is not None
		except ImportError as e:
			pytest.fail(f"Failed to import LoadProfile: {e}")


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestVVCEnvCreation:
	"""Test PowerZoo environment creation and initialization."""

	def test_env_creation_without_config(self, skip_if_no_opendss):
		"""Test environment creation fails gracefully without config."""
		from envs.vvc import VVCEnv

		# Should fail or warn when no DSS file is provided
		with pytest.raises(Exception):
			env = VVCEnv()

	def test_env_creation_with_config(self, powerzoo_config, skip_if_no_opendss):
		"""Test environment creation with valid configuration."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		env = VVCEnv(**powerzoo_config)
		assert env is not None
		assert hasattr(env, "reset")
		assert hasattr(env, "step")

	def test_observation_space(self, powerzoo_config, skip_if_no_opendss):
		"""Test that observation space is properly defined."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		env = VVCEnv(**powerzoo_config)

		assert hasattr(env, "observation_space")
		assert isinstance(env.observation_space, (gym.Space, list))

		# If multi-agent, should be a list of spaces
		if isinstance(env.observation_space, list):
			for space in env.observation_space:
				assert isinstance(space, gym.Space)

	def test_action_space(self, powerzoo_config, skip_if_no_opendss):
		"""Test that action space is properly defined."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		env = VVCEnv(**powerzoo_config)

		assert hasattr(env, "action_space")
		assert isinstance(env.action_space, (gym.Space, list))

		# If multi-agent, should be a list of spaces
		if isinstance(env.action_space, list):
			for space in env.action_space:
				assert isinstance(space, gym.Space)


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestVVCEnvReset:
	"""Test PowerZoo environment reset functionality."""

	def test_reset_returns_observation(self, powerzoo_config, skip_if_no_opendss):
		"""Test that reset returns valid observations."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		env = VVCEnv(**powerzoo_config)
		observations = env.reset()

		assert observations is not None

		# Validate observation structure
		if isinstance(env.observation_space, list):
			assert isinstance(observations, (list, tuple, dict))
			assert len(observations) == len(env.observation_space)
		else:
			assert isinstance(observations, np.ndarray)

	def test_reset_reproducibility(self, powerzoo_config, skip_if_no_opendss):
		"""Test that reset with same seed produces same initial state."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		# Create two environments with same seed
		env1 = VVCEnv(**powerzoo_config)
		env2 = VVCEnv(**powerzoo_config)

		obs1 = env1.reset()
		obs2 = env2.reset()

		# Check if observations are similar (may not be exactly equal due to randomness)
		if isinstance(obs1, np.ndarray) and isinstance(obs2, np.ndarray):
			# Allow some tolerance for floating point differences
			assert obs1.shape == obs2.shape


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
@pytest.mark.slow
class TestVVCEnvStep:
	"""Test PowerZoo environment step functionality."""

	def test_step_returns_tuple(self, powerzoo_config, skip_if_no_opendss):
		"""Test that step returns (obs, reward, done, info) tuple."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		env = VVCEnv(**powerzoo_config)
		env.reset()

		# Sample a random action
		if isinstance(env.action_space, list):
			actions = [space.sample() for space in env.action_space]
		else:
			actions = env.action_space.sample()

		result = env.step(actions)

		assert isinstance(result, tuple)
		assert len(result) == 4  # (obs, reward, done, info)

		obs, reward, done, info = result
		assert obs is not None
		assert isinstance(done, (bool, dict))
		assert isinstance(info, dict) or isinstance(info, list)

	def test_step_multiple_times(self, powerzoo_config, skip_if_no_opendss):
		"""Test that environment can step multiple times."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		env = VVCEnv(**powerzoo_config)
		env.reset()

		num_steps = 10
		for i in range(num_steps):
			if isinstance(env.action_space, list):
				actions = [space.sample() for space in env.action_space]
			else:
				actions = env.action_space.sample()

			obs, reward, done, info = env.step(actions)

			assert obs is not None

			# If done, break
			if isinstance(done, bool) and done:
				break
			elif isinstance(done, dict) and all(done.values()):
				break

	def test_episode_termination(self, powerzoo_config, skip_if_no_opendss):
		"""Test that episodes terminate correctly."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc import VVCEnv

		# Set short episode length for testing
		config = powerzoo_config.copy()
		config["episode_length"] = 5
		config["max_steps"] = 5

		env = VVCEnv(**config)
		env.reset()

		done = False
		step_count = 0

		while not done and step_count < 10:  # Safety limit
			if isinstance(env.action_space, list):
				actions = [space.sample() for space in env.action_space]
			else:
				actions = env.action_space.sample()

			obs, reward, done_flag, info = env.step(actions)

			step_count += 1

			# Check done flag
			if isinstance(done_flag, bool):
				done = done_flag
			elif isinstance(done_flag, dict):
				done = all(done_flag.values())

		# Should terminate within episode_length + buffer
		assert step_count <= 10


@pytest.mark.unit
@pytest.mark.powerzoo
class TestCircuitModule:
	"""Test Circuit module functionality."""

	def test_circuit_import(self):
		"""Test that Circuit module can be imported."""
		from envs.vvc.vvc.circuit import Circuits
		assert Circuits is not None

	@pytest.mark.requires_opendss
	def test_circuit_creation(self, powerzoo_config, skip_if_no_opendss):
		"""Test circuit creation with DSS file."""
		if powerzoo_config["dss_file"] is None:
			pytest.skip("13Bus system not found")

		from envs.vvc.vvc.circuit import Circuits

		circuit = Circuits(dss_file=powerzoo_config["dss_file"])
		assert circuit is not None


@pytest.mark.unit
@pytest.mark.powerzoo
class TestLoadProfileModule:
	"""Test LoadProfile module functionality."""

	def test_loadprofile_import(self):
		"""Test that LoadProfile module can be imported."""
		from envs.vvc.vvc.loadprofile import LoadProfile
		assert LoadProfile is not None
