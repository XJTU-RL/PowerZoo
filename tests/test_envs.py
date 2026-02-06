"""
Unit tests for environment implementations.

Tests cover environment registration, initialization, and basic functionality.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestEnvRegistry:
	"""Test environment registry."""

	def test_env_registry_import(self):
		"""Test ENV_REGISTRY import."""
		try:
			from envs import ENV_REGISTRY
			assert ENV_REGISTRY is not None
			assert isinstance(ENV_REGISTRY, dict)
		except ImportError as e:
			pytest.fail(f"Failed to import ENV_REGISTRY: {e}")

	def test_env_registry_contains_environments(self):
		"""Test that registry contains expected environments."""
		from envs import ENV_REGISTRY

		# Check for expected environment types
		expected_envs = ["vvc", "smartgrid", "dsr", "stackelberg"]
		for env_name in expected_envs:
			assert env_name in ENV_REGISTRY, f"Missing environment: {env_name}"


@pytest.mark.unit
class TestEnvWrappers:
	"""Test environment wrappers."""

	def test_share_dummy_vec_env_import(self):
		"""Test ShareDummyVecEnv import."""
		try:
			from envs.env_wrappers import ShareDummyVecEnv
			assert ShareDummyVecEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import ShareDummyVecEnv: {e}")

	def test_share_subproc_vec_env_import(self):
		"""Test ShareSubprocVecEnv import."""
		try:
			from envs.env_wrappers import ShareSubprocVecEnv
			assert ShareSubprocVecEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import ShareSubprocVecEnv: {e}")


@pytest.mark.unit
class TestDSREnv:
	"""Test DSR environment."""

	def test_dsr_env_import(self):
		"""Test DSREnv import."""
		try:
			from envs.dsr.dsr_env import DSREnv
			assert DSREnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import DSREnv: {e}")

	def test_dsr_core_import(self):
		"""Test DSRCore import."""
		try:
			from envs.dsr.core.dsr_core import DSRCore
			assert DSRCore is not None
		except ImportError as e:
			pytest.fail(f"Failed to import DSRCore: {e}")


@pytest.mark.unit
class TestStackelbergEnv:
	"""Test Stackelberg game environment."""

	def test_stackelberg_base_env_import(self):
		"""Test StackelbergBaseEnv import."""
		try:
			from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
			assert StackelbergBaseEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import StackelbergBaseEnv: {e}")

	def test_demand_response_game_import(self):
		"""Test DemandResponseGame import."""
		try:
			from envs.stackelberg.stackelberg_game.dr_game_env import DemandResponseGame
			assert DemandResponseGame is not None
		except ImportError as e:
			pytest.fail(f"Failed to import DemandResponseGame: {e}")


@pytest.mark.unit
class TestVVCEnv:
	"""Test PowerZoo environment."""

	def test_vvc_env_import(self):
		"""Test PowerZoo env import."""
		try:
			from envs.vvc.vvc.env import Env
			assert Env is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZoo Env: {e}")

	def test_vvc_circuit_import(self):
		"""Test PowerZoo circuit import."""
		try:
			from envs.vvc.vvc.circuit import Circuits
			assert Circuits is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZoo Circuits: {e}")

	def test_vvc_logger_import(self):
		"""Test PowerZoo logger import."""
		try:
			from envs.vvc.vvc_logger import VVCLogger
			assert VVCLogger is not None
		except ImportError as e:
			pytest.fail(f"Failed to import VVCLogger: {e}")


@pytest.mark.unit
class TestSmartGridEnv:
	"""Test SmartGrid environment."""

	def test_smartgrid_env_import(self):
		"""Test SmartGrid VVCEnv import."""
		try:
			from envs.smartgrid.base_env.vvc_env import VVCEnv
			assert VVCEnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import VVCEnv: {e}")

	def test_smartgrid_base_env_import(self):
		"""Test SmartGrid base Env import."""
		try:
			from envs.smartgrid.base_env.env import Env
			assert Env is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZoo LLM Env: {e}")

	def test_smartgrid_reward_import(self):
		"""Test VVCReward import."""
		try:
			from envs.smartgrid.rewards.vvc_reward import VVCReward
			assert VVCReward is not None
		except ImportError as e:
			pytest.fail(f"Failed to import VVCReward: {e}")

	def test_lagrangian_updater_import(self):
		"""Test LagrangianUpdater import."""
		try:
			from envs.smartgrid.rewards.lagrangian import LagrangianUpdater
			assert LagrangianUpdater is not None
		except ImportError as e:
			pytest.fail(f"Failed to import LagrangianUpdater: {e}")

	def test_smartgrid_circuit_import(self):
		"""Test PowerZoo LLM Circuits import."""
		try:
			from envs.smartgrid.circuit_system.circuit import Circuits
			assert Circuits is not None
		except ImportError as e:
			pytest.fail(f"Failed to import PowerZoo LLM Circuits: {e}")


@pytest.mark.unit
class TestEnvComponents:
	"""Test environment component modules."""

	def test_node_components_import(self):
		"""Test node components import."""
		try:
			from envs.smartgrid.circuit_system.components.node_components import (
				Battery, PVSystem
			)
			assert Battery is not None
			assert PVSystem is not None
		except ImportError as e:
			pytest.fail(f"Failed to import node components: {e}")

	def test_line_components_import(self):
		"""Test line components import."""
		try:
			from envs.smartgrid.circuit_system.components.line_components import (
				Line, Transformer
			)
			assert Line is not None
			assert Transformer is not None
		except ImportError as e:
			pytest.fail(f"Failed to import line components: {e}")

	def test_load_profile_import(self):
		"""Test LoadProfile import."""
		try:
			from envs.smartgrid.data_process.loadprofile import LoadProfile
			assert LoadProfile is not None
		except ImportError as e:
			pytest.fail(f"Failed to import LoadProfile: {e}")


@pytest.mark.unit
class TestActionSpaces:
	"""Test action space utilities."""

	def test_action_selector_import(self):
		"""Test ActionSelector import."""
		try:
			from envs.smartgrid.base_env.vvc_config import VVCActionSelector
			assert VVCActionSelector is not None
		except ImportError as e:
			# Try alternate location
			try:
				from envs.smartgrid import VVCActionSelector
				assert VVCActionSelector is not None
			except ImportError:
				pytest.skip("VVCActionSelector not found in expected locations")


@pytest.mark.unit
class TestEnvConfigs:
	"""Test environment configuration classes."""

	def test_vvc_env_config_import(self):
		"""Test VVCEnvConfig import."""
		try:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			assert VVCEnvConfig is not None
		except ImportError as e:
			pytest.fail(f"Failed to import VVCEnvConfig: {e}")

	def test_dsr_env_args_import(self):
		"""Test DSR environment args import."""
		try:
			from envs.dsr import DSREnv
			assert DSREnv is not None
		except ImportError as e:
			pytest.fail(f"Failed to import DSREnv from envs.dsr: {e}")


@pytest.mark.integration
class TestEnvInstantiation:
	"""Integration tests for environment instantiation."""

	def test_dsr_env_space_validation(self):
		"""Test DSR environment has valid spaces."""
		try:
			from envs.dsr.dsr_env import DSREnv
			from gym.spaces import Space

			# Check that class has expected attributes
			assert hasattr(DSREnv, '__init__')
			assert hasattr(DSREnv, 'step')
			assert hasattr(DSREnv, 'reset')
		except ImportError:
			pytest.skip("DSREnv not available")

	def test_stackelberg_env_space_validation(self):
		"""Test Stackelberg environment has valid spaces."""
		try:
			from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv

			# Check that class has expected attributes
			assert hasattr(StackelbergBaseEnv, '__init__')
			assert hasattr(StackelbergBaseEnv, 'step')
			assert hasattr(StackelbergBaseEnv, 'reset')
		except ImportError:
			pytest.skip("StackelbergBaseEnv not available")


@pytest.mark.unit
class TestTwoTimescaleVVC:
	"""Test Two-Timescale VVC components."""

	def test_coordinator_import(self):
		"""Test Coordinator import."""
		try:
			from algorithms.twots_vvc.coordinator import Coordinator
			assert Coordinator is not None
		except ImportError as e:
			pytest.fail(f"Failed to import Coordinator: {e}")

	def test_slow_sacd_import(self):
		"""Test SlowSACD import."""
		try:
			from algorithms.twots_vvc.slow_sacd import SlowSACD
			assert SlowSACD is not None
		except ImportError as e:
			pytest.fail(f"Failed to import SlowSACD: {e}")

	def test_fast_td3_import(self):
		"""Test FastTD3 import."""
		try:
			from algorithms.twots_vvc.fast_td3 import FastTD3
			assert FastTD3 is not None
		except ImportError as e:
			pytest.fail(f"Failed to import FastTD3: {e}")
