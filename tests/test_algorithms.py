"""
Unit tests for algorithm implementations.

Tests cover actor algorithms, critic algorithms, and their basic functionality.
"""

import pytest
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete


@pytest.mark.unit
class TestHAPPOAlgorithm:
	"""Test HAPPO algorithm basic functionality."""

	def test_happo_import(self):
		"""Test that HAPPO can be imported."""
		try:
			from algorithms.actors.happo import HAPPO
			assert HAPPO is not None
		except ImportError as e:
			pytest.fail(f"Failed to import HAPPO: {e}")

	def test_happo_creation(self):
		"""Test HAPPO instance creation."""
		from algorithms.actors.happo import HAPPO

		# Create mock args
		from utils.config_pipeline import ConfigLoader
		cfg = ConfigLoader().load(experiment='configs/experiments/vvc_13bus_smoke.yaml')
		args = {**cfg.algo_args['train'], **cfg.algo_args['model'], **cfg.algo_args['algo']}

		obs_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
		act_space = Discrete(5)

		actor = HAPPO(args, obs_space, act_space, device=torch.device("cpu"))
		assert actor is not None
		assert hasattr(actor, "actor")
		assert hasattr(actor, "update")
		assert hasattr(actor, "train")


@pytest.mark.unit
class TestMAPPOAlgorithm:
	"""Test MAPPO algorithm basic functionality."""

	def test_mappo_import(self):
		"""Test that MAPPO can be imported."""
		try:
			from algorithms.actors.mappo import MAPPO
			assert MAPPO is not None
		except ImportError as e:
			pytest.fail(f"Failed to import MAPPO: {e}")


@pytest.mark.unit
class TestVCritic:
	"""Test VCritic algorithm basic functionality."""

	def test_vcritic_import(self):
		"""Test that VCritic can be imported."""
		try:
			from algorithms.critics.v_critic import VCritic
			assert VCritic is not None
		except ImportError as e:
			pytest.fail(f"Failed to import VCritic: {e}")

	def test_vcritic_creation(self):
		"""Test VCritic instance creation."""
		from algorithms.critics.v_critic import VCritic

		from utils.config_pipeline import ConfigLoader
		cfg = ConfigLoader().load(experiment='configs/experiments/vvc_13bus_smoke.yaml')
		args = {**cfg.algo_args['train'], **cfg.algo_args['model'], **cfg.algo_args['algo']}

		cent_obs_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

		critic = VCritic(args, cent_obs_space, device=torch.device("cpu"))
		assert critic is not None
		assert hasattr(critic, "critic")
		assert hasattr(critic, "update")
		assert hasattr(critic, "train")


@pytest.mark.unit
class TestOnPolicyBase:
	"""Test on-policy base class."""

	def test_on_policy_base_import(self):
		"""Test that OnPolicyBase can be imported."""
		try:
			from algorithms.actors.on_policy_base import OnPolicyBase
			assert OnPolicyBase is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyBase: {e}")


@pytest.mark.unit
class TestOffPolicyBase:
	"""Test off-policy base class."""

	def test_off_policy_base_import(self):
		"""Test that OffPolicyBase can be imported."""
		try:
			from algorithms.actors.off_policy_base import OffPolicyBase
			assert OffPolicyBase is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OffPolicyBase: {e}")


@pytest.mark.unit
class TestSNMAPPO:
	"""Test SN-MAPPO algorithm."""

	def test_sn_mappo_import(self):
		"""Test that SN_MAPPO can be imported."""
		try:
			from algorithms.actors.sn_mappo import SN_MAPPO
			assert SN_MAPPO is not None
		except ImportError as e:
			pytest.fail(f"Failed to import SN_MAPPO: {e}")
