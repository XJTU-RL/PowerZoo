"""
Unit tests for common modules.

Tests cover buffers, value normalization, and logging utilities.
"""

import pytest
import numpy as np
import torch


@pytest.mark.unit
class TestValueNorm:
	"""Test value normalization."""

	def test_valuenorm_import(self):
		"""Test that ValueNorm can be imported."""
		try:
			from common.valuenorm import ValueNorm
			assert ValueNorm is not None
		except ImportError as e:
			pytest.fail(f"Failed to import ValueNorm: {e}")

	def test_valuenorm_update_normalize(self):
		"""Test ValueNorm update and normalize functionality."""
		from common.valuenorm import ValueNorm

		valuenorm = ValueNorm(input_shape=(1,))

		# Generate some values
		values = np.random.randn(100) * 10 + 5

		# Update with values
		for v in values:
			valuenorm.update(np.array([[v]]))

		# Test normalization
		test_value = np.array([[10.0]])
		normalized = valuenorm.normalize(test_value)
		assert normalized is not None

		# Test denormalization
		denormalized = valuenorm.denormalize(normalized)
		np.testing.assert_array_almost_equal(denormalized, test_value, decimal=1)


@pytest.mark.unit
class TestBaseLogger:
	"""Test base logger functionality."""

	def test_base_logger_import(self):
		"""Test that BaseLogger can be imported."""
		try:
			from common.base_logger import BaseLogger
			assert BaseLogger is not None
		except ImportError as e:
			pytest.fail(f"Failed to import BaseLogger: {e}")


@pytest.mark.unit
class TestOnPolicyActorBuffer:
	"""Test on-policy actor buffer."""

	def test_buffer_import(self):
		"""Test that OnPolicyActorBuffer can be imported."""
		try:
			from common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
			assert OnPolicyActorBuffer is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyActorBuffer: {e}")

	def test_buffer_creation(self):
		"""Test buffer creation and basic operations."""
		from common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
		from gym.spaces import Box, Discrete

		args = {
			"use_gae": True,
			"gamma": 0.99,
			"gae_lambda": 0.95,
			"use_proper_time_limits": False,
			"use_naive_recurrent_policy": False,
			"use_recurrent_policy": False,
			"recurrent_n": 1,
			"hidden_sizes": [64, 64],
		}

		obs_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
		act_space = Discrete(5)

		buffer = OnPolicyActorBuffer(
			args=args,
			obs_space=obs_space,
			act_space=act_space,
			n_rollout_threads=4,
			episode_length=100,
		)

		assert buffer is not None
		assert hasattr(buffer, "obs")
		assert hasattr(buffer, "actions")


@pytest.mark.unit
class TestOnPolicyCriticBuffer:
	"""Test on-policy critic buffers."""

	def test_ep_buffer_import(self):
		"""Test that OnPolicyCriticBufferEP can be imported."""
		try:
			from common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
			assert OnPolicyCriticBufferEP is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyCriticBufferEP: {e}")

	def test_fp_buffer_import(self):
		"""Test that OnPolicyCriticBufferFP can be imported."""
		try:
			from common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
			assert OnPolicyCriticBufferFP is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyCriticBufferFP: {e}")


@pytest.mark.unit
class TestOffPolicyBuffer:
	"""Test off-policy buffers."""

	def test_ep_buffer_import(self):
		"""Test that OffPolicyBufferEP can be imported."""
		try:
			from common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
			assert OffPolicyBufferEP is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OffPolicyBufferEP: {e}")

	def test_fp_buffer_import(self):
		"""Test that OffPolicyBufferFP can be imported."""
		try:
			from common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
			assert OffPolicyBufferFP is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OffPolicyBufferFP: {e}")
