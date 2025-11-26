"""
Unit tests for neural network models.

Tests cover policy models, value function models, and base network components.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete


@pytest.mark.unit
class TestMLPBase:
	"""Test MLP base network."""

	def test_mlp_base_import(self):
		"""Test that MLPBase can be imported."""
		try:
			from models.base.mlp import MLPBase
			assert MLPBase is not None
		except ImportError as e:
			pytest.fail(f"Failed to import MLPBase: {e}")

	def test_mlp_base_forward(self):
		"""Test MLPBase forward pass."""
		from models.base.mlp import MLPBase

		args = {
			"hidden_sizes": [64, 64],
			"activation_func": "relu",
			"use_feature_normalization": False,
			"initialization_method": "orthogonal_",
		}

		obs_shape = [10]
		mlp = MLPBase(args, obs_shape)

		# Test forward pass
		batch_size = 4
		x = torch.randn(batch_size, 10)
		output = mlp(x)

		assert output.shape == (batch_size, 64)


@pytest.mark.unit
class TestRNNLayer:
	"""Test RNN layer."""

	def test_rnn_layer_import(self):
		"""Test that RNNLayer can be imported."""
		try:
			from models.base.rnn import RNNLayer
			assert RNNLayer is not None
		except ImportError as e:
			pytest.fail(f"Failed to import RNNLayer: {e}")


@pytest.mark.unit
class TestACTLayer:
	"""Test ACT layer."""

	def test_act_layer_import(self):
		"""Test that ACTLayer can be imported."""
		try:
			from models.base.act import ACTLayer
			assert ACTLayer is not None
		except ImportError as e:
			pytest.fail(f"Failed to import ACTLayer: {e}")


@pytest.mark.unit
class TestDistributions:
	"""Test distribution classes."""

	def test_categorical_distribution_import(self):
		"""Test that Categorical distribution can be imported."""
		try:
			from models.base.distributions import Categorical
			assert Categorical is not None
		except ImportError as e:
			pytest.fail(f"Failed to import Categorical: {e}")

	def test_diagonal_gaussian_import(self):
		"""Test that DiagGaussian can be imported."""
		try:
			from models.base.distributions import DiagGaussian
			assert DiagGaussian is not None
		except ImportError as e:
			pytest.fail(f"Failed to import DiagGaussian: {e}")


@pytest.mark.unit
class TestVNet:
	"""Test value network."""

	def test_vnet_import(self):
		"""Test that VNet can be imported."""
		try:
			from models.value_function_models.v_net import VNet
			assert VNet is not None
		except ImportError as e:
			pytest.fail(f"Failed to import VNet: {e}")

	def test_vnet_forward(self):
		"""Test VNet forward pass."""
		from models.value_function_models.v_net import VNet

		args = {
			"hidden_sizes": [64, 64],
			"gain": 0.01,
			"initialization_method": "orthogonal_",
			"use_naive_recurrent_policy": False,
			"use_recurrent_policy": False,
			"recurrent_n": 1,
			"activation_func": "relu",
			"use_feature_normalization": False,
		}

		obs_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
		vnet = VNet(args, obs_space, device=torch.device("cpu"))

		batch_size = 4
		obs = torch.randn(batch_size, 20)
		rnn_states = torch.zeros(batch_size, 1, 64)
		masks = torch.ones(batch_size, 1)

		values, new_rnn_states = vnet(obs, rnn_states, masks)

		assert values.shape == (batch_size, 1)


@pytest.mark.unit
class TestStochasticPolicy:
	"""Test stochastic policy model."""

	def test_stochastic_policy_import(self):
		"""Test that StochasticPolicy can be imported."""
		try:
			from models.policy_models.stochastic_policy import StochasticPolicy
			assert StochasticPolicy is not None
		except ImportError as e:
			pytest.fail(f"Failed to import StochasticPolicy: {e}")


@pytest.mark.unit
class TestDeterministicPolicy:
	"""Test deterministic policy model."""

	def test_deterministic_policy_import(self):
		"""Test that DeterministicPolicy can be imported."""
		try:
			from models.policy_models.deterministic_policy import DeterministicPolicy
			assert DeterministicPolicy is not None
		except ImportError as e:
			pytest.fail(f"Failed to import DeterministicPolicy: {e}")


@pytest.mark.unit
class TestQMixModels:
	"""Test QMix-related models."""

	def test_agent_q_function_import(self):
		"""Test that AgentQFunction can be imported."""
		try:
			from models.value_function_models.agent_q_function import AgentQFunction
			assert AgentQFunction is not None
		except ImportError as e:
			pytest.fail(f"Failed to import AgentQFunction: {e}")

	def test_mqmixer_import(self):
		"""Test that M_QMixer can be imported."""
		try:
			from models.value_function_models.mq_mixer import M_QMixer
			assert M_QMixer is not None
		except ImportError as e:
			pytest.fail(f"Failed to import M_QMixer: {e}")
