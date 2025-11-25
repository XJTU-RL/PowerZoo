"""
Unit tests for utility functions.

Tests cover configuration tools, environment tools, model tools, and transformation tools.
"""

import pytest
import numpy as np
import torch
import os
import tempfile


@pytest.mark.unit
class TestConfigsTools:
	"""Test configuration tools."""

	def test_is_json_serializable(self):
		"""Test JSON serialization check."""
		from utils.configs_tools import is_json_serializable

		# Test serializable types
		assert is_json_serializable("string")
		assert is_json_serializable(123)
		assert is_json_serializable(12.34)
		assert is_json_serializable([1, 2, 3])
		assert is_json_serializable({"key": "value"})
		assert is_json_serializable(None)
		assert is_json_serializable(True)

		# Test non-serializable types
		assert not is_json_serializable(lambda x: x)
		assert not is_json_serializable(set([1, 2, 3]))

	def test_convert_json(self):
		"""Test JSON conversion."""
		from utils.configs_tools import convert_json

		# Test basic types
		assert convert_json("string") == "string"
		assert convert_json(123) == 123
		assert convert_json([1, 2, 3]) == [1, 2, 3]

		# Test dict
		result = convert_json({"a": 1, "b": 2})
		assert result == {"a": 1, "b": 2}

	def test_get_task_name(self):
		"""Test task name retrieval."""
		from utils.configs_tools import get_task_name

		env_args = {"env_name": "test_env"}
		assert get_task_name("powerzoo", env_args) == "test_env"
		assert get_task_name("powerzoo_llm", env_args) == "test_env"
		assert get_task_name("unknown_env", env_args) == "unknown"


@pytest.mark.unit
class TestTransTools:
	"""Test transformation tools."""

	def test_t2n_conversion(self):
		"""Test tensor to numpy conversion."""
		from utils.trans_tools import _t2n

		tensor = torch.tensor([1.0, 2.0, 3.0])
		array = _t2n(tensor)

		assert isinstance(array, np.ndarray)
		np.testing.assert_array_almost_equal(array, [1.0, 2.0, 3.0])

	def test_flatten(self):
		"""Test tensor flattening."""
		from utils.trans_tools import _flatten

		T, N = 4, 8
		value = torch.randn(T, N, 10)
		flattened = _flatten(T, N, value)

		assert flattened.shape == (T * N, 10)

	def test_sa_cast(self):
		"""Test single-agent buffer casting."""
		from utils.trans_tools import _sa_cast

		# Shape: (episode_length, n_rollout_threads, dim)
		value = np.random.randn(10, 4, 5)
		result = _sa_cast(value)

		# Should be (n_rollout_threads * episode_length, dim)
		assert result.shape == (40, 5)

	def test_ma_cast(self):
		"""Test multi-agent buffer casting."""
		from utils.trans_tools import _ma_cast

		# Shape: (episode_length, n_rollout_threads, num_agents, dim)
		value = np.random.randn(10, 4, 3, 5)
		result = _ma_cast(value)

		# Should be (n_rollout_threads * num_agents * episode_length, dim)
		assert result.shape == (120, 5)

	def test_avail_choose(self):
		"""Test available action choosing."""
		from utils.trans_tools import avail_choose

		x = torch.ones(4, 5)
		avail = torch.tensor([
			[1, 1, 0, 0, 1],
			[1, 0, 1, 1, 0],
			[0, 1, 1, 0, 1],
			[1, 1, 1, 1, 1],
		])

		result = avail_choose(x, avail)

		# Unavailable actions should have very negative values
		assert result[0, 2] == pytest.approx(-1e10)
		assert result[0, 3] == pytest.approx(-1e10)
		assert result[1, 1] == pytest.approx(-1e10)

	def test_is_discrete(self):
		"""Test discrete space detection."""
		from utils.trans_tools import is_discrete
		from gym.spaces import Discrete, Box

		assert is_discrete(Discrete(5))
		assert not is_discrete(Box(low=-1, high=1, shape=(3,)))

	def test_make_onehot(self):
		"""Test one-hot encoding."""
		from utils.trans_tools import make_onehot

		actions = np.array([0, 2, 1])
		onehot = make_onehot(actions, 4)

		expected = np.array([
			[1, 0, 0, 0],
			[0, 0, 1, 0],
			[0, 1, 0, 0],
		])
		np.testing.assert_array_equal(onehot, expected)

	def test_get_dim_from_space(self):
		"""Test space dimension extraction."""
		from utils.trans_tools import get_dim_from_space
		from gym.spaces import Discrete, Box

		assert get_dim_from_space(Discrete(5)) == 5
		assert get_dim_from_space(Box(low=-1, high=1, shape=(10,))) == 10


@pytest.mark.unit
class TestModelsTools:
	"""Test model tools."""

	def test_init_device(self):
		"""Test device initialization."""
		from utils.models_tools import init_device

		args = {
			"cuda": False,
			"cuda_deterministic": False,
			"torch_threads": 1,
		}

		device = init_device(args)
		assert device == torch.device("cpu")

	def test_get_active_func(self):
		"""Test activation function retrieval."""
		from utils.models_tools import get_active_func

		relu = get_active_func("relu")
		assert isinstance(relu, torch.nn.ReLU)

		tanh = get_active_func("tanh")
		assert isinstance(tanh, torch.nn.Tanh)

		sigmoid = get_active_func("sigmoid")
		assert isinstance(sigmoid, torch.nn.Sigmoid)

	def test_get_grad_norm(self):
		"""Test gradient norm calculation."""
		from utils.models_tools import get_grad_norm

		# Create simple model
		model = torch.nn.Linear(10, 5)
		x = torch.randn(4, 10)
		y = model(x).sum()
		y.backward()

		grad_norm = get_grad_norm(model.parameters())
		assert grad_norm >= 0

	def test_huber_loss(self):
		"""Test Huber loss."""
		from utils.models_tools import huber_loss

		e = torch.tensor([0.5, 1.5, 2.0])
		d = 1.0
		loss = huber_loss(e, d)

		assert loss.shape == e.shape
		# For |e| <= d: loss = e^2 / 2
		assert loss[0] == pytest.approx(0.125)
		# For |e| > d: loss = d * (|e| - d/2)
		assert loss[1] == pytest.approx(1.0)

	def test_mse_loss(self):
		"""Test MSE loss."""
		from utils.models_tools import mse_loss

		e = torch.tensor([1.0, 2.0, 3.0])
		loss = mse_loss(e)

		expected = torch.tensor([0.5, 2.0, 4.5])
		torch.testing.assert_close(loss, expected)


@pytest.mark.unit
class TestEnvsTools:
	"""Test environment tools."""

	def test_check_function(self):
		"""Test check function for tensor conversion."""
		from utils.envs_tools import check

		# Test numpy array
		arr = np.array([1.0, 2.0, 3.0])
		tensor = check(arr)
		assert isinstance(tensor, torch.Tensor)

		# Test already tensor
		t = torch.tensor([1.0, 2.0, 3.0])
		result = check(t)
		assert isinstance(result, torch.Tensor)

	def test_get_shape_from_obs_space(self):
		"""Test observation space shape extraction."""
		from utils.envs_tools import get_shape_from_obs_space
		from gym.spaces import Box, Discrete

		box_space = Box(low=-1, high=1, shape=(10, 20), dtype=np.float32)
		assert get_shape_from_obs_space(box_space) == [10, 20]

		discrete_space = Discrete(5)
		assert get_shape_from_obs_space(discrete_space) == [1]
