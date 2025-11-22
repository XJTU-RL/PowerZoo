"""
Space Utilities Tests
=====================

Tests for the space utility functions in envs.common.space_utils.
"""

import numpy as np
import pytest
from typing import List


class MockDiscreteSpace:
	"""Mock Discrete space for testing."""

	def __init__(self, n: int):
		self.n = n


class MockMultiDiscreteSpace:
	"""Mock MultiDiscrete space for testing."""

	def __init__(self, nvec: List[int]):
		self.nvec = np.array(nvec)


class MockBoxSpace:
	"""Mock Box space for testing."""

	def __init__(self, low: np.ndarray, high: np.ndarray, shape: tuple):
		self.low = low
		self.high = high
		self.shape = shape
		self.dtype = np.float32


class TestValidateSpaces:
	"""Tests for validate_spaces function."""

	def test_validate_spaces_correct_lengths(self):
		"""Test validation passes with correct lengths."""
		from envs.common.space_utils import validate_spaces

		obs_spaces = [MockBoxSpace(np.zeros(5), np.ones(5), (5,)) for _ in range(3)]
		act_spaces = [MockDiscreteSpace(4) for _ in range(3)]

		assert validate_spaces(obs_spaces, act_spaces, 3) is True

	def test_validate_spaces_wrong_obs_length(self):
		"""Test validation fails with wrong observation space length."""
		from envs.common.space_utils import validate_spaces

		obs_spaces = [MockBoxSpace(np.zeros(5), np.ones(5), (5,)) for _ in range(2)]
		act_spaces = [MockDiscreteSpace(4) for _ in range(3)]

		with pytest.raises(ValueError) as exc_info:
			validate_spaces(obs_spaces, act_spaces, 3)
		assert "Observation space length" in str(exc_info.value)

	def test_validate_spaces_wrong_act_length(self):
		"""Test validation fails with wrong action space length."""
		from envs.common.space_utils import validate_spaces

		obs_spaces = [MockBoxSpace(np.zeros(5), np.ones(5), (5,)) for _ in range(3)]
		act_spaces = [MockDiscreteSpace(4) for _ in range(2)]

		with pytest.raises(ValueError) as exc_info:
			validate_spaces(obs_spaces, act_spaces, 3)
		assert "Action space length" in str(exc_info.value)


class TestGetAvailActionsFromSpaces:
	"""Tests for get_avail_actions_from_spaces function."""

	def test_discrete_spaces(self):
		"""Test available actions for discrete spaces."""
		from envs.common.space_utils import get_avail_actions_from_spaces

		spaces = [MockDiscreteSpace(5), MockDiscreteSpace(3)]
		avail = get_avail_actions_from_spaces(spaces)

		assert len(avail) == 2
		assert avail[0] == [1, 1, 1, 1, 1]
		assert avail[1] == [1, 1, 1]

	def test_multidiscrete_spaces(self):
		"""Test available actions for MultiDiscrete spaces."""
		from envs.common.space_utils import get_avail_actions_from_spaces

		spaces = [MockMultiDiscreteSpace([2, 3])]  # 2 * 3 = 6 combinations
		avail = get_avail_actions_from_spaces(spaces)

		assert len(avail) == 1
		assert len(avail[0]) == 6

	def test_continuous_spaces(self):
		"""Test available actions for continuous spaces."""
		from envs.common.space_utils import get_avail_actions_from_spaces

		spaces = [MockBoxSpace(np.zeros(4), np.ones(4), (4,))]
		avail = get_avail_actions_from_spaces(spaces)

		assert len(avail) == 1
		assert avail[0] == [1]


class TestGetSpaceDim:
	"""Tests for get_space_dim function."""

	def test_box_space_dim(self):
		"""Test dimension for Box space."""
		from envs.common.space_utils import get_space_dim

		space = MockBoxSpace(np.zeros(10), np.ones(10), (10,))
		assert get_space_dim(space) == 10

	def test_discrete_space_dim(self):
		"""Test dimension for Discrete space."""
		from envs.common.space_utils import get_space_dim

		space = MockDiscreteSpace(7)
		assert get_space_dim(space) == 7

	def test_multidiscrete_space_dim(self):
		"""Test dimension for MultiDiscrete space."""
		from envs.common.space_utils import get_space_dim

		space = MockMultiDiscreteSpace([2, 3, 4])
		assert get_space_dim(space) == 3  # Number of dimensions


class TestIsDiscreteSpace:
	"""Tests for is_discrete_space function."""

	def test_discrete_is_discrete(self):
		"""Test Discrete space is recognized as discrete."""
		from envs.common.space_utils import is_discrete_space

		space = MockDiscreteSpace(5)
		assert is_discrete_space(space) is True

	def test_multidiscrete_is_discrete(self):
		"""Test MultiDiscrete space is recognized as discrete."""
		from envs.common.space_utils import is_discrete_space

		space = MockMultiDiscreteSpace([2, 3])
		assert is_discrete_space(space) is True

	def test_box_is_not_discrete(self):
		"""Test Box space is not recognized as discrete."""
		from envs.common.space_utils import is_discrete_space

		space = MockBoxSpace(np.zeros(5), np.ones(5), (5,))
		assert is_discrete_space(space) is False


class TestIsContinuousSpace:
	"""Tests for is_continuous_space function."""

	def test_box_is_continuous(self):
		"""Test Box space is recognized as continuous."""
		from envs.common.space_utils import is_continuous_space

		space = MockBoxSpace(np.zeros(5), np.ones(5), (5,))
		assert is_continuous_space(space) is True

	def test_discrete_is_not_continuous(self):
		"""Test Discrete space is not recognized as continuous."""
		from envs.common.space_utils import is_continuous_space

		space = MockDiscreteSpace(5)
		assert is_continuous_space(space) is False


class TestFlattenAction:
	"""Tests for flatten_action function."""

	def test_flatten_int_action(self):
		"""Test flattening integer action."""
		from envs.common.space_utils import flatten_action

		action = 3
		space = MockDiscreteSpace(5)
		result = flatten_action(action, space)

		assert isinstance(result, np.ndarray)
		assert result.shape == (1,)
		assert result[0] == 3

	def test_flatten_array_action(self):
		"""Test flattening array action."""
		from envs.common.space_utils import flatten_action

		action = np.array([[1, 2], [3, 4]])
		space = MockBoxSpace(np.zeros(4), np.ones(4), (4,))
		result = flatten_action(action, space)

		assert result.shape == (4,)
		np.testing.assert_array_equal(result, [1, 2, 3, 4])


class TestClipAction:
	"""Tests for clip_action function."""

	def test_clip_action_within_bounds(self):
		"""Test action within bounds is not modified."""
		from envs.common.space_utils import clip_action

		action = np.array([0.5, 0.5])
		space = MockBoxSpace(np.zeros(2), np.ones(2), (2,))
		result = clip_action(action, space)

		np.testing.assert_array_equal(result, action)

	def test_clip_action_above_bounds(self):
		"""Test action above bounds is clipped."""
		from envs.common.space_utils import clip_action

		action = np.array([1.5, 2.0])
		space = MockBoxSpace(np.zeros(2), np.ones(2), (2,))
		result = clip_action(action, space)

		np.testing.assert_array_equal(result, [1.0, 1.0])

	def test_clip_action_below_bounds(self):
		"""Test action below bounds is clipped."""
		from envs.common.space_utils import clip_action

		action = np.array([-0.5, -1.0])
		space = MockBoxSpace(np.zeros(2), np.ones(2), (2,))
		result = clip_action(action, space)

		np.testing.assert_array_equal(result, [0.0, 0.0])


class TestCheckSpaceCompatibility:
	"""Tests for check_space_compatibility function."""

	def test_same_discrete_spaces_compatible(self):
		"""Test same Discrete spaces are compatible."""
		from envs.common.space_utils import check_space_compatibility

		space1 = MockDiscreteSpace(5)
		space2 = MockDiscreteSpace(5)
		assert check_space_compatibility(space1, space2) is True

	def test_different_discrete_spaces_incompatible(self):
		"""Test different Discrete spaces are incompatible."""
		from envs.common.space_utils import check_space_compatibility

		space1 = MockDiscreteSpace(5)
		space2 = MockDiscreteSpace(10)
		assert check_space_compatibility(space1, space2) is False

	def test_same_box_spaces_compatible(self):
		"""Test same Box spaces are compatible."""
		from envs.common.space_utils import check_space_compatibility

		space1 = MockBoxSpace(np.zeros(5), np.ones(5), (5,))
		space2 = MockBoxSpace(np.zeros(5), np.ones(5), (5,))
		assert check_space_compatibility(space1, space2) is True

	def test_different_box_shapes_incompatible(self):
		"""Test Box spaces with different shapes are incompatible."""
		from envs.common.space_utils import check_space_compatibility

		space1 = MockBoxSpace(np.zeros(5), np.ones(5), (5,))
		space2 = MockBoxSpace(np.zeros(10), np.ones(10), (10,))
		assert check_space_compatibility(space1, space2) is False

	def test_different_types_incompatible(self):
		"""Test spaces of different types are incompatible."""
		from envs.common.space_utils import check_space_compatibility

		space1 = MockDiscreteSpace(5)
		space2 = MockBoxSpace(np.zeros(5), np.ones(5), (5,))
		assert check_space_compatibility(space1, space2) is False


# Run tests if executed directly
if __name__ == "__main__":
	pytest.main([__file__, "-v"])
