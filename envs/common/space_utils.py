"""
Space Utilities
===============

Utility functions for working with Gym/Gymnasium action and observation spaces.

Features:
---------
- Space validation
- Available actions computation
- Space compatibility checking
"""

from typing import Any, List, Optional, Union

import numpy as np


def validate_spaces(
	observation_space: List,
	action_space: List,
	n_agents: int
) -> bool:
	"""
	Validate that observation and action spaces match the number of agents.

	Args:
		observation_space: List of observation spaces
		action_space: List of action spaces
		n_agents: Number of agents

	Returns:
		True if spaces are valid

	Raises:
		ValueError: If spaces are invalid
	"""
	if len(observation_space) != n_agents:
		raise ValueError(
			f"Observation space length ({len(observation_space)}) "
			f"does not match n_agents ({n_agents})"
		)

	if len(action_space) != n_agents:
		raise ValueError(
			f"Action space length ({len(action_space)}) "
			f"does not match n_agents ({n_agents})"
		)

	return True


def get_avail_actions_from_spaces(action_spaces: List) -> List[List[int]]:
	"""
	Compute available actions from action spaces.

	For discrete spaces, returns [1, 1, ..., 1] with length = n_actions.
	For continuous spaces, returns [1] indicating all actions available.

	Args:
		action_spaces: List of action spaces

	Returns:
		List of available action masks for each agent
	"""
	available_actions = []
	for space in action_spaces:
		if hasattr(space, 'n'):
			# Discrete action space
			available_actions.append([1] * space.n)
		elif hasattr(space, 'nvec'):
			# MultiDiscrete action space - total combinations
			total = int(np.prod(space.nvec))
			available_actions.append([1] * total)
		else:
			# Continuous action space
			available_actions.append([1])
	return available_actions


def get_space_dim(space: Any) -> int:
	"""
	Get the dimension of a space.

	Args:
		space: A Gym/Gymnasium space

	Returns:
		The dimension of the space
	"""
	if hasattr(space, 'shape'):
		return int(np.prod(space.shape))
	elif hasattr(space, 'n'):
		return space.n
	elif hasattr(space, 'nvec'):
		return len(space.nvec)
	return 0


def is_discrete_space(space: Any) -> bool:
	"""
	Check if a space is discrete.

	Args:
		space: A Gym/Gymnasium space

	Returns:
		True if the space is discrete (Discrete or MultiDiscrete)
	"""
	return hasattr(space, 'n') or hasattr(space, 'nvec')


def is_continuous_space(space: Any) -> bool:
	"""
	Check if a space is continuous (Box).

	Args:
		space: A Gym/Gymnasium space

	Returns:
		True if the space is continuous
	"""
	return hasattr(space, 'low') and hasattr(space, 'high') and not hasattr(space, 'n')


def flatten_action(
	action: Union[int, np.ndarray],
	action_space: Any
) -> np.ndarray:
	"""
	Flatten an action to a 1D array.

	Args:
		action: The action to flatten
		action_space: The corresponding action space

	Returns:
		Flattened action array
	"""
	if isinstance(action, (int, np.integer)):
		return np.array([action])
	return np.asarray(action).flatten()


def clip_action(
	action: np.ndarray,
	action_space: Any
) -> np.ndarray:
	"""
	Clip an action to be within the action space bounds.

	Args:
		action: The action to clip
		action_space: The corresponding action space

	Returns:
		Clipped action
	"""
	if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
		return np.clip(action, action_space.low, action_space.high)
	return action


def check_space_compatibility(space1: Any, space2: Any) -> bool:
	"""
	Check if two spaces are compatible (same type and shape).

	Args:
		space1: First space
		space2: Second space

	Returns:
		True if spaces are compatible
	"""
	# Check type
	if type(space1) != type(space2):
		return False

	# Check shape for Box spaces
	if hasattr(space1, 'shape') and hasattr(space2, 'shape'):
		if space1.shape != space2.shape:
			return False

	# Check n for Discrete spaces
	if hasattr(space1, 'n') and hasattr(space2, 'n'):
		if space1.n != space2.n:
			return False

	# Check nvec for MultiDiscrete spaces
	if hasattr(space1, 'nvec') and hasattr(space2, 'nvec'):
		if not np.array_equal(space1.nvec, space2.nvec):
			return False

	return True
