"""
Pytest configuration and shared fixtures for PowerZoo tests.

This module provides common fixtures and utilities used across all test suites.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
import gym
import gymnasium


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# Session-level fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
	"""Return the project root directory."""
	return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
	"""Return the test data directory."""
	data_dir = project_root / "tests" / "data"
	data_dir.mkdir(exist_ok=True)
	return data_dir


@pytest.fixture(scope="session")
def node_systems_dir(project_root: Path) -> Path:
	"""Return the node_systems directory containing IEEE test systems."""
	return project_root / "node_systems"


# ==============================================================================
# Module-level fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def temp_output_dir():
	"""Create a temporary directory for test outputs."""
	with tempfile.TemporaryDirectory() as tmpdir:
		yield Path(tmpdir)


# ==============================================================================
# Function-level fixtures
# ==============================================================================

@pytest.fixture
def random_seed():
	"""Set random seed for reproducibility."""
	seed = 42
	np.random.seed(seed)
	return seed


@pytest.fixture
def mock_config() -> Dict[str, Any]:
	"""Return a mock configuration dictionary for testing."""
	return {
		"env_name": "test_env",
		"num_agents": 3,
		"episode_length": 96,
		"seed": 42,
	}


# ==============================================================================
# PowerZoo-specific fixtures
# ==============================================================================

@pytest.fixture
def powerzoo_config(node_systems_dir: Path) -> Dict[str, Any]:
	"""Return configuration for PowerZoo environment."""
	system_13bus = node_systems_dir / "13Bus"

	config = {
		"dss_file": str(system_13bus / "IEEE13Nodeckt.dss") if system_13bus.exists() else None,
		"num_agents": 3,
		"episode_length": 96,
		"max_steps": 96,
		"seed": 42,
		"use_sparse_matrix": True,
		"voltage_limits": (0.95, 1.05),
	}
	return config


@pytest.fixture
def skip_if_no_opendss():
	"""Skip test if OpenDSS is not installed."""
	try:
		import dss as opendss
		return True
	except ImportError:
		pytest.skip("OpenDSS not installed")


# ==============================================================================
# SmartGrid-specific fixtures
# ==============================================================================

@pytest.fixture
def smartgrid_config(node_systems_dir: Path) -> Dict[str, Any]:
	"""Return configuration for SmartGrid environment."""
	system_34bus = node_systems_dir / "34Bus_PV_Aggressive"

	config = {
		"dss_folder_path": str(system_34bus) if system_34bus.exists() else None,
		"dss_file": "ieee34Mod1_duty.dss",
		"num_agents": 10,
		"episode_length": 96,
		"max_steps": 96,
		"seed": 42,
		"reward_type": "powerzoo",
		"use_sparse_matrix": True,
		"enable_logging": False,  # Disable logging in tests
	}
	return config


# ==============================================================================
# Gym/Gymnasium environment fixtures
# ==============================================================================

@pytest.fixture
def gym_env():
	"""Create a simple Gym environment for testing."""
	env = gym.make("CartPole-v1")
	yield env
	env.close()


@pytest.fixture
def gymnasium_env():
	"""Create a simple Gymnasium environment for testing."""
	env = gymnasium.make("CartPole-v1")
	yield env
	env.close()


# ==============================================================================
# Test utilities
# ==============================================================================

def assert_valid_gym_space(space):
	"""Assert that a space is a valid Gym/Gymnasium space."""
	from gym.spaces import Space as GymSpace
	from gymnasium.spaces import Space as GymnasiumSpace

	assert isinstance(space, (GymSpace, GymnasiumSpace)), \
		f"Expected Gym/Gymnasium space, got {type(space)}"


def assert_valid_observation(observation, observation_space):
	"""Assert that an observation is valid for the given space."""
	assert observation_space.contains(observation), \
		f"Observation {observation} not in space {observation_space}"


def assert_valid_action(action, action_space):
	"""Assert that an action is valid for the given space."""
	assert action_space.contains(action), \
		f"Action {action} not in space {action_space}"


# ==============================================================================
# Pytest hooks
# ==============================================================================

def pytest_configure(config):
	"""Configure pytest with custom settings."""
	config.addinivalue_line(
		"markers", "unit: Unit tests (fast, isolated)"
	)
	config.addinivalue_line(
		"markers", "integration: Integration tests (slower)"
	)
	config.addinivalue_line(
		"markers", "slow: Slow tests"
	)
	config.addinivalue_line(
		"markers", "powerzoo: PowerZoo environment tests"
	)
	config.addinivalue_line(
		"markers", "smartgrid: SmartGrid environment tests"
	)
	config.addinivalue_line(
		"markers", "requires_opendss: Requires OpenDSS installation"
	)
	config.addinivalue_line(
		"markers", "requires_gpu: Requires GPU/CUDA"
	)


def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add markers automatically."""
	for item in items:
		# Add markers based on test path
		if "smartgrid" in str(item.fspath):
			item.add_marker(pytest.mark.smartgrid)
		elif "powerzoo" in str(item.fspath):
			item.add_marker(pytest.mark.powerzoo)

		# Mark slow tests
		if "integration" in item.nodeid or "test_full" in item.nodeid:
			item.add_marker(pytest.mark.slow)
