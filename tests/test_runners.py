"""
Unit tests for runner implementations.

Tests cover on-policy and off-policy runners for multi-agent environments.
"""

import pytest
import numpy as np
import torch


@pytest.mark.unit
class TestRunnerImports:
	"""Test runner module imports."""

	def test_on_policy_base_runner_import(self):
		"""Test OnPolicyBaseRunner import."""
		try:
			from runners.on_policy_base_runner import OnPolicyBaseRunner
			assert OnPolicyBaseRunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyBaseRunner: {e}")

	def test_off_policy_base_runner_import(self):
		"""Test OffPolicyBaseRunner import."""
		try:
			from runners.off_policy_base_runner import OffPolicyBaseRunner
			assert OffPolicyBaseRunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OffPolicyBaseRunner: {e}")

	def test_on_policy_ma_runner_import(self):
		"""Test OnPolicyMARunner import."""
		try:
			from runners.on_policy_ma_runner import OnPolicyMARunner
			assert OnPolicyMARunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyMARunner: {e}")

	def test_off_policy_ma_runner_import(self):
		"""Test OffPolicyMARunner import."""
		try:
			from runners.off_policy_ma_runner import OffPolicyMARunner
			assert OffPolicyMARunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OffPolicyMARunner: {e}")

	def test_on_policy_ha_runner_import(self):
		"""Test OnPolicyHARunner import."""
		try:
			from runners.on_policy_ha_runner import OnPolicyHARunner
			assert OnPolicyHARunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OnPolicyHARunner: {e}")

	def test_off_policy_ha_runner_import(self):
		"""Test OffPolicyHARunner import."""
		try:
			from runners.off_policy_ha_runner import OffPolicyHARunner
			assert OffPolicyHARunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import OffPolicyHARunner: {e}")


@pytest.mark.unit
class TestRunnerRegistry:
	"""Test runner registry."""

	def test_runner_registry_import(self):
		"""Test RUNNER_REGISTRY import."""
		try:
			from runners import RUNNER_REGISTRY
			assert RUNNER_REGISTRY is not None
			assert isinstance(RUNNER_REGISTRY, dict)
		except ImportError as e:
			pytest.fail(f"Failed to import RUNNER_REGISTRY: {e}")

	def test_runner_registry_contains_algorithms(self):
		"""Test that runner registry contains expected algorithms."""
		from runners import RUNNER_REGISTRY

		# Check for on-policy algorithms
		assert "happo" in RUNNER_REGISTRY
		assert "mappo" in RUNNER_REGISTRY

		# Check that entries are callables
		for algo, runner in RUNNER_REGISTRY.items():
			assert callable(runner) or isinstance(runner, type)


@pytest.mark.unit
class TestTwoTimescaleRunner:
	"""Test two-timescale VVC runner."""

	def test_two_ts_runner_import(self):
		"""Test TwoTimescaleRunner import."""
		try:
			from runners.two_ts_runner import TwoTimescaleRunner
			assert TwoTimescaleRunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import TwoTimescaleRunner: {e}")


@pytest.mark.unit
class TestQmixRunner:
	"""Test QMIX runner."""

	def test_qmix_runner_import(self):
		"""Test QMixRunner import."""
		try:
			from runners.Qmix_runner import QMixRunner
			assert QMixRunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import QMixRunner: {e}")

	def test_qmix_base_runner_import(self):
		"""Test QMixBaseRunner import."""
		try:
			from runners.Qmix_base_runner import QMixBaseRunner
			assert QMixBaseRunner is not None
		except ImportError as e:
			pytest.fail(f"Failed to import QMixBaseRunner: {e}")
