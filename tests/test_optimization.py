# -*- coding: utf-8 -*-
"""
Test suite for QMIX and SHOM optimization verification.

This module tests:
1. QMIX GPU optimization - mixer output stays on GPU
2. SHOM decoupled architecture - sensitivity calculation and ordering strategies

Run with: python -m pytest tests/test_optimization.py -v
Or simply: python tests/test_optimization.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

# Try to import torch, set flag if not available
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. GPU tests will be skipped.")


class TestSensitivityModule(unittest.TestCase):
    """Test the decoupled sensitivity calculation module."""

    def test_power_grid_sensitivity_compute(self):
        """Test PowerGridSensitivity.compute() method."""
        from utils.sensitivity import PowerGridSensitivity

        # Setup test data
        agents_bus_mapping = {
            "Regulator.reg1": ["650.1", "650.2"],
            "Capacitor.cap1": ["675.1", "675.3"],
            "Battery.bat1": ["611.3"]
        }

        calculator = PowerGridSensitivity(agents_bus_mapping)

        # Test with sample env_info
        env_info = {
            'S': {
                "650.1": 10.5,
                "650.2": 5.5,
                "675.1": 3.0,
                "675.3": 2.0,
                "611.3": 1.0
            }
        }

        result = calculator.compute(env_info)

        # Verify results
        self.assertAlmostEqual(result["Regulator.reg1"], 16.0)  # 10.5 + 5.5
        self.assertAlmostEqual(result["Capacitor.cap1"], 5.0)   # 3.0 + 2.0
        self.assertAlmostEqual(result["Battery.bat1"], 1.0)     # 1.0

        print("  [PASS] PowerGridSensitivity.compute()")

    def test_power_grid_sensitivity_aggregate(self):
        """Test PowerGridSensitivity.aggregate() method."""
        from utils.sensitivity import PowerGridSensitivity

        agents_bus_mapping = {
            "Regulator.reg1": ["650.1"],
            "Capacitor.cap1": ["675.1"]
        }

        calculator = PowerGridSensitivity(agents_bus_mapping)

        # Simulate multiple timesteps of sensitivity data
        sensitivity_history = [
            {"650.1": 2.0, "675.1": 1.0},
            {"650.1": 3.0, "675.1": 1.5},
            {"650.1": 1.0, "675.1": 0.5}
        ]

        result = calculator.aggregate(sensitivity_history)

        # Verify aggregation (sum over time)
        self.assertAlmostEqual(result["Regulator.reg1"], 6.0)  # 2.0 + 3.0 + 1.0
        self.assertAlmostEqual(result["Capacitor.cap1"], 3.0)  # 1.0 + 1.5 + 0.5

        print("  [PASS] PowerGridSensitivity.aggregate()")

    def test_uniform_sensitivity(self):
        """Test UniformSensitivity returns empty dict."""
        from utils.sensitivity import UniformSensitivity

        calculator = UniformSensitivity(num_agents=5)

        result = calculator.compute({})
        self.assertEqual(result, {})

        result = calculator.aggregate([])
        self.assertEqual(result, {})

        print("  [PASS] UniformSensitivity")

    def test_create_sensitivity_calculator_factory(self):
        """Test the factory function for creating sensitivity calculators."""
        from utils.sensitivity import (
            create_sensitivity_calculator,
            PowerGridSensitivity,
            UniformSensitivity
        )

        # Test: useS=False should return UniformSensitivity
        calc = create_sensitivity_calculator(
            env_name="powerzoo",
            env_args={"useS": False},
            num_agents=3
        )
        self.assertIsInstance(calc, UniformSensitivity)

        # Test: useS=True with proper mapping should return PowerGridSensitivity
        calc = create_sensitivity_calculator(
            env_name="powerzoo",
            env_args={"useS": True},
            agents_bus_mapping={"agent1": ["bus1"]},
            num_agents=3
        )
        self.assertIsInstance(calc, PowerGridSensitivity)

        print("  [PASS] create_sensitivity_calculator factory")


class TestAgentOrderingModule(unittest.TestCase):
    """Test the decoupled agent ordering module."""

    def test_sensitivity_order_descending(self):
        """Test SensitivityOrder with descending order (big to small)."""
        from utils.sensitivity import PowerGridSensitivity
        from utils.agent_ordering import SensitivityOrder

        agents_bus_mapping = {
            "Agent.a": ["bus1"],
            "Agent.b": ["bus2"],
            "Agent.c": ["bus3"]
        }
        agent_id_mapping = {
            "Agent.a": 0,
            "Agent.b": 1,
            "Agent.c": 2
        }

        sensitivity_calc = PowerGridSensitivity(agents_bus_mapping)
        strategy = SensitivityOrder(
            sensitivity_calculator=sensitivity_calc,
            agent_id_mapping=agent_id_mapping,
            descending=True  # big to small
        )

        sensitivity_values = {
            "Agent.a": 5.0,
            "Agent.b": 10.0,
            "Agent.c": 1.0
        }

        order = strategy.get_order(3, sensitivity_values=sensitivity_values)

        # Expected order: b(10) -> a(5) -> c(1) => [1, 0, 2]
        self.assertEqual(order, [1, 0, 2])
        self.assertEqual(strategy.name, "sensitivity_big2small")

        print("  [PASS] SensitivityOrder (descending)")

    def test_sensitivity_order_ascending(self):
        """Test SensitivityOrder with ascending order (small to big)."""
        from utils.sensitivity import PowerGridSensitivity
        from utils.agent_ordering import SensitivityOrder

        agents_bus_mapping = {
            "Agent.a": ["bus1"],
            "Agent.b": ["bus2"],
            "Agent.c": ["bus3"]
        }
        agent_id_mapping = {
            "Agent.a": 0,
            "Agent.b": 1,
            "Agent.c": 2
        }

        sensitivity_calc = PowerGridSensitivity(agents_bus_mapping)
        strategy = SensitivityOrder(
            sensitivity_calculator=sensitivity_calc,
            agent_id_mapping=agent_id_mapping,
            descending=False  # small to big
        )

        sensitivity_values = {
            "Agent.a": 5.0,
            "Agent.b": 10.0,
            "Agent.c": 1.0
        }

        order = strategy.get_order(3, sensitivity_values=sensitivity_values)

        # Expected order: c(1) -> a(5) -> b(10) => [2, 0, 1]
        self.assertEqual(order, [2, 0, 1])
        self.assertEqual(strategy.name, "sensitivity_small2big")

        print("  [PASS] SensitivityOrder (ascending)")

    def test_fixed_order(self):
        """Test FixedOrder strategy."""
        from utils.agent_ordering import FixedOrder

        strategy = FixedOrder()
        order = strategy.get_order(5)

        self.assertEqual(order, [0, 1, 2, 3, 4])
        self.assertEqual(strategy.name, "fixed")

        # Test with custom order
        strategy = FixedOrder(custom_order=[2, 0, 1])
        order = strategy.get_order(3)

        self.assertEqual(order, [2, 0, 1])

        print("  [PASS] FixedOrder")

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_random_order(self):
        """Test RandomOrder strategy."""
        from utils.agent_ordering import RandomOrder

        strategy = RandomOrder(seed=42)
        order = strategy.get_order(5)

        # Verify it's a permutation
        self.assertEqual(sorted(order), [0, 1, 2, 3, 4])
        self.assertEqual(len(order), 5)
        self.assertEqual(strategy.name, "random")

        print("  [PASS] RandomOrder")

    def test_agent_order_manager(self):
        """Test AgentOrderManager integration."""
        from utils.sensitivity import PowerGridSensitivity
        from utils.agent_ordering import AgentOrderManager, SensitivityOrder

        # Bus IDs must contain '.' to match real power grid format
        agents_bus_mapping = {
            "Agent.a": ["650.1"],
            "Agent.b": ["675.1"]
        }
        agent_id_mapping = {
            "Agent.a": 0,
            "Agent.b": 1
        }

        sensitivity_calc = PowerGridSensitivity(agents_bus_mapping)
        strategy = SensitivityOrder(
            sensitivity_calculator=sensitivity_calc,
            agent_id_mapping=agent_id_mapping,
            descending=True
        )

        manager = AgentOrderManager(
            strategy=strategy,
            sensitivity_calculator=sensitivity_calc
        )

        # Simulate buffer_infos structure matching real format
        # Bus IDs contain '.' (e.g., "650.1" means bus 650, phase 1)
        buffer_infos = {
            0: [{"650.1": 3.0, "675.1": 8.0}],  # timestep 0
            1: [{"650.1": 4.0, "675.1": 2.0}]   # timestep 1
        }
        # After aggregation:
        # 650.1 total: 3.0 + 4.0 = 7.0 -> Agent.a
        # 675.1 total: 8.0 + 2.0 = 10.0 -> Agent.b

        order = manager.compute_order(
            num_agents=2,
            buffer_infos=buffer_infos,
            verbose=False
        )

        # Agent.b has higher total sensitivity (10.0) than Agent.a (7.0)
        # So descending order should be [1, 0]
        self.assertEqual(order, [1, 0])

        print("  [PASS] AgentOrderManager")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestQMIXOptimization(unittest.TestCase):
    """Test QMIX GPU optimization."""

    def test_mixer_output_device(self):
        """Test that M_QMixer output stays on the same device as input."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            print("  [SKIP] CUDA not available, skipping GPU test")
            return

        from models.value_function_models.mq_mixer import M_QMixer

        # Create mock args
        class Args:
            use_orthogonal = True
            mixer_hidden_dim = 32
            hypernet_hidden_dim = 64
            hypernet_layers = 2

        args = Args()
        device = torch.device("cuda:0")

        mixer = M_QMixer(
            args=args,
            num_agents=3,
            cent_obs_dim=10,
            device=device
        )

        # Create test inputs on GPU
        batch_size = 4
        agent_q_inps = torch.randn(batch_size, 3).to(device)
        states = torch.randn(batch_size, 10).to(device)

        # Forward pass
        q_tot = mixer(agent_q_inps, states)

        # Verify output is on GPU (not moved to CPU)
        self.assertEqual(q_tot.device.type, "cuda")
        print(f"  [PASS] M_QMixer output device: {q_tot.device}")

    def test_mixer_output_cpu(self):
        """Test M_QMixer on CPU still works."""
        from models.value_function_models.mq_mixer import M_QMixer

        class Args:
            use_orthogonal = True
            mixer_hidden_dim = 32
            hypernet_hidden_dim = 64
            hypernet_layers = 1

        args = Args()
        device = torch.device("cpu")

        mixer = M_QMixer(
            args=args,
            num_agents=4,
            cent_obs_dim=20,
            device=device
        )

        batch_size = 8
        agent_q_inps = torch.randn(batch_size, 4)
        states = torch.randn(batch_size, 20)

        q_tot = mixer(agent_q_inps, states)

        # Expected shape: (batch_size, 1, 1)
        self.assertEqual(q_tot.shape, (batch_size, 1, 1))
        self.assertEqual(q_tot.device.type, "cpu")
        print("  [PASS] M_QMixer CPU output shape and device")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Running Optimization Tests for QMIX and SHOM")
    print("=" * 60)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSensitivityModule))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentOrderingModule))
    if TORCH_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestQMIXOptimization))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print(f"  Success: {result.wasSuccessful()}")
    print("=" * 60)

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
