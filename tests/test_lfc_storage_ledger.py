"""Synthetic tests for the offline storage energy ledger; no OpenDSS or MATLAB."""
import math
import unittest

from envs.lfc.diagnostics.storage_ledger import (
    StorageContract,
    TraceSample,
    audit_trace,
    fixed_pf_q_kvar,
)


class TestStorageLedger(unittest.TestCase):
    def test_one_second_small_pulse_is_not_rounded_away(self):
        samples = (TraceSample(1, -500, 50_000), TraceSample(1, -500, 50_000))
        first, second = audit_trace(StorageContract(100_000), 50_000, samples)
        self.assertAlmostEqual(first.model_delta_kwh, -500 / 3600)
        self.assertAlmostEqual(second.step_residual_kwh, 500 / 3600)
        self.assertAlmostEqual(second.cumulative_residual_kwh, 1000 / 3600)
        self.assertEqual(second.proxy_end_kwh, 50_000)

    def test_terminal_sign_and_efficiencies(self):
        initial = 100
        after_charge = initial + 100 / 3600 * 0.95
        after_discharge = after_charge - 75 * 2 / 3600 / 0.9
        samples = (
            TraceSample(1, 100, after_charge),
            TraceSample(2, -75, after_discharge),
            TraceSample(1, 0, after_discharge),
        )
        rows = audit_trace(StorageContract(200, 20, 0.95, 0.9), initial, samples)
        for row in rows:
            self.assertAlmostEqual(row.step_residual_kwh, 0)
            self.assertAlmostEqual(row.cumulative_residual_kwh, 0)
            self.assertTrue(row.expected_within_bounds and row.observed_within_bounds)
        self.assertGreater(rows[0].model_delta_kwh, 0)
        self.assertLess(rows[1].model_delta_kwh, 0)
        self.assertEqual(rows[2].model_delta_kwh, 0)

    def test_clipping_cannot_make_reference_pass(self):
        contract = StorageContract(100, 10)
        low, = audit_trace(contract, 10, [TraceSample(1, -100, 10)])
        high, = audit_trace(contract, 100, [TraceSample(1, 100, 100)])
        self.assertFalse(low.expected_within_bounds)
        self.assertTrue(low.observed_within_bounds)
        self.assertGreater(low.cumulative_residual_kwh, 0)
        self.assertFalse(high.expected_within_bounds)
        self.assertLess(high.cumulative_residual_kwh, 0)

    def test_fixed_pf_is_not_p_divided_by_pf(self):
        p, pf = 100, 0.95
        q = fixed_pf_q_kvar(p, pf)
        self.assertAlmostEqual(q, 32.86841051788632)
        self.assertAlmostEqual(p / math.hypot(p, q), pf)
        self.assertNotAlmostEqual(q, p / pf)
        self.assertAlmostEqual(fixed_pf_q_kvar(-p, pf, q_sign=-1), -q)
        self.assertEqual(fixed_pf_q_kvar(p, 1), 0)

    def test_explicit_input_validation(self):
        for invalid_seconds in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                TraceSample(invalid_seconds, 1, 5)
        with self.assertRaises(ValueError):
            TraceSample(1, float("inf"), 5)
        with self.assertRaises(ValueError):
            StorageContract(10, charge_efficiency=0)
        with self.assertRaises(ValueError):
            audit_trace(StorageContract(10, 1), 0, ())
        with self.assertRaises(ValueError):
            fixed_pf_q_kvar(1, 0.0)
