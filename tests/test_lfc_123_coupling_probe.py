"""Engine-free proof of 123 direct-engine gates (mocked sensors, no simulation)."""

import math
import unittest

from envs.lfc.native_feeder_123 import PQ
from scripts.check_lfc_123_coupling import (
    _check_measurement,
    _set_pq,
    _tail,
    _validate_pause,
)


class FakeEngine:
    def __init__(self):
        self.parameters = {}

    def set_param(self, block, name, value, *, nargout):
        self.parameters[block, name] = value

    def get_param(self, block, name):
        return self.parameters[block, name]


class TestDirectProbeGuards(unittest.TestCase):
    def test_only_dynamic_load_p_q_set_and_once_converted(self):
        engine = FakeEngine()
        _set_pq(engine, "IEEE9BusSystem", PQ(7851.739, -12.5))
        self.assertEqual(
            set(engine.parameters),
            {("IEEE9BusSystem/load1p", "Value"), ("IEEE9BusSystem/load1q", "Value")},
        )
        self.assertAlmostEqual(
            float(engine.get_param("IEEE9BusSystem/load1p", "Value")), 7851739
        )
        self.assertEqual(
            float(engine.get_param("IEEE9BusSystem/load1q", "Value")), -12500
        )

    def test_valid_clock_and_repeated_pause_guard(self):
        _validate_pause(21, 21 + 2 / 60, 20 + 2 / 60, 1, 2 / 60)
        for actual in (20 + 2 / 60, 21 - 1 / 60, 21 + 3 / 60, math.nan):
            with self.assertRaises(ValueError):
                _validate_pause(21, actual, 20 + 2 / 60, 1, 2 / 60)

    def test_stale_or_malformed_signal_fails(self):
        for values in (
            [],
            [(17, 1), (17.5, 1)],
            [(19, 1), (19, 1), (20, 1)],
            [(19, 1), (19.5, math.nan), (20, 1)],
        ):
            with self.assertRaises(RuntimeError):
                _tail(values, 20, 1, 0.3)

    def test_missing_q_or_bad_frequency_or_realized_q_cannot_pass(self):
        anchor = PQ(7000, 10000)
        t = [18 + j / 60 for j in range(121)]
        signals = lambda p, q, f: {
            "p_kw": [(x, p) for x in t],
            "q_kvar": [(x, q) for x in t],
            "frequency_hz": [(x, f(x)) for x in t],
        }
        policy = dict(
            p_abs_tol_kw=100,
            q_abs_tol_kvar=100,
            frequency_center_hz=60,
            frequency_abs_tol_hz=0.2,
            frequency_max_span_hz=0.03,
        )
        validate = lambda log: _check_measurement(
            anchor, log, policy, 20, 1, 0.04, settled=True
        )
        self.assertEqual(
            validate(signals(7001, 9999, lambda x: 60))["measured_q_kvar"], 9999
        )
        with self.assertRaises(RuntimeError):
            validate(signals(7001, 9000, lambda x: 60))
        with self.assertRaises(RuntimeError):
            validate(signals(7001, 9999, lambda x: 60 + 0.09 * math.sin(8 * x)))
        missing = signals(7001, 9999, lambda x: 60)
        missing["q_kvar"] = []
        with self.assertRaises(RuntimeError):
            validate(missing)
