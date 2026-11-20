"""Engine-free proof of 123 direct-engine gates (mocked sensors, no simulation)."""

import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from envs.lfc.native_feeder_123 import PQ
from scripts.check_lfc_123_coupling import (
    _check_measurement,
    _check_pulse_order,
    _check_storage_delivery,
    _copy_model,
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
            [(19, 1), (18, 1), (20, 1)],
            [(19, 1), (19.5, math.nan), (20, 1)],
        ):
            with self.assertRaises(RuntimeError):
                _tail(values, 20, 1, 0.3)

    def test_missing_q_or_bad_frequency_or_realized_q_cannot_pass(self):
        anchor = PQ(7000, 10000)
        t = [18 + j / 60 for j in range(121)]
        tm5 = [18 + j / 5 for j in range(11)]
        signals = lambda p, q, f: {
            "p_kw": [(x, p) for x in t],
            "q_kvar": [(x, q) for x in t],
            "frequency_hz": [(x, f(x)) for x in tm5],
        }
        policy = dict(
            p_abs_tol_kw=100,
            q_abs_tol_kvar=100,
            frequency_center_hz=60,
            frequency_abs_tol_hz=0.2,
            frequency_max_span_hz=0.03,
        )
        validate = lambda log: _check_measurement(
            anchor,
            log,
            policy,
            20,
            1,
            dict(power=0.04, frequency=0.2000001),
            settled=True,
        )
        self.assertEqual(
            validate(signals(7001, 9999, lambda x: 60))["measured_q_kvar"], 9999
        )
        with self.assertRaises(RuntimeError):
            validate(signals(7001, 9000, lambda x: 60))
        with self.assertRaises(RuntimeError):
            validate(signals(7001, 9999, lambda x: 60 + 0.09 * math.sin(8 * x)))
        lost_frequency = signals(7001, 9999, lambda x: 60)
        del lost_frequency["frequency_hz"][-2]
        with self.assertRaises(RuntimeError):
            validate(lost_frequency)
        missing = signals(7001, 9999, lambda x: 60)
        missing["q_kvar"] = []
        with self.assertRaises(RuntimeError):
            validate(missing)

    def test_storage_requires_measured_terminal_p_and_q(self):
        request = {"one": 50, "two": -50, "three": 0}
        policy = dict(storage_abs_tol_kw=0.05, storage_abs_tol_kvar=0.05)

        def rows(positive=50, negative=-50, q=0):
            return [
                SimpleNamespace(device=device, measured_injection=PQ(p, q))
                for device, p in zip(request, (positive, negative, 0), strict=True)
            ]

        _check_storage_delivery(request, rows(), policy)
        for wrong in (rows(0), rows(negative=50), rows(q=1), rows()[1:],
                      rows() + rows()[:1]):
            with self.assertRaises(RuntimeError):
                _check_storage_delivery(request, wrong, policy)

    def test_pulse_order_requires_two_distinct_signed_responses(self):
        _check_pulse_order([3615, 3409, 3824, 3615], 150, "PCC")
        _check_pulse_order([7875, 7690, 8085, 7890], 100, "Bus8")
        for wrong in ([3615] * 4, [3615, 3824, 3409, 3615],
                      [3615, 3409, 3650, 3615], [3615, 3409, 3824, 3450]):
            with self.assertRaises(RuntimeError):
                _check_pulse_order(wrong, 150, "PCC")

    def test_assets_are_allowlisted_and_all_verified_before_staging(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            root, staged = Path(src), Path(dst)
            (root / "grid.slx").write_bytes(b"model")
            (root / "helper.m").write_bytes(b"helper")
            (root / "unlisted.m").write_bytes(b"not part of probe")
            cfg = dict(
                model="grid.slx",
                model_sha256=hashlib.sha256(b"model").hexdigest(),
                model_helpers={"helper.m": hashlib.sha256(b"helper").hexdigest()},
            )
            with patch("scripts.check_lfc_123_coupling.ROOT", root):
                (root / "helper.m").unlink()
                with self.assertRaisesRegex(RuntimeError, "missing pinned MATLAB asset"):
                    _copy_model(cfg, staged)
                self.assertEqual(list(staged.iterdir()), [])
                (root / "helper.m").write_bytes(b"unexpected")
                with self.assertRaisesRegex(RuntimeError, "unexpected pinned MATLAB asset"):
                    _copy_model(cfg, staged)
                self.assertEqual(list(staged.iterdir()), [])
                (root / "helper.m").write_bytes(b"helper")
                self.assertEqual(_copy_model(cfg, staged), "grid")
                self.assertEqual({p.name for p in staged.iterdir()}, {"grid.slx", "helper.m"})
