from dataclasses import replace
import json
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from envs.lfc.droop_123_draft import Arm, Frequency, load_protocol, validate_storage_hold
from envs.lfc.native_feeder_123 import PQ, StorageReading
from scripts.check_lfc_123_droop import (
    TRANSPORT_SCENARIO, _instrument, _native_arm, _native_frequency, _run_arms, _verify_prefixes,
)


class TestHarness(unittest.TestCase):
    def test_same_native_object_only_and_exact_post_prefix(self):
        eng = Mock()
        seen = []

        def column(_, expr):
            seen.append(expr)
            return {"out_f_load1.Time": [20, 20.2],
                    "out_f_load1.Data": [59.95, 59.96]}[expr]

        with patch("scripts.check_lfc_123_droop._column", side_effect=column):
            snapshot = _native_frequency(eng)
        self.assertEqual(seen, ["out_f_load1.Time", "out_f_load1.Data"])
        self.assertEqual(snapshot, (Frequency(20, 59.95), Frequency(20.2, 59.96)))
        _verify_prefixes([snapshot], snapshot + (Frequency(20.4, 60),))
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            _verify_prefixes([snapshot], (Frequency(20, 59.99), snapshot[-1]))
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            _verify_prefixes([snapshot], snapshot[:1])

    def test_source_provenance_and_only_format_changed(self):
        transport = json.loads(TRANSPORT_SCENARIO.read_bytes())
        eng = Mock()
        eng.get_param.side_effect = ["off", "Array", "0.2", "Timeseries"]
        _instrument(eng, "IEEE9BusSystem", transport["sensor"])
        eng.set_param.assert_called_once_with(
            "IEEE9BusSystem/To Workspace1", "SaveFormat", "Timeseries", nargout=0,
        )
        source_verification = eng.eval.call_args.args[0]
        self.assertIn("IEEE9BusSystem:3596", source_verification)
        self.assertIn("numel(pz_b)==1", source_verification)
        eng.reset_mock()
        eng.get_param.side_effect = ["on"]
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            _instrument(eng, "IEEE9BusSystem", transport["sensor"])
        eng.set_param.assert_not_called()

    def test_arm_order_and_freezing_only_no_control_reference(self):
        transport, spec = json.loads(TRANSPORT_SCENARIO.read_bytes()), load_protocol()
        events, order = [], []

        def backend(mode, ref, *_):
            order.append((mode, ref))
            p = PQ(1, 2)
            hz = tuple(Frequency(t, 60 if mode == "disturbed_no_control" else 59)
                       for t in (22, 23, 24))
            return Arm((), hz, p, p), ("one", "two", "three", "four")

        with patch("scripts.check_lfc_123_droop.assess", return_value={"status": "fake"}):
            _run_arms(transport, spec, events.append, backend)
        self.assertEqual(order, [("undisturbed_drift", None), ("disturbed_no_control", None),
                                 ("disturbed_fixed_droop", 60)])
        self.assertIn({"phase": "reference_frozen", "arm": "disturbed_no_control", "hz": 60},
                      events)
        backend = Mock(side_effect=RuntimeError("native Timeseries missing"))
        with self.assertRaisesRegex(RuntimeError, "Timeseries missing"):
            _run_arms(transport, spec, events.append, backend)
        backend.assert_called_once()

    def test_live_soc_energy_must_be_finite_continuous_and_physically_signed(self):
        spec = load_protocol()
        ids = tuple(f"Storage.batt{i}" for i in range(1, 5))
        req = (50, -50, 0, 0)
        after = (0.5 - 50 / (3600 * 200 * 0.95), 0.5 + 50 * 0.95 / (3600 * 200),
                 0.5, 0.5)
        good = tuple(StorageReading(d, u, u, PQ(u, 0), 0.5, a)
                     for d, u, a in zip(ids, req, after))
        ledger = validate_storage_hold(good, req, None, ids, spec)
        self.assertLess(ledger[0]["expected_delta_kwh"], 0)
        self.assertGreater(ledger[1]["expected_delta_kwh"], 0)
        wrong = (dict(soc_after=float("nan")), dict(soc_after=1.1),
                 dict(soc_before=.500001, soc_after=after[0] + .000001),
                 dict(soc_after=0.5))
        for values in wrong:
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, "SOC/energy"):
                validate_storage_hold((replace(good[0], **values), *good[1:]),
                                      req, None, ids, spec)

    def test_live_invalid_soc_stops_before_transmission_command_or_next_feeder_step(self):
        spec = load_protocol()
        cfg = json.loads(TRANSPORT_SCENARIO.read_bytes())
        ids = tuple(f"Storage.batt{i}" for i in range(1, 5))
        feeder, eng = Mock(), Mock()
        feeder.storage_ids = ids
        feeder.calibration.feeder_t0 = PQ(3600, 1000)
        feeder.scenario_sha256, feeder.session.model_sha256 = "scenario", "deck"
        feeder.step.return_value = SimpleNamespace(
            storage=tuple(StorageReading(d, 0, 0, PQ(0, 0), 0.5, float("nan")) for d in ids),
        )
        manager = Mock()
        manager.__enter__ = Mock(return_value=feeder)
        manager.__exit__ = Mock(return_value=False)
        eng.eval.return_value = 1 / 60
        eng.get_param.return_value = 20 + 2 / 60
        matlab, engine = ModuleType("matlab"), ModuleType("matlab.engine")
        matlab.engine = engine
        engine.start_matlab = Mock(return_value=eng)
        prefix = "scripts.check_lfc_123_droop."
        evidence = []
        with patch.dict("sys.modules", {"matlab": matlab, "matlab.engine": engine}), \
             patch(prefix + "Native123Feeder", return_value=manager), \
             patch(prefix + "_copy_model", return_value="IEEE9BusSystem"), \
             patch(prefix + "_instrument", return_value={}), \
             patch(prefix + "_poll"), \
             patch(prefix + "_native_frequency", return_value=(Frequency(20, 60),)), \
             patch(prefix + "_set_pq") as set_pq, patch(prefix + "_advance") as advance:
            with self.assertRaisesRegex(RuntimeError, "SOC/energy mismatch"):
                _native_arm("undisturbed_drift", None, cfg, spec, evidence.append)
        feeder.step.assert_called_once()
        set_pq.assert_called_once()  # fixed warm-up anchor only
        advance.assert_not_called()
        self.assertEqual(evidence[-1]["phase"], "pre_application_rejected")
        self.assertIn("nan", evidence[-1]["observed_native_row"])
