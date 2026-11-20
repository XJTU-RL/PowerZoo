import json
import unittest
from unittest.mock import Mock, patch

from envs.lfc.droop_123_draft import Arm, Frequency, load_protocol
from envs.lfc.native_feeder_123 import PQ
from scripts.check_lfc_123_droop import (
    TRANSPORT_SCENARIO, _instrument, _native_frequency, _run_arms, _verify_prefixes,
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
