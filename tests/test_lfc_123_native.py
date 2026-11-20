"""Pure arithmetic and scenario checks; native gate is scripts.check_lfc_123_native."""
import math
import unittest

from envs.lfc.native_feeder_123 import GSPCalibration, PQ, _requests, stock_probe_spec


class TestNative123Contract(unittest.TestCase):
    def test_measured_feeder_delta_added_to_independently_supplied_anchor(self):
        model = GSPCalibration(PQ(100_000, 35_000), PQ(3_600, -20))
        self.assertEqual(model.background, PQ(96_400, 35_020))
        self.assertEqual(model.project(model.feeder_t0), model.target_t0)
        self.assertEqual(model.project(PQ(3_400, 5)), PQ(99_800, 35_025))

    def test_closed_requests_retain_order_and_validate_entire_frame(self):
        ids = ("Storage.a", "Storage.b")
        self.assertEqual(
            list(_requests({"Storage.b": -25, "Storage.a": 25}, ids)), list(ids)
        )
        for bad in (
            {"Storage.a": 0},
            {"Storage.a": 0, "extra": 0},
            {"Storage.a": False, "Storage.b": 0},
            {"Storage.a": math.nan, "Storage.b": 0},
        ):
            with self.assertRaises(ValueError):
                _requests(bad, ids)

    def test_scenario_maps_phase_and_interval_explicitly(self):
        spec, digest = stock_probe_spec()
        self.assertEqual(len(digest), 64)
        self.assertEqual(spec["feeder_directory"], "node_systems/123Bus")
        self.assertEqual(spec["feeder_master"], "network.dss")
        self.assertEqual(
            [(r["bus"], r["phases"]) for r in spec["storage_devices"]],
            [("33", [1]), ("114", [1]), ("67", [1, 2, 3]), ("300", [1, 2, 3])],
        )
        self.assertEqual([p["start_s"] for p in spec["input_pulses"]], [0, 1, 2, 3])
        self.assertEqual(len({r["id"] for r in spec["storage_devices"]}), 4)

    def test_nonfinite_pq_is_rejected(self):
        with self.assertRaises(ValueError):
            PQ(12, float("inf"))
