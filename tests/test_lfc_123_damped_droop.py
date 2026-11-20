from dataclasses import replace
import unittest

from envs.lfc.damped_droop_123 import MeasuredHeadroom, load_protocol, planned_events, step
from envs.lfc.droop_123_draft import Frequency


class TestDampedDroop(unittest.TestCase):
    cfg = load_protocol()
    ids = tuple(f"Storage.batt{i}" for i in range(1, 5))
    caps = tuple(MeasuredHeadroom(device, 0.5, 21.0333333, 50, 50)
                 for device in ids)

    def dispatch(self, hz, previous=0, caps=None):
        return step(60, lambda: (Frequency(21, hz),), 20.0333333, 21.0333333,
                    previous, self.caps if caps is None else caps, self.ids, self.cfg)

    def test_frozen_filter_deadband_and_release(self):
        self.assertEqual(sum(self.dispatch(59.99995).by_device_kw.values()), 0)
        self.assertAlmostEqual(sum(self.dispatch(59.9996).by_device_kw.values()), 12)
        self.assertEqual(sum(self.dispatch(59).by_device_kw.values()), 30)
        self.assertEqual(sum(self.dispatch(60.1, previous=100).by_device_kw.values()), 70)
        self.assertEqual(sum(self.dispatch(60.1).by_device_kw.values()), 0)

    def test_real_headroom_constrains_each_device_and_aggregate(self):
        caps = (
            replace(self.caps[0], soc=0.1),
            replace(self.caps[1], current_cap_kw=2),
            replace(self.caps[2], voltage_cap_kw=3),
            replace(self.caps[3], current_cap_kw=1),
        )
        self.assertEqual(list(self.dispatch(59, caps=caps).by_device_kw.values()), [0, 2, 3, 1])
        with self.assertRaisesRegex(RuntimeError, "headroom collapse"):
            self.dispatch(59, previous=40, caps=caps)
        for changes in ({"voltage_cap_kw": float("nan")}, {"soc": .01},
                        {"current_cap_kw": -1}, {"observed_boundary_s": 20.0333333}):
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                self.dispatch(59, caps=(replace(self.caps[0], **changes), *self.caps[1:]))

    def test_causal_source_is_mandatory(self):
        for reader in (None, lambda: (Frequency(21.2, 59),)):
            with self.assertRaises(RuntimeError):
                step(60, reader, 20.0333333, 21.0333333, 0, self.caps, self.ids, self.cfg)

    def test_design_and_held_out_preregistered(self):
        with self.assertRaises(RuntimeError):
            planned_events(self.cfg)
        design = planned_events(self.cfg, envelope_reviewed=True)
        blind = planned_events(self.cfg, design_passed=True, envelope_reviewed=True)
        self.assertEqual([(x["load_kw"], x["duration_s"]) for x in design], [(80, 8), (120, 8)])
        self.assertEqual([(x["load_kw"], x["duration_s"]) for x in blind], [(90, 12), (110, 12)])
        self.assertFalse({x["id"] for x in design} & {x["id"] for x in blind})
