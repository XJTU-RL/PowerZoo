import unittest
from types import SimpleNamespace

from envs.lfc.receiver_123 import Frequency
from scripts.check_lfc_123_transfer import deltas, first_run


class CalibrationTests(unittest.TestCase):
    def test_single_noise_spike_does_not_start_response(self):
        points = [(1.2, -2), (1.4, 0), (1.6, -3), (1.8, -4)]
        self.assertEqual(first_run(points, lambda value: value < -1, 2), 1.6)
        self.assertIsNone(first_run(points, lambda value: value > 1, 2))

    def test_offset_or_short_native_trace_fails(self):
        base = SimpleNamespace(hz=[Frequency(2, 60)])
        for samples in ([], [Frequency(2.02, 60)]):
            with self.assertRaises(ValueError):
                deltas(SimpleNamespace(hz=samples), base)
