import unittest

from scripts.summarize_lfc_bus8_calibration import frequency_metrics


class FrequencyMetrics(unittest.TestCase):
    def test_flat_trace_has_zero_integral_and_no_recovery_event(self):
        points = [(i/5, 60.) for i in range(41)]
        stats = frequency_metrics(points, 2.03, 5.03, 8.03)
        self.assertEqual(stats["iae_hz_s"], 0)
        self.assertIsNone(stats["recovery_from_release_s"])

    def test_uses_native_sample_spacing(self):
        points = [(i/5, 60.+(0.001 if i == 16 else 0)) for i in range(41)]
        stats = frequency_metrics(points, 2.03, 5.03, 8.03)
        self.assertAlmostEqual(stats["max_abs_rocof_hz_s"], .005)
        self.assertAlmostEqual(stats["iae_hz_s"], .0002)
        with self.assertRaises(ValueError):
            frequency_metrics(points[:16]+points[17:], 2.03, 5.03, 8.03)
