import unittest
from .hierarchy import attribution


class CreditBalance(unittest.TestCase):
    def test_sustainable_shortfall(self):
        r=attribution(100,85,70)
        self.assertEqual((r["upper_kw"],r["lower_kw"]),(15,15))

    def test_excess_delivery_offsets_upper_credit(self):
        r=attribution(100,85,92)
        self.assertEqual((r["upper_raw_kw"],r["over_credit_kw"],r["upper_kw"],r["gap_kw"]),(15,7,8,8))
        self.assertEqual(attribution(100,85,105)["excess_over_request_kw"],5)

    def test_reverse_measured_delivery_counts_against_upward_request(self):
        r=attribution(100,85,-4)
        self.assertEqual((r["reverse_kw"],r["lower_kw"],r["gap_kw"]),(4,89,104))

    def test_negative_request_is_outside_case_a(self):
        with self.assertRaises(ValueError):
            attribution(-2,3,1)
