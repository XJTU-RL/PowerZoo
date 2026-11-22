from dataclasses import replace
import unittest

from .online_handoff import SustainedOffer, limit_handoff


class HandoffTest(unittest.TestCase):
    def test_partial_and_late_witness_are_excluded(self):
        offer = SustainedOffer(10., 10.5, 11., 16., 16., 9, "prefix", 4.)
        self.assertEqual(offer.capacity_at(11., 16., 9, "prefix"), 4.)
        self.assertIsNone(offer.capacity_at(11., 16.01, 9, "prefix"))
        self.assertIsNone(offer.capacity_at(11., 16., 9, "new prefix"))
        self.assertIsNone(offer.capacity_at(11., 16., 10, "prefix"))
        self.assertIsNone(replace(offer, arrival=11.).capacity_at(11., 16., 9, "prefix"))
        self.assertIsNone(offer.capacity_at(16., 17., 9, "prefix"))

    def test_release_uses_physical_choices_and_reports_unavoidable_excess(self):
        self.assertEqual(limit_handoff({2: 0., 3: 1.1}, 0., 0.), (2, 0.))
        self.assertEqual(limit_handoff({2: .6, 3: 1.1}, 0., 0.), (2, .6))
        self.assertEqual(limit_handoff({0: .4, 1: .9, 2: 1.2}, .9, 1.), (1, 0.))
