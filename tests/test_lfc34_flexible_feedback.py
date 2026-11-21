import json
import unittest

from scripts.check_lfc34_bus8_loop import MANIFEST, support


class FlexibleDemandPolicy(unittest.TestCase):
    def test_underfrequency_reduces_demand_and_ramps_back(self):
        cfg = json.loads(MANIFEST.read_text())["controller"]
        self.assertEqual(support(59.99, 60., 0., cfg), 5.)
        self.assertEqual(support(60., 60., 5., cfg), 0.)
        self.assertEqual(support(60.01, 60., 0., cfg), -5.)
        self.assertEqual(support(59.99, 60., 9., cfg), 10.)
