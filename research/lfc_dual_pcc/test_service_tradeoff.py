import unittest

import torch

from .service_tradeoff import ProbeMaster, allowed_splits, compare_risk, service_terms


class ServiceTradeoffTest(unittest.TestCase):
    def test_deadband_and_numerical_tradeoff(self):
        self.assertAlmostEqual(service_terms(2.5, 2.135)[0], .365)
        self.assertAlmostEqual(service_terms(2.5, 4.270)[1], 1.720)
        self.assertEqual(service_terms(2.5, 2.525), (0.,0.))
        self.assertAlmostEqual((2.5-2.135)/(4.270-2.5), .2062146893)

    def test_unwitnessed_geometry_packet_loss_and_urgency_default_together(self):
        reports = dict(A=23,B=23)
        self.assertEqual(allowed_splits(2.5,.625,1.875,reports,.0002,.0002), [True]*3)
        self.assertEqual(allowed_splits(2.5,.625,1.875,dict(A=None,B=23),0.,.0002), [True,False,False])
        self.assertEqual(allowed_splits(2.5,.7,1.8,reports,0.,.0002), [True,False,False])
        self.assertEqual(allowed_splits(2.5,.625,1.875,reports,.0005,.0002), [True,False,False])

    def test_policy_probability_matches_behavior_and_forced_has_no_actor_sample(self):
        master = ProbeMaster()
        x = torch.zeros(37)
        action, record = master.decision((x,), [True,True,True], True)
        self.assertEqual(action,0)
        self.assertTrue(torch.equal(master.evaluate_sample(record)[0],record["old_logp"]))
        _, forced = master.decision((x,), [True,False,False], False)
        self.assertIsNone(master.evaluate_sample(forced)[0])

    def test_accumulated_shortfall_fails_even_when_every_step_fits(self):
        sample = dict(iae_hz_s=.0017,nadir_hz=59.96,peak_unmet_external_kw=0.,
                      unmet_external_kwh=0.,excess_external_kwh=.005)
        low = dict(sample,peak_unmet_external_kw=.365,unmet_external_kwh=12*.365/3600,
                   excess_external_kwh=0.)
        self.assertTrue(compare_risk(dict(metrics=sample),dict(metrics=low))["passed"])
        low["unmet_external_kwh"] = 13*.365/3600
        outcome = compare_risk(dict(metrics=sample),dict(metrics=low))
        self.assertFalse(outcome["checks"]["cumulative_shortfall"])
        self.assertTrue(outcome["checks"]["peak_shortfall"])
