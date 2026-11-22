from copy import deepcopy
import unittest

import torch

from .paired_service import fit_bias, paired_window, post_service_swing
from .service_tradeoff import TradeoffPolicy


def controlled_window():
    return dict(slot=24,allocatable_kw=2.5,interval_kw=[.625,1.875],reports={"A":22,"B":22},
                frequency_error_hz=.0002,observation=[0.]*37,alternatives=[
                    dict(latent=u,request_kw={"A":a,"B":2.5-a},delivered_kw=[{"A":x,"B":y}]*2)
                    for u,a,x,y in ((.5,1.25,2.135,2.135),(.125,.78125,0.,2.135),(.875,1.71875,2.135,0.))])


class PairedServiceTest(unittest.TestCase):
    def test_pair_covers_ties_and_uses_common_physical_clock(self):
        stamps={25:dict(slot=27,used_slot=25,write_s=27.,end_s=28.),
                26:dict(slot=28,used_slot=26,write_s=28.,end_s=29.5)}
        result=paired_window(controlled_window(),stamps,.0002)
        self.assertEqual([x["weight_s"] for x in result["clock"]],[1.,.95*1.5])
        self.assertEqual(len(result["pairs"]),3)
        self.assertAlmostEqual(result["pairs"][0]["service_advantage_a_over_b_kw"],-1.355)
        self.assertFalse(result["pairs"][-1]["resolved"])
        self.assertEqual(result["pairs"][-1]["weight"],0.)
        with self.assertRaises(ValueError):
            paired_window(controlled_window(),{25:stamps[25]},.0002)

    def test_one_update_targets_both_preferred_alternatives_and_only_bias(self):
        stamps={25:dict(slot=27,used_slot=25,write_s=27.,end_s=28.),
                26:dict(slot=28,used_slot=26,write_s=28.,end_s=29.)}
        data=[paired_window(controlled_window(),stamps,.0002)]
        model=TradeoffPolicy()
        prior=deepcopy(model.state_dict())
        change=fit_bias(model,data)
        self.assertEqual(change["optimizer_steps"],1)
        self.assertEqual(change["resolved"],2)
        self.assertAlmostEqual(change["bias_after"][1],change["bias_after"][2])
        self.assertLess(change["bias_after"][0],change["bias_after"][1])
        self.assertEqual([name for name,t in model.state_dict().items() if not torch.equal(prior[name],t)],
                         ["heads.master.actor.layers.2.bias"])

    def test_post_service_swing_starts_on_the_physical_consumer(self):
        frames=[dict(used_slot=37,slot=39,write_s=39.)]
        values=[(39.,59.5),(40.,60.1),(41.,60.),(43.,59.9)]
        stats=post_service_swing(frames,values)
        self.assertEqual(stats["post_service_sample_count"],3)
        self.assertAlmostEqual(stats["post_service_swing_hz"],.2)
        with self.assertRaises(ValueError):
            post_service_swing(frames,values[:-1])
