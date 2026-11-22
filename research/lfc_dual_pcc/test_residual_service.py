"""Residual policy warm start and physical opportunity threshold."""
import unittest

import torch

from .continuous_service import allocation_interval
from .precheck_residual import distinct_signatures
from .residual_service import ResidualServicePolicy, residual_interval


class ResidualServiceTests(unittest.TestCase):
    def test_rule_initialization_and_zero_width_with_saturated_center(self):
        total,lo,hi=residual_interval(2.5,dict(A=2.,B=2.))
        self.assertEqual((total,lo,hi),(2.5,.625,1.875))
        master=ResidualServicePolicy().heads["master"]
        x=torch.randn(37)
        learned,sample=master.decide(x,lo,hi,total,True,False)
        rule,_=master.decide(x,lo,hi,total,True,True)
        self.assertEqual(learned,rule)
        self.assertEqual(float(sample["latent"]),.5)
        total,lo,hi=residual_interval(2.5,dict(A=.1,B=4.))
        self.assertEqual((lo,hi),(.1,.1))
        self.assertIsNone(master.decide(x,lo,hi,total,False,False)[1]["latent"])

    def test_b_capacity_sets_the_a_lower_bound_before_residual_bounds(self):
        limits = dict(A=90., B=30.)
        self.assertEqual(allocation_interval(100., limits), (100., 70., 90.))
        total, lo, hi = residual_interval(100., limits)
        self.assertEqual((total, lo, hi), (100., 70., 70.))
        self.assertEqual(total-lo, limits["B"])

    def test_pcc_separation_is_checked_after_the_devices(self):
        self.assertEqual(distinct_signatures([[100.,100.],[100.01,100.],[100.,102.]],.05),2)
