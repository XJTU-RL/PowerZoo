"""Small falsifiers for matched credit, horizons and the nodal control."""
from contextlib import contextmanager
import unittest

import torch

from .clock_graph import GraphReadout
from .horizon_witness import repeated_witnesses
from .physical_credit import credit_budget


class PhysicalGateTests(unittest.TestCase):
    def test_discounted_credit_is_conserved_for_matched_batch(self):
        site = dict(A=dict(request_kw=10.,capacity_kw=6.,sustainable_kw=3.),
                    B=dict(request_kw=5.,capacity_kw=None,sustainable_kw=2.))
        factors = {("master",24):1.95}
        factors.update({(kind+s,k):v for kind in ("feeder","battery","pv") for s in ("A","B")
                        for k,v in ((25,1.),(26,.95))})
        old, neutral = credit_budget(site,factors,attributed=False)
        new, credit = credit_budget(site,factors,attributed=True)
        self.assertAlmostEqual(sum(old.values()),sum(new.values()))
        self.assertEqual(credit["B"],neutral["B"])
        self.assertEqual(credit["A"]["upper_kw"],4.)
        self.assertEqual(credit["A"]["lower_kw"],3.)
        self.assertNotEqual(old[("master",24)],new[("master",24)])

    def test_finite_witness_rejects_one_step_high_offer_for_long_horizon(self):
        class Plant:
            count = 0
            def probe(self,idx):
                return dict(veto=["energy"] if idx==1 and self.count>=2 else [])
            def execute(self,idx):
                self.count+=1
                return dict(delivered_kw=8. if idx==1 else 3.)
        @contextmanager
        def restore():
            yield Plant()
        result,paths = repeated_witnesses(restore,2)
        self.assertEqual([result[n]["capacity_kw"] for n in (1,2,5)],[8.,8.,3.])
        self.assertEqual(paths[1],[8.,8.])

    def test_nodal_control_parameter_parity_and_no_cross_node_messages(self):
        torch.manual_seed(1701)
        graph = GraphReadout(3,"admittance")
        self_only = GraphReadout(3,"self")
        self_only.load_state_dict(graph.state_dict())
        self.assertEqual([p.shape for p in graph.parameters()],[p.shape for p in self_only.parameters()])
        x = torch.randn((3,18))
        changed = x.clone()
        changed[1,:] += 1.
        adj = torch.eye(3).roll(1,1).repeat(3,1,1)
        self.assertTrue(torch.equal(self_only.encode_nodes(x,adj)[0],self_only.encode_nodes(changed,adj)[0]))
        self.assertFalse(torch.equal(graph.encode_nodes(x,adj)[0],graph.encode_nodes(changed,adj)[0]))
        grad, = torch.autograd.grad(self_only.encode_nodes(x,adj).sum(),self_only.message.weight)
        self.assertGreater(float(grad[:,16:].norm()),0.)
