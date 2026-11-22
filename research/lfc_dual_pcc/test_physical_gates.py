"""Small falsifiers for matched credit, horizons and the nodal control."""
from contextlib import contextmanager
import unittest

import torch

from .clock_graph import GraphReadout
from .horizon_witness import repeated_witnesses
from .physical_credit import credit_budget, nested_bound
from .run_horizon_pair import HorizonController


class PhysicalGateTests(unittest.TestCase):
    def test_nested_domains_and_missing_observed_path(self):
        def site(req,actual,best):
            return dict(request_kw=req,sustainable_kw=actual,actions=[0,0],certificate={
                "action_paths":[dict(actions=[0,0],support_kw=actual,zero_request_allowed=True),
                                dict(actions=[2,2],support_kw=best,zero_request_allowed=False)]})
        sites = dict(A=site(8.,3.,4.),B=site(2.,1.,6.))
        caps = dict(A=8.,B=6.)
        result = nested_bound(sites,caps,.005)
        self.assertEqual([result["relaxed"]["loss_kw"],result["fixed"]["loss_kw"],result["observed_kw"]],[0.,4.,6.])
        self.assertEqual(result["potentially_dispatch_avoidable_kw"],4.)
        self.assertEqual(result["device_avoidable_kw"],2.)
        sites["A"]["actions"] = [5,5]
        with self.assertRaisesRegex(ValueError,"outside the fixed evidence"):
            nested_bound(sites,caps,.005)

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

    def test_horizon_execution_rechecks_veto_and_clears_the_service_window(self):
        class Plant:
            base=(100.,0.)
            def candidates(self):
                return [dict(veto=[],pcc=[100.,0.]),dict(veto=["energy"])]
        events=[]
        prefix={24:dict(request=dict(A=0.,B=0.),actions=dict(A=0,B=0))}
        plan=dict(request_kw=dict(A=8.,B=2.),action=dict(A=1,B=0))
        ctrl=HorizonController(dict(actions_battery_pv_kw=[[0,0],[8,0]]),prefix,plan,25)
        plants=dict(A=Plant(),B=Plant())
        ctrl.initialize(plants,events.append)
        self.assertEqual(ctrl.dispatch(24,24.,23.9,0.,plants),(prefix[24]["request"],prefix[24]["actions"]))
        self.assertEqual(ctrl.dispatch(25,25.,24.9,0.,plants),(plan["request_kw"],dict(A=0,B=0)))
        self.assertEqual(events[-1]["fallback"],dict(A=0))
        self.assertEqual(ctrl.dispatch(30,30.,29.9,0.,plants),(dict(A=0.,B=0.),dict(A=0,B=0)))
        self.assertEqual(events[-1]["external_kw"],0.)

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
        topology = GraphReadout(3,"topology")
        topology.load_state_dict(graph.state_dict())
        self.assertTrue(torch.equal(topology.encode_nodes(x,adj),topology.encode_nodes(x,-5.*adj)))
        self.assertFalse(torch.equal(topology.encode_nodes(x,adj)[0],topology.encode_nodes(changed,adj)[0]))
