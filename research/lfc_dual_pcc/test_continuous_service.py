"""Bounded-action and transport-semantics regressions for the slow master."""
from dataclasses import replace
import unittest

import torch

from .clock_graph import Message, update
from .continuous_service import (ContinuousServiceCoordinator, ContinuousServicePolicy,
                                 allocation_interval, freshness)
from .master_dispatch import site_reports
from .test_clock_graph import packet


class ContinuousServiceTests(unittest.TestCase):
    def test_beta_behavior_and_point_interval(self):
        self.assertEqual(allocation_interval(10.,dict(A=11.,B=4.)),(10.,6.,10.))
        self.assertEqual(allocation_interval(10.,dict(A=5.,B=2.)),(7.,5.,5.))
        model = ContinuousServicePolicy()
        head = model.heads["master"]
        x = torch.zeros(37)
        _, collapse = head.decide(x,5.,5.,7.,False,False)
        stats = update(model,{"master":[(26,2.,-5.,collapse)]},.003)
        self.assertIsNone(stats["steps"][0]["samples"][0]["ratio"])
        self.assertTrue(all(d==0. for k,d in stats["deltas"].items() if ".actor." in k))
        a, behavior = head.decide(x,6.,10.,10.,False,False)
        self.assertAlmostEqual(a,6.+4.*float(behavior["latent"]))
        self.assertEqual(float(head.evaluate_sample(behavior)[0].detach()),float(behavior["old_logp"]))
        bias = head.actor.layers[2].bias
        with torch.no_grad():
            bias.copy_(torch.tensor([1.,-1.]))
        self.assertGreater(float(head.distribution(x).mean.detach()),.5)
        lp = head.distribution(x).log_prob(torch.tensor(.4))
        self.assertTrue(torch.all(torch.autograd.grad(lp,bias)[0].abs()>0))
        a, behavior = head.decide(x,6.,10.,10.,False,False)
        stats = update(model,{"master":[(24,2.,-5.,behavior)]},.003)
        self.assertEqual(stats["steps"][0]["samples"][0]["ratio"],1.)
        self.assertTrue(any(d>0. for k,d in stats["deltas"].items() if k.startswith("heads.master.actor.")))
        self.assertEqual(head.decide(x,6.,10.,10.,True,True)[0],6.)

    def test_one_step_freshness_from_arrived_packets(self):
        original = packet(20,21.,21.1,6.)
        msg = replace(original,data={s:dict(d,witness=2) for s,d in original.data.items()})
        self.assertIsNone(site_reports([msg],21.1)["A"])
        self.assertEqual(freshness(msg,"A",21.1,1.),0.)
        self.assertAlmostEqual(freshness(site_reports([msg],22.)["A"],"A",22.,1.),2/3)
        self.assertEqual(freshness(msg,"A",22.,2.),0.)
        self.assertEqual(freshness(msg,"A",24.,1.),0.)

    def test_issued_urgency_reaches_the_device_after_transport(self):
        cfg = dict(window_slots=[22,24], frequency_scale_hz=.01, fast_s=1.,
                   actions_battery_pv_kw=[[0,0],[0,2],[4,0],[8,0],[4,2],[8,2]])
        control = ContinuousServiceCoordinator(ContinuousServicePolicy(),cfg,True)
        class Plant:
            base = (100.,0.)
            def candidates(self):
                return [dict(veto=[],pcc=[100.-sum(pair),0.]) for pair in cfg["actions_battery_pv_kw"]]
        class Graph:
            adj = torch.zeros((3,2,2))
            pools = ([0],[1])
            def features(self,plant,request):
                return torch.zeros((2,18))
        events = []
        control.emit = events.append
        control.local_report = packet(20,21.,21.1,6.)
        control.graphs = dict(A=Graph(),B=Graph())
        control.commands = [
            Message(22,22.,22.2,23.,25.,dict(A=5.,B=5.),-.002),
            Message(24,24.,24.2,25.,27.,dict(A=5.,B=5.),.004),
        ]
        self.assertIsNone(control.command_at(25,24.2))
        self.assertIsNone(control.command_at(26,27.))
        plants = dict(A=Plant(),B=Plant())
        for slot in (23,25):
            control.dispatch(slot,float(slot),slot-.2,.09,plants)
        contexts = [r["context"] for r in events if r["phase"]=="device_input" and r["site"]=="A"]
        self.assertAlmostEqual(contexts[0][1],-.2)
        self.assertAlmostEqual(contexts[1][1],.4)
        graph = control.graphs["A"]
        head = control.model.heads["batteryA"]
        first = torch.tensor(contexts[0])
        changed = first.clone()
        changed[1] = contexts[1][1]
        args = [(torch.zeros((2,18)),graph.adj,[0],ctx) for ctx in (first,changed)]
        self.assertFalse(torch.equal(head.actor(*args[0]),head.actor(*args[1])))
        grad = [torch.autograd.grad(head.actor(*a)[0],head.actor.out.layers[0].weight)[0] for a in args]
        self.assertFalse(torch.equal(*grad))
