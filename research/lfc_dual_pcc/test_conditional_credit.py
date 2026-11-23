from dataclasses import replace
import unittest

import torch

from .clock_graph import ClockGraphPolicy, Message, tensor
from .conditional_credit import Branch, MeterStep, Pair, branch_returns, fit_and_update
from .driver import state_digest


def example(role, action, other, mask, fingerprint):
    old = Message(22,22.,22.1,23.,25.,{"A":2.,"B":1.})
    issued = Message(24,24.,24.1,25.,27.,{"A":1.,"B":2.})
    slow = role == "master"
    meter = (MeterStep(24,24.,22,26.,27.,-.4),MeterStep(25,25.,24,27.,28.,-1.)) if slow else (
        MeterStep(25,25.,24,27.,28.,-1.),MeterStep(26,26.,24,28.,29.,-1.))
    fact = Branch(role,24 if slow else 25,24. if slow else 25.,action,(.2,.4,.3),tuple(mask),
                  "same-before-action","factual-test-trace","same-queue",fingerprint,"keyed-noise","team-objective",
                  old if slow else issued,issued if slow else None,meter)
    changed = replace(fact, action=other, issued=replace(issued,data={"A":2.,"B":1.}) if slow else None,
                      meter=(meter[0],replace(meter[1],team_reward_rate=-2.)) if slow else
                            (replace(meter[0],team_reward_rate=-2.),meter[1]))
    return Pair(fact,changed,tuple(sorted((action,other))))


class ConditionalCreditTest(unittest.TestCase):
    def test_slow_only_reaches_the_meter_after_arrival_and_activation(self):
        pair=example("master",0,1,[True,True],"policy")
        observed, alternative=branch_returns(pair)
        self.assertAlmostEqual(observed-alternative,.95)
        bad=replace(pair.alternative, pending_queue="different-earlier-actions")
        with self.assertRaisesRegex(ValueError,"pending_queue"):
            branch_returns(replace(pair,alternative=bad))
        bad=replace(pair.alternative,meter=(replace(pair.alternative.meter[0],team_reward_rate=1.),
                                           pair.alternative.meter[1]))
        with self.assertRaisesRegex(ValueError,"before the intervention"):
            branch_returns(replace(pair,alternative=bad))

    def test_fast_holds_active_command_and_physical_duration(self):
        pair=example("feederA",0,1,[True,True],"policy")
        observed,alternative=branch_returns(pair)
        self.assertAlmostEqual(observed-alternative,1.)
        bad=replace(pair.alternative,active=replace(pair.alternative.active,data={"A":1.}))
        with self.assertRaisesRegex(ValueError,"active"):
            branch_returns(replace(pair,alternative=bad))
        bad=replace(pair.alternative,meter=pair.alternative.meter[:-1])
        with self.assertRaises(ValueError):
            branch_returns(replace(pair,alternative=bad))

    def test_both_losses_update_real_slow_and_fast_actors_with_equal_budget(self):
        torch.manual_seed(1701)
        policy = ClockGraphPolicy()
        fingerprint = state_digest(policy)
        records, grouped, pairs = [], {}, []
        for role,width,n in (("master",31,9),("feederA",14,6)):
            mask=[True,True]+[False]*(n-2)
            action,out=policy.heads[role].decision((tensor([.5]*width),),mask,False)
            pair=example(role,action,1-action,mask,fingerprint)
            records.append(dict(role=role,slot=pair.factual.slot,outputs=out))
            grouped[role]=[(26 if role=="master" else 25,2.,-1.,out)]
            pairs.append(pair)
        meters={(row.factual.role,row.factual.slot):row.factual.meter for row in pairs}
        with self.assertRaisesRegex(ValueError,"on-policy sample"):
            fit_and_update(policy,records,grouped,pairs,.01,factual_trace="wrong-trace",factual_meters=meters)
        result=fit_and_update(policy,records,grouped,pairs,.01,
                              factual_trace="factual-test-trace",factual_meters=meters)
        self.assertEqual(state_digest(policy),fingerprint)
        first, second = result["independent_q"], result["paired_q"]
        self.assertEqual(first["critic_steps"],second["critic_steps"])
        self.assertNotEqual(first["critic_loss"],second["critic_loss"])
        self.assertNotEqual(first["advantage_by_role_slot"],second["advantage_by_role_slot"])
        for v in result.values():
            opt=v["actor_update"]
            self.assertEqual(len(opt["steps"]),2)
            for role in grouped:
                name=f"heads.{role}.actor.layers.2.weight"
                self.assertGreater(opt["deltas"][name],0.)
            for step in opt["steps"]:
                self.assertTrue(step["samples"][0]["conditional_credit"])
                self.assertAlmostEqual(step["samples"][0]["ratio"],1.,places=6)

    def test_partial_real_coverage_does_not_start_the_comparison(self):
        with self.assertRaisesRegex(ValueError,"no certified"):
            fit_and_update(ClockGraphPolicy(),[],{},[],.01,factual_trace="test",factual_meters={})
