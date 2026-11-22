"""Counterfactual input tests on identical parameters/actions; no simulated performance."""
import unittest

import torch

from .clock_graph import ClockGraphPolicy, Message, latest, slow_input, tensor, update


def packet(n, src, arrival, offer):
    d = dict(p_kw=1480., q_kvar=-130., delivered_kw=2., available_kw=offer,
             duration_s=1., energy_only_s=5., ramp_kw_s=4., ramp_limited=True,
             energy_limited=False, network_limited=False)
    return Message(n,src,arrival,src,src+3,{"A":d,"B":d})


def step(role, inputs):
    torch.manual_seed(1701)
    model = ClockGraphPolicy()
    m = model.heads[role]
    logits = m.actor(*inputs)
    lp = logits.log_softmax(0)[0]  # Fix the very same behavior action in both interventions.
    value = m.critic(*inputs).squeeze()
    baseline = {k:p.detach().clone() for k,p in m.named_parameters()}
    stats = update(model,{role:[(-.75,(lp,value,torch.tensor(0.)))]},.003)
    grad = {k:p.grad.clone() for k,p in m.named_parameters()}
    delta = {k:p.detach()-baseline[k] for k,p in m.named_parameters()}
    return logits.detach(), value.detach(), grad, delta


class ClockGraphTests(unittest.TestCase):
    def test_report_delivery_controls_learning(self):
        a = packet(0, 1.,1.1,2.)
        b = packet(1,2.,2.1,9.)
        def inputs(q, now):
            return (slow_input(.00015,.0002,latest(q,now),now,{"A":3., "B":4.}),)
        self.assertTrue(torch.equal(*[inputs(q,2.1)[0] for q in ([a],[a,b])]))
        self.assertIsNone(latest([b],5.))
        self.assertIs(latest([a,b],2.1001),b)
        cmd = Message(22,22.,22.1,23.,25.,{})
        self.assertIsNone(latest([cmd],22.9))
        self.assertIs(latest([cmd],23.),cmd)
        self.assertIsNone(latest([cmd],25.))
        x = step("master",inputs([a],2.2))
        y = step("master",inputs([a,b],2.2))
        self.assertFalse(torch.equal(x[0],y[0]))
        for prefix in ("actor","critic"):
            self.assertGreater(sum((g-y[2][k]).abs().sum().item() for k,g in x[2].items() if k.startswith(prefix)),.01)
            self.assertGreater(sum((g-y[3][k]).abs().sum().item() for k,g in x[3].items() if k.startswith(prefix)),1e-6)

    def test_graph_changes_actor_and_value_updates(self):
        torch.manual_seed(27)
        x = torch.randn(5,18)
        adj = torch.stack([torch.eye(5).roll(1,0),torch.eye(5).roll(2,0),torch.eye(5).roll(-1,0)])
        z = adj.clone()
        z[0,0] = z[0,0].roll(2)
        ctx = tensor([.3,.1,1.,1.,.9,1.])
        for role in ("batteryA","pvB"):
            a = step(role,(x,adj,[0],ctx))
            b = step(role,(x,z,[0],ctx))
            self.assertFalse(torch.equal(a[0],b[0]))
            self.assertFalse(torch.equal(a[1],b[1]))
            for prefix in ("actor","critic"):
                k = prefix+".message.weight"
                self.assertGreater(float((a[2][k]-b[2][k]).norm()),1e-5)
                self.assertGreater(float((a[3][k]-b[3][k]).norm()),1e-6)


if __name__ == "__main__":
    unittest.main()
