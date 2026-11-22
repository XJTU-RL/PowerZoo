"""Counterfactual input tests on identical parameters/actions; no simulated performance."""
import unittest

import torch

from .clock_graph import ClockGraphPolicy, Coordinator, Message, latest, slow_input, tensor, update
from .master_dispatch import site_reports


def packet(n, src, arrival, offer):
    d = dict(p_kw=1480., q_kvar=-130., delivered_kw=2., available_kw=offer,
             duration_s=1., energy_only_s=5., ramp_kw_s=4., ramp_limited=True,
             energy_limited=False, network_limited=False, consumed_request_kw=3., bid_kw=offer,
             consumed_s=src-1., consumed_command=0)
    return Message(n,src,arrival,src,src+3,{"A":d,"B":d})


def step(role, inputs):
    torch.manual_seed(1701)
    model = ClockGraphPolicy()
    m = model.heads[role]
    logits = m.actor(*inputs)
    lp = logits.log_softmax(0)[0]
    value = m.critic(*inputs).squeeze()
    baseline = {k:p.detach().clone() for k,p in m.named_parameters()}
    record = dict(args=inputs,mask=torch.ones_like(logits,dtype=torch.bool),action=torch.tensor(0),
                  old_logp=lp.detach(),old_value=value.detach())
    update(model,{role:[(24,1.,-.75,record)]},.003)
    grad = {k:p.grad.clone() for k,p in m.named_parameters()}
    delta = {k:p.detach()-baseline[k] for k,p in m.named_parameters()}
    return logits.detach(), value.detach(), grad, delta


class ClockGraphTests(unittest.TestCase):
    def test_slow_return_uses_both_measured_holds(self):
        ctrl = Coordinator(None, {"frequency_scale_hz":1.,"slow_s":2}, True)
        ctrl.emit = lambda row: None
        ctrl.records = [dict(role=role,slot=slot,command=22,outputs=None)
                        for role,slot in (("master",22),("feederA",23),("feederA",24))]
        frames = [dict(slot=k,used_slot=None,request={"A":0,"B":0},delivered={"A":0,"B":0}) for k in (23,24)]
        frames += [dict(slot=25,used_slot=23,write_s=25.,end_s=25.9),
                   dict(slot=26,used_slot=24,write_s=25.9,end_s=27.02)]
        grouped = ctrl.returns(frames,[(25.2,49.),(26.4,48.)],50.)
        _, duration, reward, _ = grouped["master"][0]
        self.assertAlmostEqual(duration,2.02)
        self.assertAlmostEqual(reward,-.9-.95**.9*2*1.12)
        self.assertEqual(len(grouped["feederA"]),2)

    def test_report_delivery_controls_learning(self):
        a = packet(0, 1.,1.1,2.)
        b = packet(1,2.,2.1,9.)
        def inputs(q, now):
            return (slow_input(.00015,.0002,site_reports(q,now),now),)
        self.assertTrue(torch.equal(*[inputs(q,2.1)[0] for q in ([a],[a,b])]))
        self.assertIsNone(latest([b],5.))
        self.assertIs(latest([a,b],2.1001),b)
        cmd = Message(22,22.,22.1,23.,25.,{})
        self.assertIsNone(latest([cmd],22.9))
        self.assertIs(latest([cmd],23.),cmd)
        self.assertIsNone(latest([cmd],25.))
        # A newer dispatch in transit cannot change the old sample's tracking error.
        self.assertAlmostEqual(float(inputs([a],2.2)[0][15]),.1,places=6)
        masked = packet(1,2.,2.1,9.)
        for site in masked.data.values():
            for k in ("p_kw","q_kvar","delivered_kw","bid_kw","available_kw"):
                site[k] = 0.
        original = step("master",inputs([b],2.2))
        ablated = step("master",inputs([masked],2.2))
        self.assertFalse(torch.equal(original[2]["actor.layers.0.weight"],ablated[2]["actor.layers.0.weight"]))
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
