"""Synthetic tests of causal reports, bounded allocation and the live coordinator."""
from dataclasses import replace
from itertools import product
import unittest

import torch

from .clock_graph import ClockGraphPolicy, Coordinator, Message, slow_input
from .master_dispatch import allocate_support, bid_limits, feasible_actions, site_reports
from .test_clock_graph import packet


class MasterDispatchTests(unittest.TestCase):
    def test_allocate_support_uses_available_headroom(self):
        for action, cap_a, cap_b in product(range(9), (0., 1., 9.), (0., 2., 20.)):
            limits = dict(A=cap_a, B=cap_b)
            alloc, request = allocate_support(.1, action, dict(A=None, B=None), limits, 10., 100.)
            self.assertAlmostEqual(sum(alloc.values()), min(request, sum(limits.values())))
            for site, value in alloc.items():
                self.assertGreaterEqual(value, 0.)
                self.assertLessEqual(value, limits[site])
        for error in (0., -.1):
            report = packet(1, 2., 2.1, 4.)
            alloc, request = allocate_support(error, 8, dict(A=report, B=report), dict(A=4., B=4.), 10., 100.)
            self.assertEqual((alloc, request), (dict(A=0., B=0.), 0.))

    def test_asynchronous_reports_and_conflicting_redelivery(self):
        first = packet(0, 1., 1.1, 2.)
        second = packet(1, 2., 2.1, 9.)
        only_a = replace(second, data={"A": second.data["A"]})
        retransmit = replace(only_a, arrival=2.3)
        corrupt = replace(only_a, arrival=2.4, data={"A": dict(second.data["A"], bid_kw=5.)})
        invalid = replace(second, sequence=2, data={"A": dict(second.data["A"], p_kw=float("nan"))})
        infinite = replace(only_a, sequence=3, expiry=float("inf"))
        oversized = replace(only_a, sequence=4, data={"A": dict(second.data["A"], q_kvar=10**500)})
        self.assertEqual(site_reports([only_a, first], 2.1), dict(A=first, B=first))
        self.assertEqual(site_reports([retransmit, first, infinite, oversized, only_a, invalid], 2.3),
                         dict(A=only_a, B=first))
        self.assertEqual(site_reports([corrupt, first, only_a], 2.5), dict(A=first, B=first))
        selected = site_reports([retransmit, only_a, first, invalid], 4.)
        self.assertEqual(selected, dict(A=only_a, B=None))
        x = slow_input(.01, .2, selected, 4.)
        self.assertEqual(x.shape, (31,))
        self.assertEqual(x[1:5].tolist(), [1., 2., 0., 0.])
        self.assertEqual(x[18:].tolist(), [0.]*13)
        self.assertEqual(site_reports([only_a, first], 5.), dict(A=None, B=None))

    def test_limits_require_one_full_fast_step(self):
        msg = packet(0, 1., 1.1, 2.)
        msg = replace(msg, data={
            "A": dict(msg.data["A"], available_kw=1.5, bid_kw=3.),
            "B": dict(msg.data["B"], duration_s=.5),
        })
        limits = bid_limits(site_reports([msg], 2.), 1.)
        self.assertEqual(limits, dict(A=1.5, B=0.))

    def test_zero_target_preserves_safe_ramp_down(self):
        table = [[0, 0], [4, 0], [8, 0]]
        rows = [{"veto": ["ramp"]}, {"veto": []}, {"veto": []}]
        self.assertEqual(feasible_actions(table, rows, 0.), [1])
        self.assertEqual(feasible_actions(table, rows, 2.), [1, 2])
        with self.assertRaisesRegex(ValueError, "no feasible feeder action"):
            feasible_actions(table, rows[:1], 0.)

    def test_coordinator_survives_single_uplink_loss_and_downlink_gap(self):
        table = [[0, 0], [0, 2], [4, 0], [8, 0], [4, 2], [8, 2]]
        cfg = dict(window_slots=[22, 24], frequency_scale_hz=.01, max_request_kw=18.,
                   droop_kw_hz=1000., fast_s=1., clock={"report_latency_s": .1},
                   actions_battery_pv_kw=table)
        events = []
        control = Coordinator(ClockGraphPolicy(), cfg, True, fixed_master=True)
        control.emit = events.append

        class Graph:
            adj = torch.zeros((3, 2, 2))
            pools = ([0], [1])

            def features(self, plant, request):
                return torch.zeros(2, 18)

        class Plant:
            def __init__(self, veto_zero):
                self.base = (100., 0.)
                self.veto_zero = veto_zero
                self.only_action = None

            def probe(self, action):
                if self.only_action is not None and action != self.only_action:
                    return {"veto": ["network"]}
                return {"veto": ["ramp"] if self.veto_zero and action in (0, 1) else [],
                        "pcc": [100.-sum(table[action]), 0.]}

            def candidates(self):
                return [self.probe(i) for i in range(len(table))]

        available = packet(20, 21., 21.1, 6.)
        control.local_report = available
        control.reports = [replace(available, data={"A": available.data["A"]})]
        plants = {"A": Plant(False), "B": Plant(True)}
        control.graphs = {s: Graph() for s in plants}
        request, actions = control.dispatch(22, 22., 21.8, .02, plants)
        self.assertEqual(request, dict(A=0., B=0.))
        self.assertEqual(actions, dict(A=0, B=2))
        master = next(r for r in events if r.get("role") == "master")
        self.assertEqual(master["reports"], dict(A=20, B=None))
        self.assertEqual(master["action"], 4)
        self.assertEqual(control.report_consumer, {(20, "A"): 22})
        request, actions = control.dispatch(23, 23., 22.8, .02, plants)
        self.assertEqual(request, dict(A=6., B=0.))
        self.assertEqual(actions["B"], 2)
        self.assertNotIn("logit_offsets", next(r for r in events if r.get("role") == "batteryA"))
        control.tracking_logit_weight = 1.
        plants["A"].only_action = 1
        request, actions = control.dispatch(24, 24., 23.8, .02, plants)
        self.assertEqual(request, dict(A=6., B=0.))
        self.assertEqual(actions["A"], 1)
        self.assertEqual(control.report_consumer, {(20, "A"): 22})
        a_pv = next(r for r in events if r.get("role") == "pvA" and r["slot"] == 24)
        self.assertEqual(a_pv["mask"], [False, True])
        self.assertEqual(a_pv["logit_offsets"], [0., 0.])
        control.commands.clear()
        request, actions = control.dispatch(25, 25., 24.8, .02, plants)
        self.assertEqual(request, dict(A=0., B=0.))
        self.assertEqual(actions, dict(A=1, B=2))
        self.assertEqual(events[-1]["phase"], "command_fallback")

    def test_unexpected_overlap_does_not_hide_the_scheduled_command(self):
        control = Coordinator(None, {"window_slots": [22, 24]}, True)
        expected = Message(22, 22., 22.1, 23., 25., dict(A=6., B=0.))
        overlap = Message(24, 23.1, 23.2, 23.3, 27., dict(A=0., B=4.))
        control.commands = [overlap, expected]
        self.assertIs(control.command_at(24, 24.), expected)
        self.assertIs(control.command_at(25, 25.), overlap)

    def test_offers_only_receive_credit_from_their_own_first_consumer(self):
        control = Coordinator(None, {"frequency_scale_hz": 1., "slow_s": 2}, True)
        events = []
        control.emit = events.append
        control.report_consumer = {(24, "A"): 26}
        control.records = [
            dict(role=role, slot=slot, command=command, outputs=None)
            for command in (22, 26)
            for slot, roles in ((command, ("master",)), (command+1, ("feederA", "feederB")),
                                (command+2, ("feederA", "feederB")))
            for role in roles
        ]
        frames = [dict(slot=k, used_slot=k-2 if k in (25, 26, 29, 30) else None,
                       write_s=float(k), end_s=float(k+1), request=dict(A=0., B=0.), delivered=dict(A=0., B=0.))
                  for k in range(23, 31)]
        trace = [(k+.2, 48. if k >= 29 else 49.) for k in range(23, 31)]
        grouped = control.returns(frames, trace, 50.)
        self.assertLess(grouped["feederA"][1][2], grouped["feederB"][1][2])
        self.assertEqual([r["role"] for r in events if r.get("offer_consuming_master") == 26], ["feederA"])


if __name__ == "__main__":
    unittest.main()
