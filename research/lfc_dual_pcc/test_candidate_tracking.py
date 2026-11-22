"""Small counterexamples for the optional candidate tracking prior."""
import unittest

import torch
from torch import nn

from .candidate_tracking import BATTERY_LEVELS, PV_LEVELS, tracking_offsets
from .clock_graph import ClockGraphPolicy, PolicyValue, update
from .master_dispatch import feasible_actions


TABLE = [[0, 0], [0, 2], [4, 0], [8, 0], [4, 2], [8, 2]]
BASE_KW = 100.
# Measured PCC response need not equal the sum of the two device setpoints.
SUPPORT = [0., 1.9, 4.3, 8.4, 6.7, 10.6]


def rows():
    result = [dict(veto=[], pcc=[BASE_KW-support, 0.]) for support in SUPPORT]
    result[-1]["veto"] = ["voltage"]
    return result


def zero_head(width):
    head = PolicyValue(nn.Linear(1, width), nn.Linear(1, 1))
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    return head


class CandidateTrackingTests(unittest.TestCase):
    def test_crossing_a_physical_boundary_changes_prior_preference(self):
        previews = rows()
        eligible = feasible_actions(TABLE, previews, 1.)
        battery = zero_head(3)
        pv = zero_head(2)
        args = (torch.zeros(1),)

        def pick(target):
            off = tracking_offsets(TABLE, previews, eligible, BASE_KW, target, 1.)
            b_idx, _ = battery.decision(args, [True]*3, True, logit_offsets=off)
            b = BATTERY_LEVELS[b_idx]
            pv_mask = [any(TABLE[i] == [b, v] for i in eligible) for v in PV_LEVELS]
            off = tracking_offsets(TABLE, previews, eligible, BASE_KW, target, 1., battery=b)
            v_idx, _ = pv.decision(args, pv_mask, True, logit_offsets=off)
            return TABLE.index([b, PV_LEVELS[v_idx]])

        self.assertEqual(pick(1.8), 1)
        self.assertEqual(pick(5.8), 4)
        self.assertEqual(pick(6.2), 4)
        self.assertEqual(pick(11.), 3)
        self.assertEqual(pick(12.), 3)
        self.assertEqual(
            tracking_offsets(TABLE, previews, eligible, BASE_KW, 11., 1.),
            tracking_offsets(TABLE, previews, eligible, BASE_KW, 12., 1.),
        )

    def test_veto_still_blocks_and_learned_score_can_override_prior(self):
        previews = rows()
        eligible = feasible_actions(TABLE, previews, 1.)
        battery, pv = zero_head(3), zero_head(2)
        with torch.no_grad():
            battery.actor.bias[2] = 20.
            pv.actor.bias[1] = 20.
        args = (torch.zeros(1),)
        off = tracking_offsets(TABLE, previews, eligible, BASE_KW, 0., 1.)
        self.assertNotEqual(off[2], max(off))
        self.assertEqual(battery.decision(args, [True]*3, True, logit_offsets=off)[0], 2)
        off = tracking_offsets(TABLE, previews, eligible, BASE_KW, 10.6, 1., battery=8)
        self.assertEqual(pv.decision(args, [True, False], True, logit_offsets=off)[0], 0)

    def test_zero_weight_and_behavior_ratio(self):
        model = ClockGraphPolicy()
        head = model.heads["batteryA"]
        args = (torch.zeros((2, 18)), torch.zeros((3, 2, 2)), [0], torch.zeros(6))
        previews = rows()
        eligible = feasible_actions(TABLE, previews, 1.)
        zero = tracking_offsets(TABLE, previews, eligible, BASE_KW, 6., 0.)
        self.assertTrue(torch.equal(head.logits(args), head.logits(args, zero)))
        off = tracking_offsets(TABLE, previews, eligible, BASE_KW, 6., 1.)
        _, record = head.decision(args, [True]*3, True, logit_offsets=off)
        self.assertEqual(record["logit_offsets"], off)
        stats = update(model, {"batteryA": [(25, 1., -10., record)]}, .003)
        self.assertAlmostEqual(stats["steps"][0]["samples"][0]["ratio"], 1.)
        self.assertGreater(stats["deltas"]["heads.batteryA.actor.out.layers.2.bias"], 0.)


if __name__ == "__main__":
    unittest.main()
