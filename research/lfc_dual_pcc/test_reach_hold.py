import unittest
from .reach_hold import select

RULE={"power_resolution_kw":.1,"power_guard_kw":.005,"reach_deadline_s":3,"report_local_step_s":1,"hold_seconds":[2,5,10]}


class ReachHold(unittest.TestCase):
    def test_reach_step_is_not_a_sustain_step(self):
        self.assertEqual(select({8:[4.,8.,8.,0.,0.],0:[0.]*5},2,RULE)[:3],(3.9,1,8))

    def test_rise_time_is_distinct_from_duration(self):
        self.assertEqual(select({8:[2.,5.,8.2,8.2,8.2],0:[0.]*5},2,RULE)[:3],(8.1,3,8))

    def test_late_crossing_is_not_accepted(self):
        self.assertEqual(select({8:[0.,0.,0.,8.,8.,8.]},2,RULE)[:2],(0.,1))

    def test_equal_deliveries_prefer_earliest_then_smallest_ceiling(self):
        self.assertEqual(select({10:[6.6]*7,8:[6.6]*7},5,RULE)[:3],(6.5,1,8))
