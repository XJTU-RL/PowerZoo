import json
import unittest
from .handoff import HANDOFF,Release


class ReleaseGate(unittest.TestCase):
    def test_only_two_post_issue_measurements_can_release(self):
        cfg=json.loads(HANDOFF.read_bytes())
        a,b=Release("issued",cfg),Release("delivered",cfg)
        traces=[]
        for t in range(22,29):
            old=None if t<24 else {"slot":t-2,"delivered":{"A":4,"B":8}}
            traces.append([x.dispatch(t,.001,old,True)[1] for x in (a,b)])
        self.assertEqual(a.issue,23)
        self.assertEqual(a.release_at,23)
        self.assertEqual(b.release_at,26)
        self.assertEqual(b.reason,"B_PCC_confirmed")

    def test_deadline_and_bounded_shared_backup(self):
        c=json.loads(HANDOFF.read_bytes());r=Release("delivered",c)
        for t in range(22,35):
            old=None if t<24 else {"slot":t-2,"delivered":{"A":0,"B":0}}
            r.dispatch(t,.001,old,True)
        self.assertEqual(r.release_at-r.issue,c['max_wait_after_issue_s'])
        self.assertEqual(r.reason,'common_deadline')
        self.assertEqual(r.backup,c['backup_max_ticks'])
