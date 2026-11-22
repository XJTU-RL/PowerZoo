"""Protocol tests for the equal-service master experiment."""
import unittest

from .equal_service import service_kw, split_candidates
from .run_equal_service import ledger


class EqualServiceTests(unittest.TestCase):
    def test_projection_preserves_common_total_and_masks_duplicates(self):
        options, mask = split_candidates(10., dict(A=11., B=4.))
        self.assertEqual(options, [dict(A=6.,B=4.),dict(A=6.,B=4.),dict(A=8.,B=2.)])
        self.assertEqual(mask,[False,True,True])
        options, mask = split_candidates(10., dict(A=7., B=2.))
        self.assertEqual(options,[dict(A=7.,B=2.)]*3)
        self.assertEqual(mask,[False,True,False])
        self.assertEqual([service_kw(k) for k in (22,24,34,36)],[0.,10.,10.,0.])

    def test_common_ledger_uses_actual_signed_delivery_and_consuming_time(self):
        def hold(slot, used, request, delivered):
            return dict(slot=slot, used_slot=used, write_s=float(slot), end_s=float(slot+1),
                        request=request, delivered=delivered, pcc=dict(A=[1,2],B=[3,4]),
                        actual=dict(A=dict(veto=[]),B=dict(veto=[])))
        rows = ledger([hold(25,None,dict(A=6.,B=3.),dict(A=7.,B=-1.)),
                       hold(27,25,dict(A=5.,B=5.),dict(A=12.,B=1.))],
                      dict(A=[[1,2],[3,4]],B=[[5,6],[7,8]]))
        self.assertEqual((rows[0]["external_demand_kw"],rows[0]["signed_gap_kw"]),(10.,4.))
        self.assertEqual(rows[0]["physical_slot"],27)
        self.assertEqual(rows[0]["native_port_kw_kvar"]["A"],[3,4])
        self.assertEqual((rows[1]["unmet_kw"],rows[1]["excess_kw"]),(0.,3.))
