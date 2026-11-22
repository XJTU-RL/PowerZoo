import unittest
from .grouped_metrics import calibration, summary
from .grouped_source import condition, PROTOCOL, read


class GroupedContracts(unittest.TestCase):
    def test_five_second_witness_cannot_be_reported_as_ten(self):
        rows=[];pred=[]
        for h,k,t in ((2,9.,1),(5,8.,2),(10,4.,0)):
            rows.append({"key":["sample",24,"A",h,6],"kw":k,"time_class":t})
            pred.append({"power_kw":9.,"time_class":2,"p_witness":.8})
        m=summary(rows,pred)
        self.assertEqual((m["actual_certified_5"],m["certified_5_claimed_10"]),(1,1))
        self.assertEqual(m["duration_overcommit_rate"],1.)
        self.assertEqual(m["false_arrival_claims"],1)
        self.assertEqual(m["max_positive_power_excess_kw"],5.)

    def test_nearest_conservative_derating_uses_only_validation_grid(self):
        pairs=[{"key":["d",24,"B",h,4],"kw":5.,"time_class":1} for h in (2,5,10)]
        preds=[{"power_kw":5.7,"time_class":1,"p_witness":1.} for _ in pairs]
        self.assertEqual(calibration(pairs,preds,[0,.5,1,2],.1)["derate_kw"],1)

    def test_groups_are_unique_full_conditions(self):
        groups=read(PROTOCOL)["groups"]
        keys=set()
        for split,names in groups.items():
            for name in names:
                c,s=condition(name)
                self.assertEqual(s,split)
                key=(tuple(c["storage"]["initial_soc"].values()),tuple(c["local_exogenous"].values()))
                self.assertNotIn(key,keys);keys.add(key)
        self.assertEqual(len(keys),11)
