import unittest
from types import SimpleNamespace

from scripts.check_lfc34_operating_point import checks, phases_and_terminals


class OperatingPointTest(unittest.TestCase):
    def test_winding_terminal_decoding(self):
        values = [3 + 4j, 4 - 2j, 0j, -3 - 4j, -3 + 2j, 0j]
        e = SimpleNamespace(NumTerminals=2, NumConductors=3, Powers=values,
                            BusNames=["up", "down"], NodeOrder=[1, 2, 0, 1, 2, 0],
                            CurrentsMagAng=[5, 0, 4, 0, 0, 0] * 2)
        measured = phases_and_terminals(e)
        e.Powers = [x for c in values for x in (c.real, c.imag)]
        self.assertEqual(measured, phases_and_terminals(e))
        self.assertEqual([x["p_kw"] for x in measured], [7, -6])
        self.assertEqual(measured[0]["conductors"][1]["terminal_current_amp"], 4)
        self.assertGreater(measured[0]["sum_conductor_s_kva"], measured[0]["s_kva"])

    def test_gate_and_no_ranking_when_nonconverged(self):
        self.assertTrue(checks({"converged": False}, .95, 1.05)["unscored"])
        r = {"converged": True, "network_power_balance_ok": False,
             "batteries": {"b1": {"initial_soc": 8, "observed_wrapper_soc": 8}},
             "subxf_windings": [{"net_s_over_rated": 1.001}],
             "voltage_phase_pu": {"800.1": {"v_pu": .94}, "800.2": {"v_pu": 1.05}}}
        got = checks(r, .95, 1.05)
        self.assertFalse(got["power_balance_ok"])
        self.assertEqual(got["invalid_soc_devices"], ["b1"])
        self.assertEqual(got["overrated_windings_net_s"], [1])
        self.assertEqual(got["out_of_band_phase_nodes"], ["800.1"])
