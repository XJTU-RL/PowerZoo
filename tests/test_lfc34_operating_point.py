import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from scripts import check_lfc34_operating_point as audit
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


    def test_report_checkout_and_helper_origins(self):
        checkout = Path(audit.__file__).resolve().parents[1]
        expected = checkout / audit.CORE / "loadprofile.py"
        self.assertEqual(audit.require_report_repo(checkout), checkout)
        audit.require_helper_origin(SimpleNamespace(__file__=str(expected)), expected)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "checkout containing this script"):
                audit.require_report_repo(Path(td))
            with self.assertRaisesRegex(RuntimeError, "import origin"):
                audit.require_helper_origin(SimpleNamespace(__file__=str(Path(td) / expected.name)),
                                            expected)
        with self.assertRaisesRegex(RuntimeError, "import origin"):
            audit.require_helper_origin(SimpleNamespace(), expected)


    def test_matrix_keeps_every_frozen_scenario_in_order(self):
        failed = {"stages": {"standard_snapshot": {"converged": False}}}
        with patch.object(audit, "run_standard", side_effect=lambda *a, **kw: failed.copy()) as run:
            matrix = audit.run_boundary_matrix(Path("repo"), Path("scratch"),
                                               SimpleNamespace(vmin=.95, vmax=1.05))
        self.assertEqual(matrix["load_multipliers"], (.50, .75, 1.00))
        self.assertEqual([c.kwargs["load_multiplier"] for c in run.call_args_list], [.50, .75, 1.00])
        self.assertTrue(all(c.kwargs["source_pu"] == 1.05 and
                            c.args[2].standard_tolerance == 1e-9 for c in run.call_args_list))
        self.assertEqual(len(matrix["scenarios"]), 3)
        self.assertTrue(all(not s["matrix_screen"]["passed"] for s in matrix["scenarios"]))


    def test_shunt_study_keeps_all_fixed_rows(self):
        failed = {"stages": {"standard_snapshot": {"converged": False}}}
        with patch.object(audit, "run_standard", side_effect=lambda *a, **kw: failed.copy()) as run:
            study = audit.run_midload_shunt(Path("repo"), Path("scratch"),
                                           SimpleNamespace(vmin=.95, vmax=1.05))
        self.assertEqual(study["synthetic_nameplate_kvar"], (0, 75, 150))
        self.assertEqual([c.kwargs["study_shunt_kvar"] for c in run.call_args_list], [0, 75, 150])
        self.assertTrue(all(c.kwargs["load_multiplier"] == .75 and c.kwargs["source_pu"] == 1.05
                            for c in run.call_args_list))
        self.assertEqual(len(study["scenarios"]), 3)
        self.assertTrue(all(not s["matrix_screen"]["passed"] for s in study["scenarios"]))


    def test_shunt_evidence_rejects_wrong_conductor_inventory(self):
        wrong = SimpleNamespace(NumPhases=3, NumTerminals=2, NumConductors=4)
        fake = SimpleNamespace(Buses={"890": SimpleNamespace(Nodes=[1, 2, 3], Voltages=[1j, 2j, 3j])},
                               SetActiveElement=lambda name: 0, ActiveCktElement=wrong)
        with self.assertRaisesRegex(RuntimeError, "terminal/conductor inventory"):
            audit.collect_shunt_evidence(fake, 75)
