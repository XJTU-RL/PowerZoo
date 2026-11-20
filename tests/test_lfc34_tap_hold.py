"""Pure tests and mocked command scope; no simulator imports."""
from types import SimpleNamespace
import unittest

from scripts.check_lfc34_tap_hold import (
    PULSE, all_finite, apply_arm, finite_output, read_dss_clock, record_clock,
    tap_differences, verify_taps,
)


class TapHoldTest(unittest.TestCase):
    def test_failed_readback_retains_name_expected_and_observed(self):
        before = {"reg1a": {"index": 1, "ratio": 1.00625}, "reg2b": {"index": 2, "ratio": 1.0125}}
        after = {"reg1a": {"index": 0, "ratio": 1.0}}
        audit = {}
        with self.assertRaisesRegex(RuntimeError, "after"):
            verify_taps(before, after, "after", audit)
        self.assertEqual(audit["after_mismatches"]["reg1a"],
                         {"expected": before["reg1a"], "observed": after["reg1a"]})
        self.assertEqual(audit["after_mismatches"]["reg2b"]["observed"], None)
        self.assertEqual(tap_differences(before, before), {})

    def test_pulse_scope_does_not_assign_or_retarget_taps(self):
        context = SimpleNamespace(Text=SimpleNamespace(Command=None))
        circuit = SimpleNamespace(AllElementNames=[])
        apply_arm(context, circuit, (0., 0.))
        self.assertIsNone(context.Text.Command)
        apply_arm(context, circuit, (0., -15.))
        self.assertTrue(context.Text.Command.startswith("New Load."))
        self.assertIn("Model=1 Status=Fixed kW=0.0 kvar=-15.0", context.Text.Command)
        circuit.AllElementNames = [PULSE.lower()]
        apply_arm(context, circuit, (0., 0.))
        self.assertEqual(context.Text.Command, f"Edit {PULSE} Enabled=no")
        apply_arm(context, circuit, (20., 0.))
        self.assertEqual(context.Text.Command, f"Edit {PULSE} Enabled=yes kW=20.0 kvar=0.0")


    def test_dss_clock_does_not_use_arm_ordinal(self):
        sol = SimpleNamespace(Hour=0, Seconds=0., dblHour=0.,
                              ControlMode=-1, ControlActionsDone=True)
        c = SimpleNamespace(Solution=sol)
        start = read_dss_clock(sol)
        audit = {}
        self.assertEqual(record_clock(c, start, audit), 0.)
        self.assertIsNone(audit["per_arm_static_settlement"])
        self.assertEqual(audit["native_control_mode"], "OFF")
        self.assertTrue(audit["native_control_actions_done_flag"])
        sol.Seconds, sol.dblHour = 1., 1/3600
        with self.assertRaisesRegex(RuntimeError, "advanced"):
            record_clock(c, start, audit)
        self.assertEqual(audit["native_clock"]["absolute_seconds"], 1.)

    def test_invalid_clock_and_failed_nonfinite_diagnostics(self):
        bad = SimpleNamespace(Hour=0, Seconds=0., dblHour=float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            read_dss_clock(bad)
        evidence = {"passed": False, "raw": (float("inf"), 1.2)}
        self.assertFalse(all_finite(evidence))
        self.assertEqual(finite_output(evidence), {"passed": False, "raw": [None, 1.2]})
