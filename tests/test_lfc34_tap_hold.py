"""Pure tests and mocked command scope; no simulator imports."""
from types import SimpleNamespace
import unittest

from scripts.check_lfc34_tap_hold import PULSE, apply_arm, tap_differences, verify_taps


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
