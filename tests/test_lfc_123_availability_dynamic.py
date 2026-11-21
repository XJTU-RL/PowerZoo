from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from envs.lfc.native_123_availability import DeviceDispatchMismatch, OperatingPoint, envelope
from envs.lfc.native_feeder_123 import PQ, StorageReading
from scripts.check_lfc_123_availability_dynamic import _hold, trial


MOD = "scripts.check_lfc_123_availability_dynamic."


class TestDynamicTrial(unittest.TestCase):
    ids = tuple(f"Storage.batt{i}" for i in range(1, 5))
    zero = dict.fromkeys(ids, 0)
    joint = dict.fromkeys(ids, 25)

    def setUp(self):
        self.live = Mock()
        self.live.storage_ids = self.ids
        self.before = OperatingPoint(0, (), (.5,) * 4, (PQ(0, 0),) * 4, PQ(3600, 1300), ())
        self.projected = replace(self.before, storage_pq=(PQ(25, 0),) * 4)
        self.after_soc = .5 - 25 / (3600 * .95 * 200)
        self.after = replace(self.projected, clock_s=1, soc=(self.after_soc,) * 4)
        self.live.step.return_value = SimpleNamespace(
            start_s=0, end_s=1, active_balance_error_kw=0, feeder_import=PQ(3600, 1300),
            storage=tuple(StorageReading(d, 25, 25, PQ(25, 0), .5, self.after_soc)
                          for d in self.ids),
        )

    def test_rejected_or_wrong_pre_step_prediction_never_advances_live(self):
        with patch(MOD+"collect", return_value=self.before), \
             patch(MOD+"_point"), patch(MOD+"prospective", side_effect=RuntimeError("rejected")):
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                _hold(self.live, self.zero, self.joint, envelope(), Mock())
        with patch(MOD+"collect", return_value=self.before), \
             patch(MOD+"_point"), patch(MOD+"prospective", return_value=self.after):
            with self.assertRaisesRegex(RuntimeError, "pre-step"):
                _hold(self.live, self.zero, self.joint, envelope(), Mock())
        self.live.step.assert_not_called()

    def test_applies_exact_vector_once_and_checks_end_soc_separately(self):
        spec = envelope()
        emit = Mock()
        with patch(MOD+"collect", side_effect=(self.before, self.after)), \
             patch(MOD+"_point"), patch(MOD+"verify_device_envelope"), \
             patch(MOD+"prospective", return_value=self.projected):
            result = _hold(self.live, self.zero, self.joint, spec, emit)
        self.live.step.assert_called_once_with(self.joint)
        self.assertEqual(result, self.after)
        self.assertEqual([c.args[0]["phase"] for c in emit.call_args_list], [
            "hold_start", "hold_predicted_no_finish", "hold_observed_after_finish", "hold_verified",
        ])
        self.assertLess(emit.call_args.args[0]["ledger"][0]["expected_delta_kwh"], 0)

    def test_bad_reference_rejected_between_joint_and_zero(self):
        spec, emit = envelope(), Mock()
        with patch(MOD+"_hold") as hold, patch(MOD+"collect", return_value=self.after), \
             patch(MOD+"prospective", side_effect=DeviceDispatchMismatch(
                 "Storage.batt1: signed terminal P/Q delivery outside screen"
             )):
            self.assertEqual(trial(self.live, spec, emit)["status"], "two_native_holds_verified")
        self.assertEqual(hold.call_args_list, [
            call(self.live, self.zero, self.joint, spec, emit),
            call(self.live, self.joint, self.zero, spec, emit),
        ])
        self.assertEqual(emit.call_args.args[0]["phase"], "stale_reference_rejected")

    def test_wrapped_direct_failure_or_transient_live_parity_stops_before_t1(self):
        spec, emit = envelope(), Mock()
        signed = "Storage.batt1: signed terminal P/Q delivery outside screen"
        rejected = DeviceDispatchMismatch(signed)
        wrapper = RuntimeError(f"LIVE feeder parity failed after scratch rejection: {signed}")
        wrapper.__cause__ = rejected
        for error, states in ((wrapper, (self.after, self.after)),
                              (rejected, (self.after, replace(self.after, soc=(.42,) * 4)))):
            with self.subTest(error=error), patch(MOD+"_hold") as hold, \
                 patch(MOD+"collect", side_effect=states), \
                 patch(MOD+"prospective", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "parity|baseline differs"):
                    trial(self.live, spec, emit)
            hold.assert_called_once_with(self.live, self.zero, self.joint, spec, emit)
        emit.assert_not_called()
