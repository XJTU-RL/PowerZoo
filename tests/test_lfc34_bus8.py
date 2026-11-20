from dataclasses import replace
import unittest

from envs.lfc.diagnostics.ieee34_bus8 import AnchoredBus8, BUS8_ANCHOR, Snapshot


def reference():
    return Snapshot(0., -1486.417, 138.427, 0., (1.,)*95, (.7,)*4, (.9,)*4,
                    (0,)*6, 0., 0., 0, True, True)


class AnchorTest(unittest.TestCase):
    def test_neutral_preserves_anchor_and_signed_export(self):
        x = reference()
        a = AnchoredBus8(x, initial_write_target_s=20.)
        r = a.project(x, index=0, observed_pause_s=20 + 2/60)
        self.assertEqual(r.feeder_import_kw_kvar, (1486.417, -138.427))
        self.assertEqual(r.bus8_load_w_var, tuple(y*1000 for y in BUS8_ANCHOR))
        with self.assertRaisesRegex(ValueError, "zero-delta"):
            a.project(replace(x, source_terminal_kw=-1487.), index=0, observed_pause_s=20.)

    def test_unity_signed_perturbation(self):
        x = reference()
        a = AnchoredBus8(x, initial_write_target_s=20.)
        for q in (-20., 20.):
            y = replace(x, time_s=1., source_terminal_kw=-1516.417,
                        source_terminal_kvar=138.427-q)
            r = a.project(y, index=1, observed_pause_s=21.)
            self.assertAlmostEqual(r.feeder_delta_kw_kvar[0], 30.)
            self.assertAlmostEqual(r.bus8_load_w_var[1], (BUS8_ANCHOR[1]+q)*1000)

    def test_good_reference_required(self):
        with self.assertRaisesRegex(ValueError, "published static"):
            AnchoredBus8(replace(reference(), source_terminal_kvar=130.), initial_write_target_s=20.)

    def test_envelope_and_clocks(self):
        a = AnchoredBus8(reference(), initial_write_target_s=20.)
        with self.assertRaisesRegex(ValueError, "clocks"):
            a.project(replace(reference(), time_s=1.), index=1, observed_pause_s=21.04)
        with self.assertRaisesRegex(ValueError, "clocks"):
            a.project(replace(reference(), time_s=.5), index=1, observed_pause_s=21.)
        with self.assertRaisesRegex(ValueError, "envelope"):
            a.project(replace(reference(), time_s=1., source_terminal_kw=-1540.),
                      index=1, observed_pause_s=21.)

    def test_physical_failures(self):
        a = AnchoredBus8(reference(), initial_write_target_s=20.)
        for edits in (
            dict(converged=False), dict(controls_settled=False),
            dict(voltages_pu=(1.,)*94), dict(voltages_pu=(1.051,)*95),
            dict(winding_kva_ratios=(.2, .2, .2, 1.01)),
            dict(phase_current_ratios=(.2, .2, 1.01, .9)),
            dict(enabled_storage_count=1), dict(residual_kw=.02),
            dict(remaining_source_terminal_kva=.01),
            dict(source_terminal_kw=float("nan")), dict(taps=(1, 0, 0, 0, 0, 0)),
        ):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                a.project(replace(reference(), time_s=1., **edits),
                          index=1, observed_pause_s=21.)
