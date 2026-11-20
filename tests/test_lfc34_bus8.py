"""Synthetic contract fixtures only; never represent actual phase IDs or settings."""
from dataclasses import replace
import hashlib
import json
import unittest

from envs.lfc.diagnostics.ieee34_bus8 import (
    AnchorSequence, BUS8_ANCHOR, DECK_TREE, PINNED_BINDING, RATINGS,
    REGULATORS, STATIC_COMMIT, Snapshot, Winding,
)

PHASES = tuple((f"dummy{i}.1", 1.0) for i in range(95))
TAPS = tuple((name, 1.0) for name in REGULATORS)
WINDINGS = tuple(Winding(id_, bus, kva, kv, 1., ((0, 0.), (1, 1.), (2, 1.), (3, 1.)))
                 for id_, (bus, kva, kv) in RATINGS.items())


def context():
    windings = {}
    for x in WINDINGS:
        key = {"subxf": "SubXF", "xfm1": "XFM1"}[x.terminal_id.split(":")[0]]
        windings.setdefault(key, []).append(dict(
            terminal_bus=x.bus, rating_kva=x.rating_kva, rating_kv_ll=x.rating_kv_ll,
            passed_model_nameplate_screen=True, phase_current_amp_abc=[1.]*3, neutral_current_amp=0.))
    snap = dict(converged=True, network_power_balance_ok=True,
                voltage_phase_pu={i: {"v_pu": v} for i, v in PHASES},
                pcc_import_kw=1486.417, pcc_import_kvar=-138.427,
                enabled_electrical_elements_in_power_accounting=["vsource.source"])
    row = dict(study_model_feasible_static_candidate=True, source_commit=STATIC_COMMIT,
               input_tree=DECK_TREE, source_pu_command=1.045, native_load_multiplier=.75,
               shunt_study_evidence={"synthetic_shunt_nameplate_kvar": 75},
               native_voltage_convergence_tolerance=1e-9, control_mode="STATIC",
               disabled_der=list(PINNED_BINDING.original_disabled_der),
               native_control_actions_done=True, stages={"standard_snapshot": snap},
               regulator_secondary_taps=dict(TAPS), model_winding_screens=windings)
    return dict(mode="optimized-headroom-pair", source_pu=1.045, load_multiplier=.75,
                scenarios=[row])


def adapter(data=None):
    report = json.dumps(context() if data is None else data).encode()
    return AnchorSequence(report, approved_report_sha256=hashlib.sha256(report).hexdigest())


def sample():
    return Snapshot(PINNED_BINDING, 0., -1486.417, 138.427, 0., PHASES, WINDINGS,
                    TAPS, 0., 0., True, True)


def reset(a, snap=None):
    return a.reset(sample() if snap is None else snap,
                   initial_write_target_s=20., observed_pause_s=20+2/60)


def project(a, snap=None, *, index=1, pause=21.):
    return a.project(replace(sample(), time_s=1.) if snap is None else snap,
                     index=index, observed_pause_s=pause)


class AnchorTest(unittest.TestCase):
    def test_provenance(self):
        b = json.dumps(context()).encode()
        with self.assertRaisesRegex(ValueError, "digest"):
            AnchorSequence(b, approved_report_sha256="0"*64)
        d = context()
        d["scenarios"][0]["input_tree"] = "bad"
        with self.assertRaisesRegex(ValueError, "settings"):
            adapter(d)
        d = context()
        d["scenarios"][0]["regulator_secondary_taps"].pop("reg1a")
        with self.assertRaisesRegex(ValueError, "taps"):
            adapter(d)
        d = context()
        d["scenarios"][0]["stages"]["standard_snapshot"]["enabled_electrical_elements_in_power_accounting"].append("storage.batt1")
        with self.assertRaisesRegex(ValueError, "DER"):
            adapter(d)

    def test_neutral_and_signed_unity_delta(self):
        for q in (-20., 20.):
            a = adapter()
            c = reset(a)
            self.assertEqual(c.feeder_import_kw_kvar, (1486.417, -138.427))
            self.assertEqual(c.bus8_load_w_var, tuple(x*1000 for x in BUS8_ANCHOR))
            r = project(a, replace(sample(), time_s=1., source_terminal_kw=-1516.417,
                                   source_terminal_kvar=138.427-q))
            self.assertAlmostEqual(r.feeder_delta_kw_kvar[0], 30.)
            self.assertAlmostEqual(r.bus8_load_w_var[1], (BUS8_ANCHOR[1]+q)*1000)

    def test_neutral_and_sequence_fail_closed(self):
        a = adapter()
        with self.assertRaisesRegex(ValueError, "reset required"):
            project(a)
        with self.assertRaisesRegex(ValueError, "zero-delta"):
            reset(a, replace(sample(), source_terminal_kw=-1487.))
        with self.assertRaisesRegex(ValueError, "reset required"):
            project(a)
        reset(a)
        project(a)
        with self.assertRaisesRegex(ValueError, "replayed"):
            project(a)
        with self.assertRaisesRegex(ValueError, "reset required"):
            project(a, index=2)
        reset(a)
        with self.assertRaisesRegex(ValueError, "replayed"):
            project(a, index=2)
        reset(a)
        a.invalidate()
        with self.assertRaisesRegex(ValueError, "reset required"):
            project(a)

    def test_envelope_and_clocks(self):
        for snap, pause, error in (
            (replace(sample(), time_s=1.), 21.04, "clocks"),
            (replace(sample(), time_s=.5), 21., "clocks"),
            (replace(sample(), time_s=1., source_terminal_kw=-1540.), 21., "envelope"),
        ):
            a = adapter()
            reset(a)
            with self.assertRaisesRegex(ValueError, error):
                project(a, snap, pause=pause)
            with self.assertRaisesRegex(ValueError, "reset required"):
                project(a)

    def test_physical_failures(self):
        for edits in (
            dict(converged=False), dict(controls_settled=False),
            dict(phases_pu=PHASES[:-1]+(PHASES[0],)),
            dict(phases_pu=PHASES[:-1]+(("fake.1", 1.),)),
            dict(phases_pu=PHASES[:-1]+((PHASES[-1][0], 1.051),)),
            dict(windings=WINDINGS[:-1]+(replace(WINDINGS[-1], terminal_id="xfm1:1"),)),
            dict(windings=WINDINGS[:-1]+(replace(WINDINGS[-1], rating_kva=501.),)),
            dict(windings=WINDINGS[:-1]+(replace(WINDINGS[-1], node_current_amp=((1,1.), (2,1.), (3,1.))),)),
            dict(windings=WINDINGS[:-1]+(replace(WINDINGS[-1], measured_kva=501.),)),
            dict(binding=replace(PINNED_BINDING, active_profile_bindings=("duty",))),
            dict(binding=replace(PINNED_BINDING, active_der=("pv",))),
            dict(binding=replace(PINNED_BINDING, shunt=("890.1.2.3","delta",4.16,0.))),
            dict(taps=TAPS[:-1]+(("reg2c", 1.0125),)), dict(residual_kw=.02),
            dict(remaining_source_terminal_kva=.01), dict(source_terminal_kw=float("nan")),
        ):
            a = adapter()
            reset(a)
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                project(a, replace(sample(), time_s=1., **edits))
