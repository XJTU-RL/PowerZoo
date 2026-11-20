"""Draft distinct seconds-scale tap-hold experiment; no Bus8 or MATLAB I/O."""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import tempfile

from scripts import check_lfc34_coupling_inputs as native
from scripts.check_lfc34_coupling_inputs import ARMS, PULSE
from envs.lfc.diagnostics.ieee34_bus8 import (
    MAX_DELTA_KW_KVAR, ObservedReference, PINNED_BINDING, REGULATORS,
)

HELD_BINDING = replace(PINNED_BINDING, control_mode="OFF")


def tap_differences(expected, observed):
    result = {}
    for name in sorted(expected.keys() | observed.keys()):
        before, after = expected.get(name), observed.get(name)
        if before != after:
            result[name] = {"expected": before, "observed": after}
    return result


def verify_taps(expected, observed, phase, audit):
    audit[phase] = observed
    mismatches = tap_differences(expected, observed)
    if mismatches:
        audit[phase + "_mismatches"] = mismatches
        raise RuntimeError(f"named mechanical tap readback failed at {phase}")


def read_taps(c):
    tap_ratio = dict(native._phase_taps(c))
    ratio_index = {}
    expected = {"creg" + n.removeprefix("reg") for n in REGULATORS}
    controls = c.RegControls
    if {str(n).lower() for n in controls.AllNames} != expected:
        raise RuntimeError("native regulator control inventory differs")
    for name in REGULATORS:
        key = "creg" + name.removeprefix("reg")
        controls.Name = key
        if str(controls.Transformer).lower() != name or int(controls.Winding) != 2:
            raise RuntimeError(f"regulator {key} winding association differs")
        value = controls.TapNumber
        if not math.isfinite(value) or value != int(value):
            raise RuntimeError(f"noninteger mechanical tap index at {key}")
        ratio_index[name] = {"ratio": tap_ratio[name], "index": int(value)}
    return ratio_index


def apply_arm(context, c, demand):
    # One runtime-owned load only; the published feeder stays unchanged.
    active = {n.lower() for n in c.AllElementNames}
    if demand == (0., 0.):
        if PULSE.lower() in active:
            context.Text.Command = f"Edit {PULSE} Enabled=no"
        return
    p, q = demand
    if PULSE.lower() in active:
        context.Text.Command = f"Edit {PULSE} Enabled=yes kW={p} kvar={q}"
    else:
        context.Text.Command = (f"New {PULSE} Bus1=890.1.2.3 Phases=3 Conn=Delta "
                                f"kV=4.16 Model=1 Status=Fixed kW={p} kvar={q}")


def capture_held(context, c, demand, index, baseline):
    # Disabled pulse remains in the native registry after a recovery reset.
    # _capture's strict hold no-pulse check applies only before first addition.
    capture_request = demand if demand != (0., 0.) or index == 0 else None
    snap, details = native._capture(context, c, capture_request, index, control_mode="OFF")
    power = snap.checked_import(baseline, expected_binding=HELD_BINDING)
    delta = tuple(a-b for a,b in zip(power, baseline.feeder_import_kw_kvar))
    if demand == (0., 0.):
        if any(abs(x) > 1e-6 for x in delta):
            raise RuntimeError("held neutral did not recover exact nominal PCC")
    elif any(abs(x) > cap for x,cap in zip(delta,MAX_DELTA_KW_KVAR)) or any(
        d*b <= 0 for d,b in zip(delta,demand) if b
    ):
        raise RuntimeError("held PCC pulse failed sign or magnitude envelope")
    return dict(snapshot=asdict(snap), native_control_mode="OFF",
                controls_settled_scope="one nominal STATIC solve, not per-arm settling",
                feeder_import_kw_kvar=power,
                feeder_delta_kw_kvar=delta, **details)


def run(context, deck, baseline):
    c = native._configure(context, deck, (0., 0.))  # exactly one native STATIC settle
    nominal, _ = native._capture(context, c, (0., 0.), 0)
    settled = nominal.checked_import(baseline)
    if any(abs(a-b) > 1e-6 for a,b in zip(settled, baseline.feeder_import_kw_kvar)):
        raise RuntimeError("nominal STATIC point differs from pinned reference")
    start = read_taps(c)
    context.Text.Command = "Set ControlMode=OFF"
    outcome = {}
    for i, (name, demand) in enumerate(ARMS.items()):
        audit = {}
        record = dict(passed=False, request_kw_kvar=demand, tap_readbacks=audit)
        outcome[name] = record
        try:
            if int(c.Solution.ControlMode) != -1:
                raise RuntimeError("native controls are not OFF")
            verify_taps(start, read_taps(c), "before", audit)
            apply_arm(context, c, demand)
            c.Solution.SolveNoControl()
            verify_taps(start, read_taps(c), "after", audit)
            record.update(capture_held(context, c, demand, i, baseline))
            record["passed"] = True
        except Exception as exc:
            record.update(reason=str(exc), error_type=type(exc).__name__)
        # Recover from each non-hold arm before the next; stop if recovery fails.
        if name != "hold":
            try:
                apply_arm(context, c, (0., 0.))
                c.Solution.SolveNoControl()
                verify_taps(start, read_taps(c), "recovery", audit)
                record["neutral_recovery"] = capture_held(context, c, (0., 0.), i, baseline)
            except Exception as exc:
                record.update(passed=False, recovery_failure=str(exc))
                break
        elif not record["passed"]:
            break
    return start, outcome


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_report", type=Path)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    native.verify_checkout(repo, Path.cwd())
    if Path(native.__file__).resolve() != repo / "scripts/check_lfc34_coupling_inputs.py":
        raise ValueError("wrong collector import origin")
    baseline = ObservedReference.read_report(args.reference_report.read_bytes())
    import dss

    with tempfile.TemporaryDirectory(prefix="lfc34-tap-hold-") as scratch:
        deck = native.archived_deck(repo, Path(scratch))
        c = dss.DSS.NewContext()
        c.AllowChangeDir = False
        try:
            start, arms = run(c, deck, baseline)
            result = dict(scope="distinct seconds-scale mechanical tap hold, snapshots only",
                          source_checkout=native.git(repo, "rev-parse", "HEAD"),
                          reference_sha256=baseline.sha256, dss_python_version=dss.__version__,
                          native_engine_version=c.Version, initial_named_taps=start, arms=arms)
            print(json.dumps(result, allow_nan=False, indent=2))
        finally:
            c.ClearAll()
            if Path.cwd() != repo:
                os.chdir(repo)
                raise RuntimeError("native engine changed cwd")
    return 0 if len(arms) == len(ARMS) and all(v["passed"] for v in arms.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
