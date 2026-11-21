"""Draft OpenDSS-only input collector. No Bus8/engine writes."""
import argparse
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile
import os
import tempfile
from types import SimpleNamespace

from envs.lfc.diagnostics import ieee34_bus8 as anchor_module
from scripts import check_lfc34_operating_point as operating_module
from envs.lfc.diagnostics.ieee34_bus8 import (
    ObservedReference, Binding, DECK_TREE, MAX_DELTA_KW_KVAR, PINNED_BINDING, RATINGS, REGULATORS,
    STATIC_COMMIT, Snapshot, Winding,
)
from scripts.check_lfc34_operating_point import (
    STANDARD, collect_shunt_evidence, git, phases_and_terminals,
    sample,
)

ARMS = {"hold": (0., 0.), "p_import": (20., 0.),
        "q_import": (0., 15.), "q_export": (0., -15.)}
PULSE = "Load.CouplingProbe890"


def verify_checkout(repo, cwd, script_file=__file__,
                    anchor=anchor_module, operating=operating_module):
    """Do not delegate provenance checks to a possibly shadowed import."""
    expected = Path(script_file).resolve().parents[1]
    if Path(repo).resolve() != expected or Path(cwd).resolve() != expected:
        raise ValueError("collector must run from its own resolved checkout")
    for mod, relative in (
        (anchor, "envs/lfc/diagnostics/ieee34_bus8.py"),
        (operating, "scripts/check_lfc34_operating_point.py"),
    ):
        origin = getattr(mod, "__file__", None)
        if origin is None or Path(origin).resolve() != (expected / relative).resolve():
            raise ValueError(f"helper import origin mismatch: {relative}")
    return expected


def archived_deck(repo, scratch):
    if git(repo, "rev-parse", f"HEAD:{STANDARD}") != DECK_TREE:
        raise ValueError("IEEE34 deck tree changed")
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", "HEAD", str(STANDARD)])
    with tarfile.open(fileobj=io.BytesIO(archive)) as z:
        for m in z.getmembers():
            relative = Path(m.name)
            if relative.is_absolute() or ".." in relative.parts or not (m.isdir() or m.isfile()):
                raise ValueError("unexpected archived deck entry")
        z.extractall(scratch)
    return scratch / STANDARD / "network.dss"


def _active(circuit, name):
    if circuit.SetActiveElement(name) < 0 or not circuit.ActiveCktElement.Enabled:
        raise RuntimeError(f"missing or disabled {name}")
    return circuit.ActiveCktElement


def _prop(circuit, name, key):
    return _active(circuit, name).Properties(key).Val.strip()


def _phase_taps(c):
    result = []
    for name in REGULATORS:
        _active(c, "Transformer." + name)
        c.Transformers.Name = name
        c.Transformers.Wdg = 2
        result.append((name, float(c.Transformers.Tap)))
    return tuple(result)


def _void_shape(value):
    return value.strip().lower() in ("", "none")


def _der_profile_inventory(c):
    active_der, profiles, disabled = [], [], []
    for name in c.AllElementNames:
        if c.SetActiveElement(name) < 0:
            raise RuntimeError(f"cannot activate {name}")
        element = c.ActiveCktElement
        key = name.lower()
        if key.startswith(("generator.", "pvsystem.", "storage.")):
            (active_der if element.Enabled else disabled).append(key)
        if not key.startswith("load.") or not element.Enabled:
            continue
        for prop in ("daily", "duty", "yearly"):
            value = element.Properties(prop).Val
            if not _void_shape(value):
                profiles.append((key, prop, value))
    return tuple(sorted(active_der)), tuple(sorted(disabled)), tuple(sorted(profiles))


def _single_stage_kvar(raw, num_steps, states, bank_kvar):
    """Validate one energized native capacitor stage and its two nominal readouts."""
    if num_steps != 1 or tuple(states) != (1,):
        raise ValueError("study shunt must have exactly one enabled native stage")
    number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    match = re.fullmatch(rf"(?:({number})|\[\s*({number})\s*\])", raw.strip())
    if not match:
        raise ValueError("expected scalar or one finite native capacitor kvar entry")
    kvar = float(match[1] or match[2])
    if not (math.isfinite(kvar) and kvar > 0 and math.isfinite(bank_kvar)
            and math.isclose(kvar, bank_kvar, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("study shunt nominal kvar readouts disagree")
    return kvar


def _native_binding(c, deck_tree, *, control_mode="STATIC"):
    s = c.Solution
    if (int(s.Mode), int(s.ControlMode)) != (0, {"STATIC": 0, "OFF": -1}[control_mode]):
        raise RuntimeError(f"expected native Snapshot/{control_mode} only")
    actual_der, disabled, shapes = _der_profile_inventory(c)
    shunt = "Capacitor.Study890"
    elt = _active(c, shunt)
    if (int(elt.NumPhases), int(elt.NumTerminals), int(elt.NumConductors)) != (3, 1, 3):
        raise RuntimeError("study shunt is not a three-phase one-terminal branch")
    bus = tuple(elt.BusNames)
    if bus != ("890.1.2.3",):
        raise RuntimeError("study shunt bus/conductors changed")
    conn = _prop(c, shunt, "conn").lower()
    kvar_property = _prop(c, shunt, "kvar")
    c.Capacitors.Name = shunt.split(".", 1)[1]
    if str(c.Capacitors.Name).lower() != "study890":
        raise RuntimeError("failed to bind the native study capacitor")
    kvar = _single_stage_kvar(kvar_property, c.Capacitors.NumSteps,
                             c.Capacitors.States, float(c.Capacitors.kvar))
    kv = float(_prop(c, shunt, "kV"))
    source = "Vsource.Source"
    if int(_active(c, source).NumTerminals) != 2:
        raise RuntimeError("expected exactly two source terminals")
    binding = Binding(STATIC_COMMIT, deck_tree, float(_prop(c, source, "pu")),
                      float(s.LoadMult), (bus[0], conn, kv, kvar),
                      float(s.Tolerance), control_mode, disabled, actual_der, shapes)
    if binding != replace(PINNED_BINDING, control_mode=control_mode):
        raise RuntimeError("measured runtime binding differs from pinned study")
    return binding


def _winding_records(c):
    result = []
    for terminal_id in RATINGS:
        xf, i = terminal_id.split(":")
        report = phases_and_terminals(_active(c, "Transformer." + xf))
        if len(report) != 2:
            raise RuntimeError("transformer terminal inventory changed")
        t = report[int(i)-1]
        c.Transformers.Name = xf
        c.Transformers.Wdg = int(i)
        current = tuple((z["node"], z["terminal_current_amp"]) for z in t["conductors"])
        result.append(Winding(terminal_id, t["bus"].split(".")[0].lower(),
                              float(c.Transformers.kVA), float(c.Transformers.kV),
                              t["s_kva"], current))
    return tuple(result)


def _pulse(c, demand):
    if demand is None:
        if c.SetActiveElement(PULSE) < 0 or c.ActiveCktElement.Enabled:
            raise RuntimeError("expected a disabled pulse load at recovery")
        return None
    if demand == (0., 0.):
        if PULSE.lower() in {n.lower() for n in c.AllElementNames}:
            raise RuntimeError("hold unexpectedly contains pulse load")
        return None
    e = _active(c, PULSE)
    if (list(e.BusNames), int(e.NumPhases), int(e.NumTerminals), int(e.NumConductors)) != (
        ["890.1.2.3"], 3, 1, 3
    ):
        raise RuntimeError("pulse load is not exactly 890 ABC delta")
    if _prop(c, PULSE, "conn").lower() != "delta" or int(_prop(c, PULSE, "model")) != 1:
        raise RuntimeError("pulse load is not Model=1 delta")
    if _prop(c, PULSE, "status").lower() != "fixed":
        raise RuntimeError("pulse load must be fixed against load multiplier")
    if not math.isclose(float(_prop(c, PULSE, "kV")), 4.16, abs_tol=1e-10):
        raise RuntimeError("wrong pulse nameplate voltage")
    terminal, = phases_and_terminals(_active(c, PULSE))
    measured = terminal["p_kw"], terminal["q_kvar"]
    if not all(math.isfinite(m) and abs(m-d) <= .05 for m, d in zip(measured, demand)):
        raise RuntimeError(f"pulse P/Q is not delivered within 0.05: {measured}")
    return measured


def _configure(context, deck, demand):
    command = context.Text
    command.Command = f'Compile "{deck}"'
    c = context.ActiveCircuit
    active, disabled, _ = _der_profile_inventory(c)
    declared = PINNED_BINDING.original_disabled_der
    if set(active + disabled) != set(declared):
        raise RuntimeError("uncalibrated DER inventory")
    for name in declared:
        command.Command = f"Edit {name} Enabled=no"
    # Replay the published bounded setup without touching its existing implementation.
    for setting in ("mode=snapshot", "LoadMult=0.75", "Maxiterations=100",
                    "Maxcontroliter=100", "ControlMode=STATIC", "Tolerance=1e-9"):
        command.Command = "Set " + setting
    command.Command = "Edit Vsource.Source pu=1.045"
    command.Command = ("New Capacitor.Study890 Bus1=890.1.2.3 Phases=3 "
                       "Conn=Delta kV=4.16 kvar=75")
    if demand != (0., 0.):
        p, q = demand
        command.Command = (f"New {PULSE} Bus1=890.1.2.3 Phases=3 Conn=Delta kV=4.16 "
                           f"Model=1 Status=Fixed kW={p} kvar={q}")
    c.Solution.Solve()
    return c


def _capture(context, c, demand, index, *, control_mode="STATIC", actual_clock_s=None):
    if control_mode == "OFF" and actual_clock_s is None:
        raise ValueError("OFF capture requires measured native clock")
    r = sample(SimpleNamespace(dss=context, batteries={}, diagnostic_mode="standard"))
    if not r["converged"] or (control_mode == "STATIC" and not c.Solution.ControlActionsDone):
        raise RuntimeError("solve or STATIC controls incomplete")
    source = phases_and_terminals(_active(c, "Vsource.Source"))
    a, *additional = source
    nominal = _native_binding(c, DECK_TREE, control_mode=control_mode)
    actual_pulse = _pulse(c, demand)
    evidence = collect_shunt_evidence(c, 75.)
    shunt = evidence["measured_shunt_injection_kvar"]
    if not math.isfinite(shunt) or not 0 < shunt <= 75 * 1.05**2:
        raise RuntimeError("unexpected shunt injection")
    snap = Snapshot(
        nominal, float(index) if control_mode == "STATIC" else actual_clock_s, a["p_kw"], a["q_kvar"],
        max((s["s_kva"] for s in additional), default=0.),
        tuple((n, v["v_pu"]) for n, v in r["voltage_phase_pu"].items()),
        _winding_records(c), _phase_taps(c),
        r["network_power_residual_kw"], r["network_power_residual_kvar"], True,
        True if control_mode == "STATIC" else None,
    )
    return snap, {
        "pulse_signed_kw_kvar": actual_pulse, "shunt_injected_kvar": shunt,
        "total_losses_kw_kvar": (evidence["feeder_loss_kw"], evidence["feeder_loss_kvar"]),
        "l32_terminal_flows": evidence["equipment"]["Line.L32"]["terminal_flows"],
        "l32_rating_status": "currents measured; no sourced rating, no clearance claim",
    }


def run_arm(deck, demand, index):
    import dss

    cwd = Path.cwd()
    context = dss.DSS.NewContext()
    try:
        context.AllowChangeDir = False
        c = _configure(context, deck, demand)
        snap, evidence = _capture(context, c, demand, index)
        return snap, dict(evidence, dss_python_version=str(dss.__version__),
                          native_engine_version=str(context.Version))
    finally:
        context.ClearAll()
        if Path.cwd() != cwd:
            os.chdir(cwd)
            raise RuntimeError("native session changed working directory")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_report", type=Path, help="approved external native baseline JSON")
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="own checkout root")
    args = p.parse_args()
    repo = verify_checkout(args.repo, Path.cwd())
    if git(repo, "status", "--porcelain", "--untracked-files=all", "--", str(STANDARD)):
        raise ValueError("dirty IEEE34 source tree")
    report = args.reference_report.read_bytes()
    # OpenDSS-only. Never construct pretend receiving pauses or Bus8 command telemetry.
    reference = ObservedReference.read_report(report)
    results = {}
    with tempfile.TemporaryDirectory(prefix="lfc34-inputs-") as tmp:
        for index, (name, demand) in enumerate(ARMS.items()):
            try:
                scratch = Path(tmp) / name
                scratch.mkdir()
                deck = archived_deck(repo, scratch)
                snap, details = run_arm(deck, demand, index)
                power = snap.checked_import(reference)
                delta = tuple(a-b for a,b in zip(power, reference.feeder_import_kw_kvar))
                if name == "hold":
                    if any(abs(d) > 1e-6 for d in delta):
                        raise RuntimeError("hold point differs from accepted reference")
                else:
                    if any(abs(d) > cap for d, cap in zip(delta, MAX_DELTA_KW_KVAR)):
                        raise RuntimeError("measured PCC perturbation exceeds envelope")
                    for d, sign in zip(delta, demand):
                        if sign and d*sign <= 0:
                            raise RuntimeError("unexpected PCC import direction")
                results[name] = dict(passed=True, request_kw_kvar=demand,
                                     feeder_import_kw_kvar=power, feeder_delta_kw_kvar=delta,
                                     snapshot=asdict(snap), **details)
            except Exception as exc:
                results[name] = dict(passed=False, request_kw_kvar=demand,
                                     reason=str(exc), error_type=type(exc).__name__)
                if name == "hold":
                    break
    print(json.dumps({"scope": "OpenDSS-only", "reference_sha256": reference.sha256,
                      "source_checkout": git(repo, "rev-parse", "HEAD"),
                      "cases": results}, indent=2, allow_nan=False))
    return 0 if len(results) == len(ARMS) and all(x["passed"] for x in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
