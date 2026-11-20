"""IEEE34 LFC raw and separately scoped no-DER OpenDSS snapshot diagnostic."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


import shutil
import tempfile
from types import SimpleNamespace

STANDARD = Path("node_systems/34Bus")
BASE = Path("envs/lfc/IEEE9BusSystem/systems/34Bus")
DECK_BLOB = "d76a7703c4fc803b03864a4d8b8e136e1e786fa3"
CORE = Path("envs/lfc/IEEE9BusSystem")
PINS = {"7a2c5f44cf9e662e0d6a6f6649ed0ec8c4b35e62",
        "de8ce3c7e08306f526ee4fb6f0fc8499fcbd33af"}

def require_report_repo(repo):
    actual = repo.resolve()
    expected = Path(__file__).resolve().parents[1]
    if actual != expected:
        raise ValueError("report repository must be the checkout containing this script")
    return actual


def require_helper_origin(module, expected):
    origin = getattr(module, "__file__", None)
    if origin is None or Path(origin).resolve() != expected.resolve():
        raise RuntimeError("LFC helper import origin does not match the pinned report checkout")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

def phases_and_terminals(elt):
    nt, nc = int(elt.NumTerminals), int(elt.NumConductors)
    p = np.asarray(elt.Powers)
    if np.iscomplexobj(p):
        p = p.astype(complex)
    else:
        p = p.reshape(-1, 2)
        p = p[:, 0] + 1j * p[:, 1]
    p = p.reshape(nt, nc)
    nodes = np.asarray(elt.NodeOrder).reshape(nt, nc)
    current_amp = np.asarray(elt.CurrentsMagAng).reshape(nt, nc, 2)[..., 0]
    return [{
        "bus": str(bus), "p_kw": float(row.real.sum()), "q_kvar": float(row.imag.sum()),
        "s_kva": float(abs(row.sum())),
        "sum_conductor_s_kva": float(np.abs(row).sum()),
        "conductors": [{"node": int(n), "p_kw": float(x.real), "q_kvar": float(x.imag),
                        "terminal_current_amp": float(i)}
                       for n, x, i in zip(ns, row, amps)]
    } for bus, row, ns, amps in zip(elt.BusNames, p, nodes, current_amp)]

def sample(io):
    ckt = io.dss.ActiveCircuit
    if not ckt.Solution.Converged:
        return {"converged": False}
    dev = {}
    all_terminals = []
    enabled_elements = []
    subxf = None
    for name in ckt.AllElementNames:
        if ckt.SetActiveElement(name) < 0:
            raise RuntimeError(f"cannot activate {name}")
        elt = ckt.ActiveCktElement
        if not elt.Enabled:
            continue
        k = name.lower()
        enabled_elements.append(k)
        terminals = phases_and_terminals(elt)
        all_terminals.extend((z["p_kw"], z["q_kvar"]) for z in terminals)
        if k.startswith(("vsource.", "generator.", "pvsystem.", "storage.", "load.", "capacitor.")):
            dev[k] = terminals
        if k == "transformer.subxf":
            subxf = terminals
    source = dev["vsource.source"][0]
    if len(dev["vsource.source"]) > 1 and any(
        math.hypot(z["p_kw"], z["q_kvar"]) > 1e-6
        for z in dev["vsource.source"][1:]
    ):
        raise RuntimeError("nonzero additional source terminal; PCC convention needs review")
    import_p, import_q = -source["p_kw"], -source["q_kvar"]
    if subxf is None:
        raise RuntimeError("missing enabled SubXF")
    tx = ckt.Transformers
    tx.Name = "SubXF"
    if int(tx.NumWindings) != len(subxf):
        raise RuntimeError("winding-to-terminal count mismatch")
    for winding, entry in enumerate(subxf, 1):
        tx.Wdg = winding
        kva = float(tx.kVA)
        if not math.isfinite(kva) or kva <= 0:
            raise RuntimeError("invalid winding kVA rating")
        entry["rating_kva"] = kva
        entry["rating_kv_ll"] = float(tx.kV)
        entry["nominal_terminal_line_current_amp"] = kva / (math.sqrt(3) * entry["rating_kv_ll"])
        entry["net_s_over_rated"] = entry["s_kva"] / kva
        entry["sum_conductor_s_over_rated"] = entry["sum_conductor_s_kva"] / kva
    vv = {}
    for bus in ckt.AllBusNames:
        active = ckt.Buses[bus]
        ph = np.asarray(active.Nodes)
        volts = np.asarray(active.puVmagAngle).reshape(-1, 2)
        if len(ph) != len(volts):
            raise RuntimeError(f"phase and voltage count mismatch at {bus}")
        for node, (v, _) in zip(ph, volts):
            if int(node) in (1, 2, 3):
                vv[f"{bus}.{node}"] = {"v_pu": float(v), "base_kv_ln": float(active.kVBase)}
    if not vv or any(z["base_kv_ln"] <= 0 for z in vv.values()):
        raise RuntimeError("missing usable voltage bases")
    batt = {}
    for name, b in io.batteries.items():
        k = "generator." + name.rsplit(".", 1)[-1].lower()
        batt[name] = {"initial_kwh": float(b.initial_soc * b.max_kwh),
                      "max_kwh": float(b.max_kwh), "initial_soc": float(b.initial_soc),
                      "observed_wrapper_soc": float(b.soc),
                      "observed_wrapper_kwh": float(b.kwh), "dss_terminals": dev[k]}
    if getattr(io, 'diagnostic_mode', 'raw') == 'raw' and len(batt) != 4:
        raise RuntimeError("battery inventory mismatch")
    resid_p, resid_q = (math.fsum(x) for x in zip(*all_terminals))
    # Global measured complex-power accounting (not a per-node KCL test).
    balance_tol = max(0.01, 1e-6 * max(abs(import_p), abs(import_q)))
    return {"converged": True, "batteries": batt,
            "pcc_import_kw": import_p, "pcc_import_kvar": import_q,
            "network_power_residual_kw": resid_p, "network_power_residual_kvar": resid_q,
            "network_power_balance_ok": max(abs(resid_p), abs(resid_q)) <= balance_tol,
            "network_power_balance_tolerance_kw_or_kvar": balance_tol,
            "enabled_electrical_elements_in_power_accounting": enabled_elements,
            "enabled_device_terminal_measurements": dev, "subxf_windings": subxf,
            "voltage_phase_pu": vv}

def checks(record, lo, hi):
    if not record["converged"]:
        return {"converged": False, "unscored": True}
    return {"converged": True, "power_balance_ok": record["network_power_balance_ok"],
            "invalid_soc_devices": [k for k, b in record["batteries"].items()
                                    if not (0 <= b["initial_soc"] <= 1 and
                                            0 <= b["observed_wrapper_soc"] <= 1)],
            "overrated_windings_net_s": [i for i, w in enumerate(record["subxf_windings"], 1)
                                       if w["net_s_over_rated"] > 1],
            "voltage_screen_band_pu": [lo, hi],
            "out_of_band_phase_nodes": [k for k, x in record["voltage_phase_pu"].items()
                                        if not lo <= x["v_pu"] <= hi]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("--mode", choices=("raw", "no-der", "standard-off", "standard-static"), required=True)
    p.add_argument("--standard-tolerance", type=float, help="optional native voltage convergence tolerance")
    p.add_argument("--vmin", type=float, required=True)
    p.add_argument("--vmax", type=float, required=True)
    a = p.parse_args()
    if not 0 < a.vmin < a.vmax < math.inf:
        p.error("require finite positive ordered voltage screen")
    if a.standard_tolerance is not None and (
        not a.mode.startswith("standard-") or not 0 < a.standard_tolerance < 1
    ):
        p.error("--standard-tolerance requires a standard mode and a finite value in (0, 1)")
    try:
        repo = require_report_repo(a.repo)
    except ValueError as exc:
        p.error(str(exc))
    head = git(repo, "rev-parse", "HEAD")
    runtime_files = [CORE / x for x in ("opendss_python.py", "loadprofile.py")]
    baseline_paths = ([STANDARD] if a.mode.startswith("standard-") else [BASE, *runtime_files])
    if git(repo, "status", "--porcelain", "--ignored", "--", *map(str, baseline_paths)):
        raise RuntimeError("dirty or generated raw inputs in repository")
    if not any(all(git(repo, "rev-parse", f"{head}:{path}") ==
                   git(repo, "rev-parse", f"{pin}:{path}") for path in baseline_paths)
               for pin in sorted(PINS)):
        raise RuntimeError("unexpected source input tree")

    deck = repo / BASE / "ieee34Mod1_duty.dss"
    if not a.mode.startswith("standard-") and git(repo, "hash-object", str(deck)) != DECK_BLOB:
        raise RuntimeError("unexpected raw deck blob")
    with tempfile.TemporaryDirectory(prefix="powerzoo-lfc34-") as td:
        work = Path(td) / "deck"
        report = (run_standard(repo, work, a) if a.mode.startswith("standard-") else
                  run_case(repo, deck, work, a))
    print(json.dumps(report, sort_keys=True, allow_nan=False))

def run_standard(repo, work, a):
    import dss

    source = repo / STANDARD
    shutil.copytree(source, work)
    context = dss.DSS.NewContext()
    context.Text.Command = f'Compile "{work / "network.dss"}"'
    native = context.ActiveCircuit
    disabled = sorted(n.lower() for n in native.AllElementNames if
                      n.lower().startswith(("generator.", "pvsystem.", "storage.")))
    if disabled != ["storage.batt1", "storage.batt2"]:
        raise RuntimeError("unexpected standard-deck DER inventory")
    for name in disabled:
        context.Text.Command = f"Edit {name} Enabled=no"
        native.SetActiveElement(name)
        if native.ActiveCktElement.Enabled:
            raise RuntimeError(f"failed to disable {name}")
    context.Text.Command = "Set mode=snapshot"
    context.Text.Command = "Set Maxiterations=100"
    context.Text.Command = "Set Maxcontroliter=100"
    control_mode = a.mode.removeprefix("standard-").upper()
    context.Text.Command = f"Set ControlMode={control_mode}"
    if a.standard_tolerance is not None:
        native.Solution.Tolerance = a.standard_tolerance
    solver_tolerance = float(native.Solution.Tolerance)
    if not math.isfinite(solver_tolerance) or solver_tolerance <= 0:
        raise RuntimeError("invalid native solver tolerance")
    if control_mode == "OFF":
        native.Solution.SolveNoControl()
    else:
        native.Solution.Solve()
    record = sample(SimpleNamespace(dss=context, batteries={}, diagnostic_mode="standard"))
    taps = {}
    if record["converged"]:
        for name in ("reg1a", "reg1b", "reg1c", "reg2a", "reg2b", "reg2c"):
            native.Transformers.Name = name
            native.Transformers.Wdg = 2
            taps[name] = float(native.Transformers.Tap)
    return {"source_commit": git(repo, "rev-parse", "HEAD"),
            "input_tree": git(repo, "rev-parse", f"HEAD:{STANDARD}"),
            "dss_python_version": dss.__version__, "mode": a.mode,
            "control_mode": control_mode, "disabled_der": disabled,
            "native_voltage_convergence_tolerance": solver_tolerance,
            "native_control_actions_done": (bool(native.Solution.ControlActionsDone)
                                            if control_mode == "STATIC" else None),
            "regulator_secondary_taps": taps,
            "stages": {"standard_snapshot": record},
            "checks": {"standard_snapshot": checks(record, a.vmin, a.vmax)}}


def run_case(repo, deck, work, a):
    import dss
    from envs.lfc.IEEE9BusSystem import loadprofile, opendss_python

    require_helper_origin(loadprofile, repo / CORE / "loadprofile.py")
    require_helper_origin(opendss_python, repo / CORE / "opendss_python.py")
    if opendss_python.LoadProfile is not loadprofile.LoadProfile:
        raise RuntimeError("legacy circuit imported an unexpected LoadProfile")
    LoadProfile = loadprofile.LoadProfile
    circuit = opendss_python.circuit

    head = git(repo, "rev-parse", "HEAD")
    inputs = {str(p.relative_to(deck.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(deck.parent.rglob("*")) if p.is_file()}
    shutil.copytree(deck.parent, work)
    worker = work / "ieee34Mod1_duty_0.dss"
    worker.write_text("".join(
        ("redirect loadshape_0.dss\n" if line.strip() == "redirect loadshape.dss" else
         "redirect irrad_up_down_0.dss\n" if line.strip() == "redirect irrad_up_down.dss" else line)
        for line in deck.read_text().splitlines(keepends=True)))
    lp = LoadProfile(30, str(work), worker.name, worker_idx=0)
    n_profiles = lp.gen_loadprofile()
    lp.choose_loadprofile(0)
    if a.mode == "raw":
        io = circuit(str(worker), dss_act=False)
        stages = {"constructor": sample(io)}
        lp.choose_loadprofile(0)
        lp.choose_irrad_profile("Irrad_constant")
        io.reset()
        stages["reset"] = sample(io)
        disabled = []
    else:
        # A separate structural no-DER counterfactual, not the original zero-command arm.
        lp.choose_irrad_profile("Irrad_constant")
        context = dss.DSS.NewContext()
        context.Text.Command = f'compile "{worker}"'
        context.Text.Command = "Set Maxiterations=50"
        context.Text.Command = "Set Maxcontroliter=100"
        context.Text.Command = "Set ControlMode=off"
        native = context.ActiveCircuit
        disabled = sorted(n.lower() for n in native.AllElementNames if
                          n.lower().startswith(("pvsystem.", "generator.", "storage.")))
        expected_battery = {f"generator.batt{i}" for i in range(1, 5)}
        if {n for n in disabled if not n.startswith("pvsystem.")} != expected_battery or len(disabled) != 20:
            raise RuntimeError("unexpected no-DER device inventory")
        for name in disabled:
            context.Text.Command = f"Edit {name} Enabled=no"
            native.SetActiveElement(name)
            if native.ActiveCktElement.Enabled:
                raise RuntimeError(f"failed to disable {name}")
        native.Solution.SolveNoControl()
        stages = {"no_der_nominal": sample(SimpleNamespace(
            dss=context, batteries={}, diagnostic_mode="no-der"))}
    return {"source_commit": head, "deck_blob": DECK_BLOB, "mode": a.mode,
            "dss_python_version": dss.__version__, "num_30_step_profiles": n_profiles,
            "initialization": "official rank0 copy + LoadProfile", "input_sha256": inputs,
            "disabled_der": disabled,
            "configuration": {"dss_act": False, "rated_values": "legacy research deck; not field ratings",
                              "voltage_screen_not_a_standard_claim": True,
                              "pcc_background_equivalence": "not assumed or applied"},
            "stages": stages, "checks": {key: checks(value, a.vmin, a.vmax)
                                         for key, value in stages.items()}}


if __name__ == "__main__":
    main()
