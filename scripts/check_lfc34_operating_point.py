"""IEEE34 native terminal, shunt and power-balance measurements."""
import math
from pathlib import Path
import subprocess

import numpy as np


STANDARD = Path("node_systems/34Bus")


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


def collect_shunt_evidence(native, nameplate):
    bus = native.Buses["890"]
    if list(bus.Nodes) != [1, 2, 3]:
        raise RuntimeError("study bus 890 no longer has exactly ordered ABC nodes")
    phasors = np.asarray(bus.Voltages)
    if not np.iscomplexobj(phasors):
        phasors = phasors.reshape(-1, 2) @ np.array([1, 1j])
    if len(phasors) != 3:
        raise RuntimeError("unexpected 890 voltage inventory")
    ll_kv = {f"{a}{b}": float(abs(phasors[a - 1] - phasors[b - 1]) / 1000)
             for a, b in ((1, 2), (2, 3), (3, 1))}
    equipment = {}
    expected = {"Line.L32": (3, 2, 3), "Transformer.XFM1": (3, 2, 4)}
    if nameplate:
        expected["Capacitor.Study890"] = (3, 1, 3)
    for name, inventory in expected.items():
        if native.SetActiveElement(name) < 0:
            raise RuntimeError(f"missing study equipment {name}")
        e = native.ActiveCktElement
        found = (int(e.NumPhases), int(e.NumTerminals), int(e.NumConductors))
        if found != inventory:
            raise RuntimeError(f"unexpected terminal/conductor inventory: {name}: {found}")
        equipment[name] = {"phases": found[0], "terminals": found[1],
                           "conductors_per_terminal": found[2],
                           "terminal_flows": phases_and_terminals(e)}
    xf = native.Transformers
    xf.Name = "XFM1"
    if int(xf.NumWindings) != 2:
        raise RuntimeError("unexpected positive-sequence XFM1 winding inventory")
    ratings = []
    for index in (1, 2):
        xf.Wdg = index
        ratings.append([float(xf.kVA), float(xf.kV)])
    equipment["Transformer.XFM1"]["native_winding_ratings"] = ratings
    cap = equipment.get("Capacitor.Study890")
    loss = np.asarray(native.Losses)
    loss = complex(loss.item()) if np.iscomplexobj(loss) else complex(*loss.reshape(2))
    return {"synthetic_shunt_nameplate_kvar": nameplate,
            "measured_shunt_injection_kvar": -cap["terminal_flows"][0]["q_kvar"] if cap else 0,
            "890_line_to_line_voltage_kv": ll_kv, "feeder_loss_kw": loss.real / 1000,
            "feeder_loss_kvar": loss.imag / 1000, "equipment": equipment}
