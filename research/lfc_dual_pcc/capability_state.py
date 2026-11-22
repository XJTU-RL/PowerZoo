"""Recover a native DSS state by its historical action prefix and compare physical readbacks."""
import json
import math
from pathlib import Path

from scripts.check_lfc34_flexible_load import native
from .case_a_plant import CONFIG

PROTOCOL = Path(__file__).with_name("capability_protocol.json")


def largest(left, right):
    pairs = list(zip(left, right, strict=True))
    if not pairs or any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pairs):
        raise ValueError("incomplete or nonfinite readback")
    return max(abs(x-y) for x, y in pairs)


def compare_step(actual, recorded, tol):
    for key in ("action", "end_local_s"):
        if actual[key] != recorded[key]:
            raise ValueError("DSS action or clock changed")
    if actual["veto"] or recorded["veto"] or set(actual["voltage"]) != set(recorded["voltage"]):
        raise ValueError("invalid voltage census or unsafe action")
    differences = {
        "pq_kw_kvar": largest(actual["pcc"], recorded["pcc"]),
        "voltage_pu": largest(actual["voltage"].values(), [recorded["voltage"][k] for k in actual["voltage"]]),
        "soc_pu": max(largest(actual[k], recorded[k]) for k in ("soc", "soc_after")),
        "device_kw": largest([*actual["battery_kw"], actual["pv_kw"], actual["delivered_kw"]],
                             [*recorded["battery_kw"], recorded["pv_kw"], recorded["delivered_kw"]])
    }
    for key, difference in differences.items():
        if difference > tol[key]:
            raise ValueError(f"{key} mismatch: {difference} > {tol[key]}")
    return differences


def add_max(totals, differences):
    for key, value in differences.items():
        totals[key] = max(totals.get(key, 0.), value)


def restore(plant, site, frames, slot, tol):
    totals = {}
    for row in frames:
        if row["slot"] >= slot:
            break
        out = plant.execute(row["actions"][site])
        add_max(totals, compare_step(out, row["actual"][site], tol))
    prior = next(f for f in frames if f["slot"] == slot-1)["actual"][site]
    if (largest(prior["soc_after"], plant.soc()) > tol["soc_pu"]
            or plant.s != prior["end_local_s"] or plant.history[-1] != prior["action"]):
        raise ValueError("restored initial state, previous action or time differs")
    return totals


def exogenous(plant):
    # Fixed inputs and the model-profile bindings. Only current actuator settings change.
    c = plant.c
    loads = {}
    for name in sorted(c.Loads.AllNames):
        device = "Load." + name
        loads[name] = {k: native._prop(c, device, k) for k in
                       ("kW", "kvar", "model", "status", "daily", "yearly", "duty", "enabled")}
    pv = plant.cfg["pv"]["device"]
    return {"loads": loads, "solution_mode": int(c.Solution.Mode),
            "load_multiplier": float(c.Solution.LoadMult),
            "pv": {k:native._prop(c, pv, k) for k in ("irradiance","Pmpp","kVA","daily","yearly","duty")},
            "pv_base_kw": plant.cfg["pv"]["base_kw"], "fixed_taps": plant.taps}


def read_config():
    return json.loads(CONFIG.read_bytes()), json.loads(PROTOCOL.read_bytes())
