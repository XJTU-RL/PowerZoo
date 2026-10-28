"""同步运行任务的真实物理快照；数值/单位与训练info共用同一状态。"""

from typing import Any, Dict, List
import math
import numpy as np

_LENGTH_TO_KM = {
    1: 1.609344,
    2: 0.3048,
    3: 1.0,
    4: 0.001,
    5: 0.0003048,
    6: 0.0000254,
    7: 0.00001,
    8: 0.000001,
}


def extract_bus_data(env):
    result = {}
    for name, phases in env.physical_state.voltages_pu.items():
        if not phases:
            result[name] = {
                "name": name,
                "active_in_solution": False,
                "phase_nodes": [],
                "v_mag_pu": [],
                "v_angle_deg": [],
                "v_mag_kv": [],
                "n_phases": 0,
                "base_kv": None,
                "coord_defined": False,
                "x": None,
                "y": None,
            }
            continue
        env.circuit.native.SetActiveBus(name)
        bus = env.circuit.native.ActiveBus
        nodes = list(bus.Nodes)
        raw = list(bus.puVmagAngle)
        defined = bool(bus.Coorddefined)
        result[name] = {
            "name": name,
            "active_in_solution": True,
            "v_mag_pu": list(phases.values()),
            "phase_nodes": list(phases),
            "v_angle_deg": [
                float(raw[2 * i + 1]) for i, n in enumerate(nodes) if n in phases
            ],
            "v_mag_kv": [v * float(bus.kVBase) for v in phases.values()],
            "n_phases": len(phases),
            "base_kv": float(bus.kVBase),
            "coord_defined": defined,
            "x": float(bus.x) if defined else None,
            "y": float(bus.y) if defined else None,
        }
    return result


def extract_line_data(env):
    result = {}
    for element, info in env.physical_state.branches.items():
        if not element.lower().startswith("line."):
            continue
        name = element.split(".", 1)[1]
        lines = env.circuit.native.Lines
        lines.Name = name
        length, units = float(lines.Length), int(lines.Units)
        result[name] = {
            "name": name,
            "type": "line",
            "enabled": info["enabled"],
            "open": info["open"],
            "bus1": info["buses"][0].split(".")[0],
            "bus2": info["buses"][1].split(".")[0],
            "length_native": length,
            "length_units": units,
            "length_km": (
                length * _LENGTH_TO_KM[units] if units in _LENGTH_TO_KM else None
            ),
            "phases": info["phases"],
            "loss_kw": info["losses"][0],
            "loss_kvar": info["losses"][1],
            "loading_pct": (
                info["loading_pu"] * 100 if info["loading_pu"] is not None else None
            ),
        }
    return result


def extract_transformer_data(env):
    return {
        element.split(".", 1)[1]: {
            "name": element.split(".", 1)[1],
            "type": "transformer",
            "enabled": info["enabled"],
            "open": info["open"],
            "bus1": info["buses"][0].split(".")[0],
            "bus2": info["buses"][1].split(".")[0],
            "kva": info["winding_kva"][0],
            "loss_kw": info["losses"][0],
            "phases": info["phases"],
            "loading_pct": info["loading_pu"] * 100,
        }
        for element, info in env.physical_state.branches.items()
        if element.lower().startswith("transformer.")
    }


def extract_edge_list(env):
    return [
        (v["bus1"], v["bus2"], v["type"])
        for v in {**extract_line_data(env), **extract_transformer_data(env)}.values()
    ]


def _connection(circuit, element):
    info = circuit.element_info(element)
    return {"bus": info["buses"][0].split(".")[0].lower(), "n_phases": info["phases"]}


def extract_capacitor_data(env):
    c = env.circuit
    result = {}
    for name in c.capacitors:
        state = int(c.capacitor_state(name)[0])
        kvar = float(c.native.Capacitors.kvar)
        result[name] = {
            "name": name,
            "type": "capacitor",
            "status": state,
            "kvar_rated": kvar,
            **_connection(c, "Capacitor." + name),
        }
    return result


def extract_regulator_data(env):
    c = env.circuit
    result = {}
    for name, reg in c.regulators.items():
        info = c.element_info("Transformer." + reg.transformer)
        tap = c.regulator_tap(name)
        result[name] = {
            "name": name,
            "type": "regulator",
            "tap": tap,
            "tap_position": round(
                (tap - reg.mintap) / (reg.maxtap - reg.mintap) * reg.numtaps
            ),
            "bus1": info["buses"][0].split(".")[0],
            "bus2": info["buses"][1].split(".")[0],
            "min_tap": reg.mintap,
            "max_tap": reg.maxtap,
            "n_phases": info["phases"],
        }
    return result


def extract_battery_data(env):
    c = env.circuit
    result = {}
    for name, bat in c.batteries.items():
        p, _ = -c.powers("Storage." + name)
        result[name] = {
            "name": name,
            "type": "battery",
            "soc": bat.soc,
            "power_kw": float(p),
            "power_ratio": float(p / bat.max_kw),
            "max_kw": bat.max_kw,
            "max_kwh": bat.max_kwh,
            **_connection(c, "Storage." + name),
        }
    return result


def extract_pv_data(env):
    c = env.circuit
    result = {}
    for name, pv in c.pvs.items():
        p, q = -c.powers("PVSystem." + name)
        apparent = math.hypot(p, q)
        result[name] = {
            "name": name,
            "type": "pv",
            "power_kw": float(p),
            "reactive_power_kvar": float(q),
            "power_ratio": float(p / pv.pmpp),
            "power_factor": abs(float(p)) / apparent if apparent else 0.0,
            "rated_kw": pv.pmpp,
            "kva": pv.kva,
            **_connection(c, "PVSystem." + name),
        }
    return result


def extract_svg_data(env):
    c = env.circuit
    result = {}
    for name, svg in c.svgs.items():
        p, q = -c.powers("Generator." + name)
        result[name] = {
            "name": name,
            "type": "svg",
            "power_kw": float(p),
            "reactive_power_kvar": float(q),
            "kva": svg.kva,
            "rated_current_a": svg.rated_current_a,
            **_connection(c, "Generator." + name),
        }
    return result


def extract_all_components(env):
    return {
        "capacitors": extract_capacitor_data(env),
        "regulators": extract_regulator_data(env),
        "batteries": extract_battery_data(env),
        "pvs": extract_pv_data(env),
        "svgs": extract_svg_data(env),
    }


def extract_circuit_summary(env):
    state = env.physical_state
    injections = state.injections
    loads = [v for e, v in injections.items() if e.lower().startswith("load.")]
    generation = [v for e, v in injections.items() if not e.lower().startswith("load.")]
    volts = [v for phases in state.voltages_pu.values() for v in phases.values()]
    return {
        "converged": True,
        "time_s": state.time_s,
        "step_seconds": env.circuit.step_seconds,
        "total_loss_kw": state.losses[0],
        "total_loss_kvar": state.losses[1],
        "total_load_kw": -sum(v[0] for v in loads),
        "total_load_kvar": -sum(v[1] for v in loads),
        "grid_import_kw": state.grid_import[0],
        "grid_import_kvar": state.grid_import[1],
        "total_gen_kw": sum(v[0] for v in generation),
        "total_gen_kvar": sum(v[1] for v in generation),
        "total_pv_kw": sum(
            v[0] for e, v in injections.items() if e.lower().startswith("pvsystem.")
        ),
        "total_storage_kw": sum(
            v[0] for e, v in injections.items() if e.lower().startswith("storage.")
        ),
        "total_svg_kvar": sum(
            v[1]
            for e, v in injections.items()
            if e.lower().startswith("generator.")
            and e.split(".", 1)[1] in env.circuit.svgs
        ),
        "power_loss_pct": state.losses[0] / env.base_load_kw * 100,
        "v_mean_pu": float(np.mean(volts)) if volts else None,
        "v_min_pu": min(volts) if volts else None,
        "v_max_pu": max(volts) if volts else None,
        "active_balance_error_kw": state.active_balance_error_kw,
    }


def assemble_snapshot(env, step, actions=None, rewards=None, infos=None):
    env.circuit.read_state()
    snapshot = {
        "step": step,
        "buses": extract_bus_data(env),
        "lines": {**extract_line_data(env), **extract_transformer_data(env)},
        "devices": extract_all_components(env),
        "circuit": extract_circuit_summary(env),
    }
    if actions is not None:
        snapshot["actions"] = np.asarray(actions).tolist()
    if rewards is not None:
        snapshot["rewards"] = np.asarray(rewards).tolist()
    if infos is not None:
        while isinstance(infos, (list, tuple, np.ndarray)) and len(infos):
            infos = infos[0]
        if not isinstance(infos, dict):
            raise ValueError("快照缺少真实info")
        snapshot["info"] = infos
    return snapshot
