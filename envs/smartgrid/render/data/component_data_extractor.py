"""可视化直接读取公共物理层，禁止由旧OOP缺失属性伪造设备状态。"""

from typing import Any, Dict, List
import math
import numpy as np


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


def get_component_summary(components: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """生成组件状态汇总

    Args:
            components: extract_all_components 的返回值

    Returns:
            汇总统计字典
    """
    caps = components.get("capacitors", {})
    regs = components.get("regulators", {})
    bats = components.get("batteries", {})
    pvs = components.get("pvs", {})

    cap_on = sum(1 for c in caps.values() if c.get("status") == 1)
    bat_socs = [b["soc"] for b in bats.values() if "soc" in b]
    pv_powers = [p["power_ratio"] for p in pvs.values() if "power_ratio" in p]
    reg_taps = [r["tap"] for r in regs.values() if "tap" in r]

    return {
        "n_capacitors": len(caps),
        "n_caps_on": cap_on,
        "n_regulators": len(regs),
        "avg_tap": float(np.mean(reg_taps)) if reg_taps else 0.0,
        "n_batteries": len(bats),
        "avg_soc": float(np.mean(bat_socs)) if bat_socs else 0.0,
        "n_pvs": len(pvs),
        "n_svgs": len(components.get("svgs", {})),
        "avg_pv_power": float(np.mean(pv_powers)) if pv_powers else 0.0,
    }


def get_device_actions_from_snapshot(
    snapshot: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """从快照中解析动作语义

    Args:
            snapshot: 单步快照字典

    Returns:
            {device_type: [action_values]}
    """
    devices = snapshot.get("devices", {})
    result: Dict[str, List[Any]] = {
        "capacitors": [],
        "regulators": [],
        "batteries": [],
        "pvs": [],
    }

    for name, dev in devices.get("capacitors", {}).items():
        result["capacitors"].append(
            {
                "name": name,
                "status": dev.get("status", 0),
            }
        )

    for name, dev in devices.get("regulators", {}).items():
        result["regulators"].append(
            {
                "name": name,
                "tap": dev.get("tap", 16),
            }
        )

    for name, dev in devices.get("batteries", {}).items():
        result["batteries"].append(
            {
                "name": name,
                "soc": dev.get("soc", 0.0),
                "power_kw": dev.get("power_kw", 0.0),
            }
        )

    for name, dev in devices.get("pvs", {}).items():
        result["pvs"].append(
            {
                "name": name,
                "power_ratio": dev.get("power_ratio", 0.0),
                "power_factor": dev.get("power_factor", 1.0),
            }
        )

    return result
