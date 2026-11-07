"""可视化直接读取公共物理层，禁止由旧OOP缺失属性伪造设备状态。"""

from envs.powergrid.snapshot import (
    _connection,
    extract_capacitor_data,
    extract_regulator_data,
    extract_battery_data,
    extract_pv_data,
    extract_svg_data,
    extract_all_components,
)

from typing import Any, Dict, List
import math
import numpy as np


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
