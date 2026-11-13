"""
SmartGrid Bus Data Extractor
母线数据提取器

从 SmartGrid Circuit 对象中提取所有母线的电压、相位、
基准电压等数据，供可视化图表使用。
"""

from envs.powergrid.snapshot import extract_bus_data

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def compute_voltage_stats(
    bus_data: Dict[str, Dict[str, Any]],
    v_min_threshold: float = 0.95,
    v_max_threshold: float = 1.05,
) -> Dict[str, Any]:
    """计算电压统计指标

    Args:
            bus_data: 母线数据字典
            v_min_threshold: 电压下限阈值 (p.u.)
            v_max_threshold: 电压上限阈值 (p.u.)

    Returns:
            统计指标字典
    """
    all_voltages: List[float] = []
    violated_count = 0
    total_phases = 0

    for bus_name, info in bus_data.items():
        v_pu = info.get("v_mag_pu", [])
        for v in v_pu:
            all_voltages.append(v)
            total_phases += 1
            if v < v_min_threshold or v > v_max_threshold:
                violated_count += 1

    if not all_voltages:
        return {
            "v_mean_pu": 0.0,
            "v_min_pu": 0.0,
            "v_max_pu": 0.0,
            "v_std_pu": 0.0,
            "violation_count": 0,
            "violation_rate": 0.0,
            "total_phases": 0,
        }

    arr = np.array(all_voltages)
    return {
        "v_mean_pu": float(np.mean(arr)),
        "v_min_pu": float(np.min(arr)),
        "v_max_pu": float(np.max(arr)),
        "v_std_pu": float(np.std(arr)),
        "violation_count": violated_count,
        "violation_rate": violated_count / total_phases if total_phases > 0 else 0.0,
        "total_phases": total_phases,
    }
