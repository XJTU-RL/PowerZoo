"""系统汇总使用公共物理状态；主网进出口与DER发电分开，不对流向取绝对值。"""

from envs.powergrid.snapshot import extract_circuit_summary

from typing import Any, Dict, List
import numpy as np


def extract_circuit_timeseries(
    snapshots: List[Dict[str, Any]],
    metric_key: str,
) -> List[float]:
    """从快照序列中提取电路级时间序列

    Args:
            snapshots: 快照列表
            metric_key: 要提取的指标键名

    Returns:
            时间序列列表
    """
    series: List[float] = []
    for snap in snapshots:
        circuit = snap.get("circuit", {})
        val = circuit.get(metric_key, 0.0)
        series.append(float(val) if isinstance(val, (int, float)) else 0.0)
    return series


def compute_episode_circuit_summary(
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """计算整个 episode 的电路汇总指标

    Args:
            snapshots: 快照列表

    Returns:
            episode 级别汇总字典
    """
    snapshots = [s for s in snapshots if s.get("step", 1) > 0]
    if not snapshots:
        return {}

    loss_kw_series = extract_circuit_timeseries(snapshots, "total_loss_kw")
    load_kw_series = extract_circuit_timeseries(snapshots, "total_load_kw")
    v_mean_series = extract_circuit_timeseries(snapshots, "v_mean_pu")
    v_min_series = extract_circuit_timeseries(snapshots, "v_min_pu")
    v_max_series = extract_circuit_timeseries(snapshots, "v_max_pu")

    return {
        "avg_loss_kw": float(np.mean(loss_kw_series)) if loss_kw_series else 0.0,
        "max_loss_kw": float(np.max(loss_kw_series)) if loss_kw_series else 0.0,
        "avg_load_kw": float(np.mean(load_kw_series)) if load_kw_series else 0.0,
        "avg_voltage_pu": float(np.mean(v_mean_series)) if v_mean_series else 0.0,
        "min_voltage_pu": float(np.min(v_min_series)) if v_min_series else 0.0,
        "max_voltage_pu": float(np.max(v_max_series)) if v_max_series else 0.0,
        "total_loss_kwh": sum(
            s["circuit"]["total_loss_kw"] * s["circuit"]["step_seconds"] / 3600
            for s in snapshots
            if s.get("step", 1) > 0
        ),
        "total_load_kwh": sum(
            s["circuit"]["total_load_kw"] * s["circuit"]["step_seconds"] / 3600
            for s in snapshots
            if s.get("step", 1) > 0
        ),
        "n_steps": len(snapshots),
    }
