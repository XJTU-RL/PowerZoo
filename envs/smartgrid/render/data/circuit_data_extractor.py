"""
SmartGrid Circuit Data Extractor
电路级汇总数据提取器

提取系统级功率、损耗、收敛状态等汇总指标。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_circuit_summary(env) -> Dict[str, Any]:
	"""提取电路级汇总数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		电路汇总字典
	"""
	summary: Dict[str, Any] = {
		"converged": True,
		"total_loss_kw": 0.0,
		"total_loss_kvar": 0.0,
		"total_load_kw": 0.0,
		"total_load_kvar": 0.0,
		"total_gen_kw": 0.0,
		"total_gen_kvar": 0.0,
		"total_pv_kw": 0.0,
		"total_storage_kw": 0.0,
		"power_loss_pct": 0.0,
		"v_mean_pu": 0.0,
		"v_min_pu": 0.0,
		"v_max_pu": 0.0,
	}

	try:
		circuit = env.circuit
		dss = circuit.dss

		# 收敛状态
		summary["converged"] = bool(dss.ActiveCircuit.Solution.Converged)

		# 损耗
		total_loss = circuit.total_loss()
		if len(total_loss) >= 2:
			summary["total_loss_kw"] = float(total_loss[0])
			summary["total_loss_kvar"] = float(total_loss[1])

		# 负荷
		total_load = circuit.total_load_power()
		if len(total_load) >= 2:
			summary["total_load_kw"] = float(total_load[0])
			summary["total_load_kvar"] = float(total_load[1])

		# 发电
		total_power = circuit.total_power()
		if len(total_power) >= 2:
			summary["total_gen_kw"] = abs(float(total_power[0]))
			summary["total_gen_kvar"] = abs(float(total_power[1]))

		# 损耗百分比
		summary["power_loss_pct"] = float(circuit.calculate_loss_percentage())

		# 电压统计
		v_stats = _compute_system_voltage_stats(env)
		summary.update(v_stats)

		# PV 总出力
		summary["total_pv_kw"] = _compute_total_pv_power(env)

		# 储能总出力
		summary["total_storage_kw"] = _compute_total_storage_power(env)

	except Exception as exc:
		logger.error(f"Circuit summary extraction failed: {exc}")

	return summary


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
		"total_loss_kwh": sum(loss_kw_series),
		"total_load_kwh": sum(load_kw_series),
		"n_steps": len(snapshots),
	}


def _compute_system_voltage_stats(env) -> Dict[str, float]:
	"""计算系统电压统计"""
	all_v: List[float] = []
	try:
		obs = env.obs
		bus_voltages = obs.get("bus_voltages", {})
		for mags in bus_voltages.values():
			if isinstance(mags, (list, tuple)):
				all_v.extend(float(v) for v in mags)
	except Exception:
		pass

	if not all_v:
		return {"v_mean_pu": 0.0, "v_min_pu": 0.0, "v_max_pu": 0.0}

	arr = np.array(all_v)
	return {
		"v_mean_pu": float(np.mean(arr)),
		"v_min_pu": float(np.min(arr)),
		"v_max_pu": float(np.max(arr)),
	}


def _compute_total_pv_power(env) -> float:
	"""计算 PV 总出力 (kW)"""
	total = 0.0
	try:
		pvs = getattr(env.circuit, "pvs", {})
		dss = env.circuit.dss
		for pv_name in pvs:
			dss.ActiveCircuit.SetActiveElement(f"PVSystem.{pv_name}")
			powers = dss.ActiveCircuit.ActiveElement.TotalPowers
			if len(powers) > 0:
				total += abs(float(powers[0]))
	except Exception:
		pass
	return total


def _compute_total_storage_power(env) -> float:
	"""计算储能总出力 (kW)"""
	total = 0.0
	try:
		for name, bat in env.circuit.batteries.items():
			if hasattr(bat, "actual_power"):
				total += abs(float(bat.actual_power()))
	except Exception:
		pass
	return total
