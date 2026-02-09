"""
SmartGrid Bus Data Extractor
母线数据提取器

从 SmartGrid Circuit 对象中提取所有母线的电压、相位、
基准电压等数据，供可视化图表使用。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_bus_data(env) -> Dict[str, Dict[str, Any]]:
	"""从 SmartGrid 环境提取所有母线数据

	Args:
		env: SmartGrid Env 实例 (envs.smartgrid.base_env.env.Env)

	Returns:
		{bus_name: {v_mag_pu, v_angle_deg, v_mag_kv, n_phases, base_kv, ...}}
	"""
	bus_data: Dict[str, Dict[str, Any]] = {}

	try:
		circuit = env.circuit
		dss = circuit.dss
		bus_names = dss.ActiveCircuit.AllBusNames

		for bus_name in bus_names:
			dss.ActiveCircuit.SetActiveBus(bus_name)
			bus_info = _extract_single_bus(dss, bus_name)
			bus_data[bus_name] = bus_info

	except Exception as exc:
		logger.error(f"Bus data extraction failed: {exc}")

	return bus_data


def extract_bus_voltages_from_obs(obs: dict) -> Dict[str, List[float]]:
	"""从 env.obs 字典中提取母线电压

	Args:
		obs: 环境观测字典，含 'bus_voltages' 键

	Returns:
		{bus_name: [v_mag_pu_phase1, ...]}
	"""
	return obs.get("bus_voltages", {})


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


def get_bus_voltage_series(
	snapshots: List[Dict[str, Any]],
	bus_name: str,
) -> List[float]:
	"""从快照序列中提取指定母线的电压时间序列

	Args:
		snapshots: 快照列表
		bus_name: 母线名称

	Returns:
		电压均值时间序列
	"""
	series: List[float] = []
	for snap in snapshots:
		buses = snap.get("buses", {})
		bus_info = buses.get(bus_name, {})
		v_pu = bus_info.get("v_mag_pu", [])
		if v_pu:
			series.append(float(np.mean(v_pu)))
		else:
			series.append(0.0)
	return series


def _extract_single_bus(dss, bus_name: str) -> Dict[str, Any]:
	"""提取单个母线的详细数据"""
	info: Dict[str, Any] = {
		"name": bus_name,
		"v_mag_pu": [],
		"v_angle_deg": [],
		"v_mag_kv": [],
		"n_phases": 0,
		"base_kv": 0.0,
		"coord_defined": False,
		"x": 0.0,
		"y": 0.0,
	}

	try:
		pu_v_angle = dss.ActiveCircuit.Buses.puVmagAngle
		n_vals = len(pu_v_angle)
		n_phases = n_vals // 2

		v_mag_pu = [float(pu_v_angle[i * 2]) for i in range(n_phases)]
		v_angle_deg = [float(pu_v_angle[i * 2 + 1]) for i in range(n_phases)]

		base_kv = float(dss.ActiveCircuit.Buses.kVBase)
		v_mag_kv = [v * base_kv for v in v_mag_pu]

		info["v_mag_pu"] = v_mag_pu
		info["v_angle_deg"] = v_angle_deg
		info["v_mag_kv"] = v_mag_kv
		info["n_phases"] = n_phases
		info["base_kv"] = base_kv

		if dss.ActiveCircuit.Buses.Coorddefined:
			info["coord_defined"] = True
			info["x"] = float(dss.ActiveCircuit.Buses.x)
			info["y"] = float(dss.ActiveCircuit.Buses.y)

	except Exception as exc:
		logger.debug(f"Failed to extract bus '{bus_name}': {exc}")

	return info
