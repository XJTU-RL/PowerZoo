"""
SmartGrid Component Data Extractor
OOP 组件数据提取器

使用 SmartGrid 的 OOP 组件封装（Capacitor, Regulator, Battery, PVSystem）
而非直接调 DSS COM 接口，提取所有可控设备的状态数据。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_capacitor_data(env) -> Dict[str, Dict[str, Any]]:
	"""提取电容器数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		{cap_name: {status, bus, kvar_rated, kvar_actual, ...}}
	"""
	data: Dict[str, Dict[str, Any]] = {}
	try:
		for name, cap in env.circuit.capacitors.items():
			data[name] = {
				"name": name,
				"type": "capacitor",
				"status": int(cap.status),
				"bus": getattr(cap, "bus1", ""),
				"kvar_rated": float(getattr(cap, "kvar", 0.0)),
				"n_phases": int(getattr(cap, "phases", 1)),
			}
	except Exception as exc:
		logger.error(f"Capacitor data extraction failed: {exc}")
	return data


def extract_regulator_data(env) -> Dict[str, Dict[str, Any]]:
	"""提取调压器数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		{reg_name: {tap, bus1, bus2, min_tap, max_tap, ...}}
	"""
	data: Dict[str, Dict[str, Any]] = {}
	try:
		for name, reg in env.circuit.regulators.items():
			data[name] = {
				"name": name,
				"type": "regulator",
				"tap": int(reg.tap),
				"bus1": getattr(reg, "bus1", ""),
				"bus2": getattr(reg, "bus2", ""),
				"min_tap": float(getattr(reg, "mintap", 0.9)),
				"max_tap": float(getattr(reg, "maxtap", 1.1)),
				"n_phases": int(getattr(reg, "phases", 1)),
			}
	except Exception as exc:
		logger.error(f"Regulator data extraction failed: {exc}")
	return data


def extract_battery_data(env) -> Dict[str, Dict[str, Any]]:
	"""提取电池储能数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		{bat_name: {soc, power_kw, max_kw, bus, ...}}
	"""
	data: Dict[str, Dict[str, Any]] = {}
	try:
		for name, bat in env.circuit.batteries.items():
			soc = float(getattr(bat, "soc", 0.0))
			max_kw = float(getattr(bat, "max_kw", 100.0))
			actual_power = 0.0
			if hasattr(bat, "actual_power"):
				actual_power = float(bat.actual_power())

			data[name] = {
				"name": name,
				"type": "battery",
				"soc": soc,
				"power_kw": -actual_power,  # 正值=放电
				"power_ratio": (-actual_power / max_kw) if max_kw > 0 else 0.0,
				"max_kw": max_kw,
				"max_kwh": float(getattr(bat, "max_kwh", max_kw * 4)),
				"bus": getattr(bat, "bus1", ""),
			}
	except Exception as exc:
		logger.error(f"Battery data extraction failed: {exc}")
	return data


def extract_pv_data(env) -> Dict[str, Dict[str, Any]]:
	"""提取光伏系统数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		{pv_name: {power_ratio, pf, rated_kw, bus, ...}}
	"""
	data: Dict[str, Dict[str, Any]] = {}
	try:
		pvs = getattr(env.circuit, "pvs", {})
		for name, pv in pvs.items():
			status = [0.5, 1.0]
			if hasattr(pv, "get_status"):
				status = pv.get_status()

			data[name] = {
				"name": name,
				"type": "pv",
				"power_ratio": float(status[0]) if len(status) > 0 else 0.0,
				"power_factor": float(status[1]) if len(status) > 1 else 1.0,
				"rated_kw": float(getattr(pv, "rated_kw", getattr(pv, "kVA", 100.0))),
				"bus": getattr(pv, "bus1", getattr(pv, "bus", "")),
			}
	except Exception as exc:
		logger.error(f"PV data extraction failed: {exc}")
	return data


def extract_all_components(env) -> Dict[str, Dict[str, Any]]:
	"""一次性提取所有可控组件数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		包含 capacitors, regulators, batteries, pvs 子字典的字典
	"""
	return {
		"capacitors": extract_capacitor_data(env),
		"regulators": extract_regulator_data(env),
		"batteries": extract_battery_data(env),
		"pvs": extract_pv_data(env),
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
		result["capacitors"].append({
			"name": name,
			"status": dev.get("status", 0),
		})

	for name, dev in devices.get("regulators", {}).items():
		result["regulators"].append({
			"name": name,
			"tap": dev.get("tap", 16),
		})

	for name, dev in devices.get("batteries", {}).items():
		result["batteries"].append({
			"name": name,
			"soc": dev.get("soc", 0.0),
			"power_kw": dev.get("power_kw", 0.0),
		})

	for name, dev in devices.get("pvs", {}).items():
		result["pvs"].append({
			"name": name,
			"power_ratio": dev.get("power_ratio", 0.0),
			"power_factor": dev.get("power_factor", 1.0),
		})

	return result
