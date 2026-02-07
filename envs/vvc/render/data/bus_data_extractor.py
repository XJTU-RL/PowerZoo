# -*- coding: utf-8 -*-
"""
母线数据提取器

通过 dss-python API 从 OpenDSS 引擎提取母线电压、相角、基准电压等数据。
每个母线返回各相的电压标幺值、相角和实际 kV 值。
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BusDataExtractor:
	"""母线数据提取器

	通过 dss-python 的 ActiveCircuit 接口逐母线提取电气数据。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any) -> None:
		self._dss = dss

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有母线的电气数据。

		Returns:
			{bus_name: {v_mag_pu, v_angle_deg, v_mag_kv, n_phases, base_kv}}
		"""
		result: Dict[str, Dict[str, Any]] = {}

		try:
			bus_names = self._dss.ActiveCircuit.AllBusNames
		except Exception as exc:
			logger.warning(f"Failed to get bus names: {exc}")
			return result

		for bus_name in bus_names:
			try:
				bus_data = self.extract_bus(bus_name)
				if bus_data:
					result[bus_name] = bus_data
			except Exception as exc:
				logger.debug(f"Failed to extract bus '{bus_name}': {exc}")

		return result

	def extract_bus(self, bus_name: str) -> Dict[str, Any]:
		"""提取单个母线的电气数据。

		Args:
			bus_name: 母线名称

		Returns:
			{v_mag_pu, v_angle_deg, v_mag_kv, n_phases, base_kv}
		"""
		circuit = self._dss.ActiveCircuit

		# 设置活动母线
		circuit.SetActiveBus(bus_name)
		bus = circuit.ActiveBus

		# 获取基准电压
		base_kv = bus.kVBase
		n_phases = bus.NumNodes

		# 获取电压 (复数形式)
		v_mag_pu_raw = list(bus.puVmagAngle)
		# puVmagAngle 返回交替的 [mag1, ang1, mag2, ang2, ...]
		v_mag_pu: List[float] = []
		v_angle_deg: List[float] = []
		v_mag_kv: List[float] = []

		for i in range(0, len(v_mag_pu_raw), 2):
			if i + 1 < len(v_mag_pu_raw):
				mag = float(v_mag_pu_raw[i])
				ang = float(v_mag_pu_raw[i + 1])
				v_mag_pu.append(mag)
				v_angle_deg.append(ang)
				v_mag_kv.append(mag * base_kv)

		return {
			"v_mag_pu": v_mag_pu,
			"v_angle_deg": v_angle_deg,
			"v_mag_kv": v_mag_kv,
			"n_phases": n_phases,
			"base_kv": base_kv,
		}

	def get_voltage_summary(self) -> Dict[str, Any]:
		"""获取全系统电压摘要统计。

		Returns:
			{v_mean_pu, v_min_pu, v_max_pu, v_min_bus, v_max_bus,
			 n_buses, n_violations, violation_buses}
		"""
		all_data = self.extract_all()

		if not all_data:
			return {
				"v_mean_pu": 1.0,
				"v_min_pu": 1.0,
				"v_max_pu": 1.0,
				"v_min_bus": "",
				"v_max_bus": "",
				"n_buses": 0,
				"n_violations": 0,
				"violation_buses": [],
			}

		all_v: List[float] = []
		bus_avg_v: Dict[str, float] = {}
		violation_buses: List[str] = []

		for bus_name, bus_data in all_data.items():
			v_pu = bus_data.get("v_mag_pu", [])
			if not v_pu:
				continue

			avg_v = sum(v_pu) / len(v_pu)
			bus_avg_v[bus_name] = avg_v
			all_v.extend(v_pu)

			# 检查电压越限 (0.95 ~ 1.05 pu)
			for v in v_pu:
				if v < 0.95 or v > 1.05:
					if bus_name not in violation_buses:
						violation_buses.append(bus_name)
					break

		if not all_v:
			return {
				"v_mean_pu": 1.0,
				"v_min_pu": 1.0,
				"v_max_pu": 1.0,
				"v_min_bus": "",
				"v_max_bus": "",
				"n_buses": 0,
				"n_violations": 0,
				"violation_buses": [],
			}

		v_min_bus = min(bus_avg_v, key=bus_avg_v.get)
		v_max_bus = max(bus_avg_v, key=bus_avg_v.get)

		return {
			"v_mean_pu": sum(all_v) / len(all_v),
			"v_min_pu": min(all_v),
			"v_max_pu": max(all_v),
			"v_min_bus": v_min_bus,
			"v_max_bus": v_max_bus,
			"n_buses": len(bus_avg_v),
			"n_violations": len(violation_buses),
			"violation_buses": violation_buses,
		}
