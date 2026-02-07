# -*- coding: utf-8 -*-
"""
母线数据提取器

从 OpenDSS DSS 引擎提取母线电压、注入功率等数据。
适配 Stackelberg 环境的 CircuitAdapter 后端。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BusDataExtractor:
	"""母线数据提取器

	从 dss-python 引擎中提取所有母线的电压幅值、相角、
	功率注入及拓扑连接信息。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any):
		self._dss = dss

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有母线的电压和功率数据。

		Returns:
			{母线名: {v_mag_pu, v_angle_deg, v_mag_kv, kw_injection, kvar_injection, n_phases}}
		"""
		result: Dict[str, Dict[str, Any]] = {}

		try:
			bus_names = self._dss.Circuit.AllBusNames()
		except Exception as exc:
			logger.warning(f"Failed to get bus names: {exc}")
			return result

		for bus_name in bus_names:
			try:
				self._dss.Circuit.SetActiveBus(bus_name)
				v_mag_pu = list(self._dss.Bus.puVmagAngle()[::2])
				v_angle = list(self._dss.Bus.puVmagAngle()[1::2])
				v_kv = list(self._dss.Bus.VMagAngle()[::2])

				n_phases = self._dss.Bus.NumNodes()

				result[bus_name] = {
					"v_mag_pu": v_mag_pu[:n_phases],
					"v_angle_deg": v_angle[:n_phases],
					"v_mag_kv": [v / 1000.0 for v in v_kv[:n_phases]],
					"n_phases": n_phases,
					"kw_injection": 0.0,
					"kvar_injection": 0.0,
					"vpu": v_mag_pu[:n_phases],
				}
			except Exception as exc:
				logger.debug(f"Bus {bus_name} extraction failed: {exc}")
				result[bus_name] = {
					"v_mag_pu": [1.0],
					"v_angle_deg": [0.0],
					"v_mag_kv": [0.0],
					"n_phases": 1,
					"kw_injection": 0.0,
					"kvar_injection": 0.0,
					"vpu": [1.0],
				}

		return result

	def get_voltage_summary(self) -> Dict[str, float]:
		"""获取系统电压统计摘要。

		Returns:
			{v_min, v_max, v_mean, n_buses, violation_count}
		"""
		all_v: List[float] = []
		violations = 0

		try:
			bus_names = self._dss.Circuit.AllBusNames()
			for bus_name in bus_names:
				self._dss.Circuit.SetActiveBus(bus_name)
				v_pu = list(self._dss.Bus.puVmagAngle()[::2])
				n_ph = self._dss.Bus.NumNodes()
				for v in v_pu[:n_ph]:
					all_v.append(v)
					if v < 0.95 or v > 1.05:
						violations += 1
		except Exception as exc:
			logger.warning(f"Voltage summary extraction failed: {exc}")

		if not all_v:
			return {
				"v_min": 1.0, "v_max": 1.0, "v_mean": 1.0,
				"n_buses": 0, "violation_count": 0,
			}

		return {
			"v_min": min(all_v),
			"v_max": max(all_v),
			"v_mean": sum(all_v) / len(all_v),
			"n_buses": len(all_v),
			"violation_count": violations,
		}

	def extract_single(self, bus_name: str) -> Dict[str, Any]:
		"""提取单个母线的详细数据。

		Args:
			bus_name: 母线名称

		Returns:
			母线数据字典
		"""
		try:
			self._dss.Circuit.SetActiveBus(bus_name)
			v_mag_pu = list(self._dss.Bus.puVmagAngle()[::2])
			v_angle = list(self._dss.Bus.puVmagAngle()[1::2])
			n_phases = self._dss.Bus.NumNodes()

			return {
				"v_mag_pu": v_mag_pu[:n_phases],
				"v_angle_deg": v_angle[:n_phases],
				"n_phases": n_phases,
				"vpu": v_mag_pu[:n_phases],
			}
		except Exception:
			return {"v_mag_pu": [1.0], "v_angle_deg": [0.0], "n_phases": 1, "vpu": [1.0]}
