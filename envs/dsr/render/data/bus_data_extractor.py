"""
DSR Bus Data Extractor
母线数据提取器

从 DSR 环境中提取母线电压、相位、通电状态等信息。
DSR 场景中母线分为三类: 带电 (energized)、断电 (de-energized)、故障 (faulted)。
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class BusDataExtractor:
	"""DSR 母线数据提取器

	从 DSR 环境实例中提取每个母线的电压信息和通电状态，
	支持增量提取（仅更新变化的母线）和全量提取。

	Args:
		env: DSREnv 实例
	"""

	def __init__(self, env: Any):
		self.env = env
		self._core = getattr(env, "core_env", None) or getattr(env, "dsr_core", None)
		self._bus_names: List[str] = self._get_bus_names()

	def extract(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有母线的电压和状态数据

		Returns:
			{bus_name: {v_mag_pu, v_angle_deg, v_mag_kv, phases, is_energized}} 字典
		"""
		buses: Dict[str, Dict[str, Any]] = {}

		energized = self._get_energized_buses()
		voltages = self._get_bus_voltages()

		for bus_name in self._bus_names:
			bus_data: Dict[str, Any] = {
				"v_mag_pu": [],
				"v_angle_deg": [],
				"v_mag_kv": [],
				"phases": 0,
				"is_energized": bus_name in energized,
			}

			bus_v = voltages.get(bus_name, [])
			if isinstance(bus_v, (list, np.ndarray)) and len(bus_v) > 0:
				# bus_v 可能是 [v1, v2, v3] 或 [[v_pu, angle], ...]
				if isinstance(bus_v[0], (list, tuple, np.ndarray)):
					for phase_data in bus_v:
						if len(phase_data) >= 2:
							bus_data["v_mag_pu"].append(float(phase_data[0]))
							bus_data["v_angle_deg"].append(float(phase_data[1]))
						elif len(phase_data) == 1:
							bus_data["v_mag_pu"].append(float(phase_data[0]))
							bus_data["v_angle_deg"].append(0.0)
				else:
					# 简单电压列表
					for v in bus_v:
						v_val = float(v) if not np.isnan(float(v)) else 0.0
						bus_data["v_mag_pu"].append(v_val)
						bus_data["v_angle_deg"].append(0.0)

				bus_data["phases"] = len(bus_data["v_mag_pu"])

				# 估算 kV（假设基准电压为 4.16kV / phase）
				base_kv = self._get_base_kv(bus_name)
				bus_data["v_mag_kv"] = [
					v_pu * base_kv for v_pu in bus_data["v_mag_pu"]
				]
			else:
				# 无电压数据的母线（断电或故障）
				bus_data["phases"] = 1
				if bus_name in energized:
					bus_data["v_mag_pu"] = [1.0]
				else:
					bus_data["v_mag_pu"] = [0.0]
				bus_data["v_angle_deg"] = [0.0]
				bus_data["v_mag_kv"] = [0.0]

			buses[bus_name] = bus_data

		return buses

	def get_voltage_statistics(self) -> Dict[str, float]:
		"""获取全系统电压统计

		Returns:
			{v_mean_pu, v_min_pu, v_max_pu, v_std_pu, n_violations} 统计字典
		"""
		buses = self.extract()
		all_v: List[float] = []

		for bus_data in buses.values():
			if bus_data["is_energized"]:
				all_v.extend(bus_data["v_mag_pu"])

		if not all_v:
			return {
				"v_mean_pu": 0.0,
				"v_min_pu": 0.0,
				"v_max_pu": 0.0,
				"v_std_pu": 0.0,
				"n_violations": 0,
			}

		v_arr = np.array(all_v)
		v_min_limit = getattr(self._core, "config", None)
		v_min_th = 0.95
		v_max_th = 1.05
		if v_min_limit is not None:
			v_min_th = getattr(v_min_limit, "v_min", 0.95)
			v_max_th = getattr(v_min_limit, "v_max", 1.05)

		violations = int(np.sum((v_arr < v_min_th) | (v_arr > v_max_th)))

		return {
			"v_mean_pu": float(np.mean(v_arr)),
			"v_min_pu": float(np.min(v_arr)),
			"v_max_pu": float(np.max(v_arr)),
			"v_std_pu": float(np.std(v_arr)),
			"n_violations": violations,
		}

	def _get_bus_names(self) -> List[str]:
		"""获取所有母线名称"""
		if self._core is not None:
			names = getattr(self._core, "all_bus_names", None)
			if names is not None:
				return list(names)
		return []

	def _get_energized_buses(self) -> Set[str]:
		"""获取当前带电母线集合"""
		if self._core is None:
			return set()
		# 尝试从核心环境获取
		try:
			obs = self._core._get_observations()
			return set(obs.get("energized_buses", []))
		except Exception:
			pass
		# 回退方案
		energized = getattr(self._core, "energized_buses", None)
		if energized is not None:
			return set(energized)
		return set(self._bus_names)

	def _get_bus_voltages(self) -> Dict[str, List]:
		"""获取母线电压数据"""
		if self._core is None:
			return {}
		try:
			obs = self._core._get_observations()
			return obs.get("bus_voltages", {})
		except Exception:
			pass
		return {}

	def _get_base_kv(self, bus_name: str) -> float:
		"""获取母线基准电压 (kV)"""
		try:
			circuit = getattr(self._core, "circuit", None)
			if circuit is not None:
				dss = getattr(circuit, "dss", None)
				if dss is not None:
					dss.ActiveCircuit.SetActiveBus(bus_name)
					return float(dss.ActiveCircuit.Buses.kVBase)
		except Exception:
			pass
		return 4.16  # 默认 13Bus 基准电压

	@property
	def bus_names(self) -> List[str]:
		"""所有母线名称列表"""
		return self._bus_names

	@property
	def n_buses(self) -> int:
		"""母线总数"""
		return len(self._bus_names)
