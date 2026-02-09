"""
DSR Device Data Extractor
设备数据提取器

从 DSR 环境中提取 PV、负荷、开关等设备的状态信息。
DSR 环境的三类设备对应三类异质智能体:
- Switch agent: 控制线路开关
- PV agent: 控制光伏出力级别
- Load agent: 控制负荷投切
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DeviceDataExtractor:
	"""DSR 设备数据提取器

	提取 Switch / PV / Load 三类设备的当前状态和控制参数。

	Args:
		env: DSREnv 实例
	"""

	def __init__(self, env: Any):
		self.env = env
		self._core = getattr(env, "core_env", None) or getattr(env, "dsr_core", None)

	def extract(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有设备的状态数据

		Returns:
			{
				"switches": {name: {is_closed, from_bus, to_bus, ...}},
				"pvs": {name: {current_power, max_power, power_level, bus, ...}},
				"loads": {name: {is_connected, priority, kw, bus, ...}},
			}
		"""
		return {
			"switches": self._extract_switches(),
			"pvs": self._extract_pvs(),
			"loads": self._extract_loads(),
		}

	def extract_flat(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有设备并展平为单层字典

		Returns:
			{device_id: {type, name, ...}} 展平字典
		"""
		data = self.extract()
		flat: Dict[str, Dict[str, Any]] = {}

		for sw_name, sw_data in data["switches"].items():
			flat[f"switch_{sw_name}"] = {"type": "switch", "name": sw_name, **sw_data}
		for pv_name, pv_data in data["pvs"].items():
			flat[f"pv_{pv_name}"] = {"type": "pv", "name": pv_name, **pv_data}
		for ld_name, ld_data in data["loads"].items():
			flat[f"load_{ld_name}"] = {"type": "load", "name": ld_name, **ld_data}

		return flat

	def get_agent_device_mapping(self) -> List[Dict[str, Any]]:
		"""获取智能体到设备的映射关系

		Returns:
			[{agent_id, agent_type, device_name, device_type}] 映射列表
		"""
		mapping: List[Dict[str, Any]] = []
		if self._core is None:
			return mapping

		agent_types = getattr(self._core, "agent_types", [])
		for agent_id, agent_type in enumerate(agent_types):
			entry: Dict[str, Any] = {
				"agent_id": agent_id,
				"agent_type": agent_type,
				"device_name": f"agent_{agent_id}",
				"device_type": agent_type,
			}
			# 尝试获取更精确的设备名
			if agent_type == "pv":
				agent_idx = self._get_agent_device_index(agent_id)
				pv_agents = getattr(self._core, "pv_agents", [])
				if agent_idx < len(pv_agents):
					entry["device_name"] = pv_agents[agent_idx].get("name", f"pv_{agent_idx}")
					entry["bus"] = pv_agents[agent_idx].get("bus", "")
			elif agent_type == "load":
				agent_idx = self._get_agent_device_index(agent_id)
				load_agents = getattr(self._core, "load_agents", [])
				if agent_idx < len(load_agents):
					entry["device_name"] = load_agents[agent_idx].get("load_name", f"load_{agent_idx}")
					entry["bus"] = load_agents[agent_idx].get("bus", "")
					entry["priority"] = load_agents[agent_idx].get("priority", 0)

			mapping.append(entry)

		return mapping

	def _extract_switches(self) -> Dict[str, Dict[str, Any]]:
		"""提取开关设备状态"""
		switches: Dict[str, Dict[str, Any]] = {}
		if self._core is None:
			return switches

		faultable = getattr(self._core, "faultable_lines", [])
		fault_lines = set(getattr(self._core, "fault_lines", []))

		try:
			obs = self._core._get_observations()
			line_states = obs.get("line_states", {})
		except Exception:
			line_states = {}

		for i, line_name in enumerate(faultable):
			state = line_states.get(line_name, True)
			switches[line_name] = {
				"is_closed": bool(state),
				"is_faulted": line_name in fault_lines,
				"switch_idx": i,
			}

		return switches

	def _extract_pvs(self) -> Dict[str, Dict[str, Any]]:
		"""提取 PV 设备状态"""
		pvs: Dict[str, Dict[str, Any]] = {}
		if self._core is None:
			return pvs

		pv_agents = getattr(self._core, "pv_agents", [])
		for i, pv in enumerate(pv_agents):
			name = pv.get("name", f"pv_{i}")
			max_power = pv.get("max_power", 100.0)
			current_power = pv.get("current_power", 0.0)
			pvs[name] = {
				"current_power": float(current_power),
				"max_power": float(max_power),
				"power_ratio": float(current_power / max_power) if max_power > 0 else 0.0,
				"bus": pv.get("bus", ""),
				"pv_idx": i,
			}

		return pvs

	def _extract_loads(self) -> Dict[str, Dict[str, Any]]:
		"""提取负荷设备状态"""
		loads: Dict[str, Dict[str, Any]] = {}
		if self._core is None:
			return loads

		load_agents = getattr(self._core, "load_agents", [])
		try:
			obs = self._core._get_observations()
			load_states = obs.get("load_states", {})
		except Exception:
			load_states = {}

		for i, load in enumerate(load_agents):
			name = load.get("load_name", f"load_{i}")
			managed = load.get("managed_loads", [])

			load_data: Dict[str, Any] = {
				"is_connected": bool(load_states.get(name, False)),
				"priority": load.get("priority", 0),
				"bus": load.get("bus", ""),
				"load_idx": i,
				"is_aggregated": len(managed) > 1,
				"n_managed": len(managed) if managed else 1,
			}

			# 负荷功率
			kw = load.get("kw", 0.0)
			if kw == 0.0 and self._core is not None:
				load_info = getattr(self._core, "load_info", {})
				if name in load_info:
					kw = load_info[name].get("kw", 0.0)
			load_data["kw"] = float(kw)

			loads[name] = load_data

		return loads

	def _get_agent_device_index(self, agent_id: int) -> int:
		"""获取智能体对应的设备索引"""
		if self._core is None:
			return 0
		indices = getattr(self._core, "agent_indices", {})
		return indices.get(agent_id, 0)
