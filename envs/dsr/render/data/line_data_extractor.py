"""
DSR Line Data Extractor
线路数据提取器

从 DSR 环境中提取线路状态信息，包括:
- 故障状态 (faulted): 发生故障的线路
- 开关状态 (switch_state): 开关打开/关闭
- 负载率 (loading_pct): 线路负载百分比
- 连接关系 (from_bus, to_bus): 线路端点母线
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class LineDataExtractor:
	"""DSR 线路数据提取器

	从 DSR 环境核心模块中提取线路状态数据，
	特别关注故障线路和开关操作信息。

	Args:
		env: DSREnv 实例
	"""

	def __init__(self, env: Any):
		self.env = env
		self._core = getattr(env, "core_env", None) or getattr(env, "dsr_core", None)

	def extract(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有线路的状态数据

		Returns:
			{line_name: {from_bus, to_bus, is_faulted, is_open, loading_pct, ...}} 字典
		"""
		lines: Dict[str, Dict[str, Any]] = {}

		fault_lines = self._get_fault_lines()
		line_states = self._get_line_states()
		all_lines = self._get_all_line_names()

		for line_name in all_lines:
			line_data: Dict[str, Any] = {
				"from_bus": "",
				"to_bus": "",
				"is_faulted": line_name in fault_lines,
				"is_open": False,
				"loading_pct": 0.0,
				"current_amps": 0.0,
				"rating_amps": 0.0,
				"is_overloaded": False,
				"is_switch": line_name in self._get_switchable_lines(),
			}

			# 开关/线路状态
			if line_name in line_states:
				state = line_states[line_name]
				if isinstance(state, bool):
					line_data["is_open"] = not state
				elif isinstance(state, (int, float)):
					line_data["is_open"] = float(state) < 0.5

			# 连接信息
			endpoints = self._get_line_endpoints(line_name)
			if endpoints:
				line_data["from_bus"] = endpoints[0]
				line_data["to_bus"] = endpoints[1]

			# 负载率和电流
			loading = self._get_line_loading(line_name)
			if loading is not None:
				line_data["loading_pct"] = loading.get("loading_pct", 0.0)
				line_data["current_amps"] = loading.get("current_amps", 0.0)
				line_data["rating_amps"] = loading.get("rating_amps", 0.0)
				line_data["is_overloaded"] = line_data["loading_pct"] > 100.0

			lines[line_name] = line_data

		return lines

	def get_fault_summary(self) -> Dict[str, Any]:
		"""获取故障线路汇总信息

		Returns:
			{fault_lines, n_faults, n_open_switches, n_overloaded} 汇总字典
		"""
		lines = self.extract()
		fault_lines = [n for n, d in lines.items() if d["is_faulted"]]
		open_switches = [n for n, d in lines.items() if d["is_open"] and d["is_switch"]]
		overloaded = [n for n, d in lines.items() if d["is_overloaded"]]

		return {
			"fault_lines": fault_lines,
			"n_faults": len(fault_lines),
			"open_switches": open_switches,
			"n_open_switches": len(open_switches),
			"overloaded_lines": overloaded,
			"n_overloaded": len(overloaded),
			"total_lines": len(lines),
		}

	def get_switch_operations(self) -> List[Dict[str, Any]]:
		"""获取所有开关的当前操作状态

		Returns:
			开关操作信息列表
		"""
		operations: List[Dict[str, Any]] = []
		lines = self.extract()

		for line_name, data in lines.items():
			if data["is_switch"]:
				operations.append({
					"line_name": line_name,
					"is_open": data["is_open"],
					"is_faulted": data["is_faulted"],
					"from_bus": data["from_bus"],
					"to_bus": data["to_bus"],
				})

		return operations

	def _get_fault_lines(self) -> Set[str]:
		"""获取当前故障线路集合"""
		if self._core is None:
			return set()
		# fault_lines 属性
		faults = getattr(self._core, "fault_lines", None)
		if faults is not None:
			return set(faults)
		# 从 current_faults 获取
		faults = getattr(self._core, "current_faults", None)
		if faults is not None:
			return set(faults)
		return set()

	def _get_line_states(self) -> Dict[str, Any]:
		"""获取线路开关状态"""
		if self._core is None:
			return {}
		try:
			obs = self._core._get_observations()
			return obs.get("line_states", {})
		except Exception:
			pass
		return getattr(self._core, "line_states", {})

	def _get_all_line_names(self) -> List[str]:
		"""获取所有线路名称"""
		if self._core is None:
			return []
		# 可故障线路
		faultable = getattr(self._core, "faultable_lines", [])
		# 尝试获取全部线路
		try:
			circuit = getattr(self._core, "circuit", None)
			if circuit is not None:
				dss = getattr(circuit, "dss", None)
				if dss is not None:
					return list(dss.ActiveCircuit.Lines.AllNames)
		except Exception:
			pass
		return list(faultable)

	def _get_switchable_lines(self) -> Set[str]:
		"""获取可操作开关的线路"""
		if self._core is None:
			return set()
		faultable = getattr(self._core, "faultable_lines", [])
		return set(faultable)

	def _get_line_endpoints(self, line_name: str) -> Optional[List[str]]:
		"""获取线路端点母线

		Args:
			line_name: 线路名称

		Returns:
			[from_bus, to_bus] 或 None
		"""
		try:
			circuit = getattr(self._core, "circuit", None)
			if circuit is None:
				return None
			dss = getattr(circuit, "dss", None)
			if dss is None:
				return None
			dss.ActiveCircuit.SetActiveElement(f"Line.{line_name}")
			bus1 = dss.ActiveCircuit.ActiveElement.BusNames[0].split(".")[0]
			bus2 = dss.ActiveCircuit.ActiveElement.BusNames[1].split(".")[0]
			return [bus1, bus2]
		except Exception:
			return None

	def _get_line_loading(self, line_name: str) -> Optional[Dict[str, float]]:
		"""获取线路负载率信息

		Args:
			line_name: 线路名称

		Returns:
			{loading_pct, current_amps, rating_amps} 或 None
		"""
		try:
			circuit = getattr(self._core, "circuit", None)
			if circuit is None:
				return None
			dss = getattr(circuit, "dss", None)
			if dss is None:
				return None
			dss.ActiveCircuit.SetActiveElement(f"Line.{line_name}")
			currents = dss.ActiveCircuit.ActiveElement.CurrentsMagAng
			if currents is not None and len(currents) >= 2:
				max_current = max(currents[::2])  # 取幅值
				rating = dss.ActiveCircuit.ActiveElement.NormalAmps
				if rating > 0:
					return {
						"loading_pct": (max_current / rating) * 100.0,
						"current_amps": float(max_current),
						"rating_amps": float(rating),
					}
		except Exception:
			pass
		return None
