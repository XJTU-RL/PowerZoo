"""
DSR Restoration Data Extractor (Unique)
故障恢复状态提取器 -- DSR 环境独有

提取故障恢复过程中的关键状态信息：
- fault_lines: 故障线路列表
- energized_buses: 带电母线集合
- de_energized_buses: 断电母线集合
- restored_loads: 已恢复负荷
- restoration_pct: 恢复百分比
- priority_loads: 优先级负荷分类
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# 负荷优先级定义
PRIORITY_LEVELS = {
	4: "critical",
	3: "high",
	2: "medium",
	1: "low",
	0: "none",
}

PRIORITY_DISPLAY_NAMES = {
	"critical": "Critical",
	"high": "High",
	"medium": "Medium",
	"low": "Low",
	"none": "No Priority",
}


class RestorationDataExtractor:
	"""DSR 故障恢复状态提取器

	从 DSR 环境中提取恢复过程的完整状态，
	包括故障位置、通电区域、已恢复负荷、
	按优先级分类的负荷恢复情况。

	Args:
		env: DSREnv 实例
	"""

	def __init__(self, env: Any):
		self.env = env
		self._core = getattr(env, "core_env", None) or getattr(env, "dsr_core", None)

	def extract(self) -> Dict[str, Any]:
		"""提取完整的恢复状态

		Returns:
			{
				fault_lines: 故障线路列表,
				energized_buses: 带电母线列表,
				de_energized_buses: 断电母线列表,
				restored_loads: 已恢复负荷信息列表,
				unrestored_loads: 未恢复负荷信息列表,
				restoration_pct: 恢复百分比,
				priority_breakdown: 按优先级分类的恢复统计,
				total_restored_kw: 已恢复负荷总功率,
				total_unrestored_kw: 未恢复负荷总功率,
				restoration_complete: 是否恢复完成,
			}
		"""
		result: Dict[str, Any] = {
			"fault_lines": [],
			"energized_buses": [],
			"de_energized_buses": [],
			"restored_loads": [],
			"unrestored_loads": [],
			"restoration_pct": 0.0,
			"priority_breakdown": {},
			"total_restored_kw": 0.0,
			"total_unrestored_kw": 0.0,
			"restoration_complete": False,
		}

		if self._core is None:
			return result

		# 故障线路
		result["fault_lines"] = list(getattr(self._core, "fault_lines", []))

		# 母线通电状态
		all_buses = set(getattr(self._core, "all_bus_names", []))
		energized = self._get_energized_buses()
		de_energized = all_buses - energized

		result["energized_buses"] = sorted(energized)
		result["de_energized_buses"] = sorted(de_energized)

		# 负荷恢复状态
		restored, unrestored = self._classify_loads()
		result["restored_loads"] = restored
		result["unrestored_loads"] = unrestored

		# 恢复百分比
		try:
			ratio = self._core._get_restored_load_ratio()
			result["restoration_pct"] = float(ratio) * 100.0
		except Exception:
			total_kw = sum(l.get("kw", 0) for l in restored) + sum(l.get("kw", 0) for l in unrestored)
			restored_kw = sum(l.get("kw", 0) for l in restored)
			result["restoration_pct"] = (restored_kw / total_kw * 100.0) if total_kw > 0 else 0.0

		# 功率统计
		result["total_restored_kw"] = sum(l.get("kw", 0) for l in restored)
		result["total_unrestored_kw"] = sum(l.get("kw", 0) for l in unrestored)

		# 优先级分类
		result["priority_breakdown"] = self._compute_priority_breakdown(restored, unrestored)

		# 恢复完成判定
		try:
			result["restoration_complete"] = self._core._check_restoration_complete()
		except Exception:
			result["restoration_complete"] = result["restoration_pct"] >= 100.0

		return result

	def get_restoration_timeline(self, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""从快照序列中提取恢复时间线

		Args:
			snapshots: 快照列表

		Returns:
			[{step, restoration_pct, n_energized, n_restored_loads, events}] 时间线
		"""
		timeline: List[Dict[str, Any]] = []

		prev_energized: Set[str] = set()
		prev_restored: Set[str] = set()

		for snap in snapshots:
			step = snap.get("step", 0)
			restoration = snap.get("restoration_data", {})

			current_energized = set(restoration.get("energized_buses", []))
			current_restored = set(
				l.get("name", "") for l in restoration.get("restored_loads", [])
			)

			# 本步新增事件
			events: List[str] = []
			new_energized = current_energized - prev_energized
			new_restored = current_restored - prev_restored
			if new_energized:
				events.append(f"Energized: {', '.join(sorted(new_energized))}")
			if new_restored:
				events.append(f"Restored: {', '.join(sorted(new_restored))}")

			timeline.append({
				"step": step,
				"restoration_pct": restoration.get("restoration_pct", 0.0),
				"n_energized": len(current_energized),
				"n_restored_loads": len(current_restored),
				"total_restored_kw": restoration.get("total_restored_kw", 0.0),
				"events": events,
			})

			prev_energized = current_energized
			prev_restored = current_restored

		return timeline

	def _get_energized_buses(self) -> Set[str]:
		"""获取带电母线集合"""
		try:
			obs = self._core._get_observations()
			return set(obs.get("energized_buses", []))
		except Exception:
			pass
		return set(getattr(self._core, "energized_buses", []))

	def _classify_loads(self) -> tuple:
		"""将负荷分为已恢复和未恢复两类

		Returns:
			(restored_list, unrestored_list) 元组
		"""
		restored: List[Dict[str, Any]] = []
		unrestored: List[Dict[str, Any]] = []

		if self._core is None:
			return restored, unrestored

		load_info = getattr(self._core, "load_info", {})
		load_agents = getattr(self._core, "load_agents", [])
		energized = self._get_energized_buses()

		try:
			obs = self._core._get_observations()
			load_states = obs.get("load_states", {})
		except Exception:
			load_states = {}

		# 从 load_agents 获取信息
		processed_loads: Set[str] = set()
		for agent in load_agents:
			managed = agent.get("managed_loads", [])
			load_name = agent.get("load_name", "")
			priority = agent.get("priority", 0)

			if managed:
				# 聚合负荷
				for m_load in managed:
					info = load_info.get(m_load, {})
					entry = {
						"name": m_load,
						"bus": info.get("bus", agent.get("bus", "")),
						"kw": info.get("kw", 0.0),
						"priority": priority,
						"priority_label": PRIORITY_LEVELS.get(priority, "none"),
					}
					is_restored = load_states.get(m_load, False)
					bus = entry["bus"]
					is_energized = bus in energized if bus else False

					if is_restored and is_energized:
						restored.append(entry)
					else:
						unrestored.append(entry)
					processed_loads.add(m_load)
			elif load_name:
				info = load_info.get(load_name, {})
				entry = {
					"name": load_name,
					"bus": info.get("bus", agent.get("bus", "")),
					"kw": info.get("kw", 0.0),
					"priority": priority,
					"priority_label": PRIORITY_LEVELS.get(priority, "none"),
				}
				is_restored = load_states.get(load_name, False)
				bus = entry["bus"]
				is_energized = bus in energized if bus else False

				if is_restored and is_energized:
					restored.append(entry)
				else:
					unrestored.append(entry)
				processed_loads.add(load_name)

		# 处理未被 agent 管理的负荷
		for load_name, info in load_info.items():
			if load_name in processed_loads:
				continue
			entry = {
				"name": load_name,
				"bus": info.get("bus", ""),
				"kw": info.get("kw", 0.0),
				"priority": info.get("priority", 0),
				"priority_label": PRIORITY_LEVELS.get(info.get("priority", 0), "none"),
			}
			is_restored = load_states.get(load_name, False)
			bus = entry["bus"]
			is_energized = bus in energized if bus else False

			if is_restored and is_energized:
				restored.append(entry)
			else:
				unrestored.append(entry)

		return restored, unrestored

	def _compute_priority_breakdown(
		self,
		restored: List[Dict[str, Any]],
		unrestored: List[Dict[str, Any]],
	) -> Dict[str, Dict[str, Any]]:
		"""按优先级计算恢复统计

		Args:
			restored: 已恢复负荷列表
			unrestored: 未恢复负荷列表

		Returns:
			{priority_label: {total_count, restored_count, total_kw, restored_kw, pct}}
		"""
		breakdown: Dict[str, Dict[str, Any]] = {}

		for label in PRIORITY_LEVELS.values():
			breakdown[label] = {
				"total_count": 0,
				"restored_count": 0,
				"total_kw": 0.0,
				"restored_kw": 0.0,
				"pct": 0.0,
			}

		for load in restored:
			label = load.get("priority_label", "none")
			if label not in breakdown:
				label = "none"
			breakdown[label]["total_count"] += 1
			breakdown[label]["restored_count"] += 1
			breakdown[label]["total_kw"] += load.get("kw", 0.0)
			breakdown[label]["restored_kw"] += load.get("kw", 0.0)

		for load in unrestored:
			label = load.get("priority_label", "none")
			if label not in breakdown:
				label = "none"
			breakdown[label]["total_count"] += 1
			breakdown[label]["total_kw"] += load.get("kw", 0.0)

		# 计算百分比
		for label, stats in breakdown.items():
			if stats["total_kw"] > 0:
				stats["pct"] = (stats["restored_kw"] / stats["total_kw"]) * 100.0

		return breakdown
