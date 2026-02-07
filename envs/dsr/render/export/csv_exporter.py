# -*- coding: utf-8 -*-
"""
DSR CSV 数据导出器

继承 BaseCsvExporter，增加 DSR 特有的导出类别:
- restoration_states: 恢复进度和优先级负荷数据
- action_masks: 动作掩码数据（合法/非法动作）
- fault_data: 故障线路和开关操作数据
"""

import csv
import io
from typing import Any, Callable, Dict, List

from envs.render_common.export.base_csv_exporter import BaseCsvExporter, _fmt


class DSRCsvExporter(BaseCsvExporter):
	"""DSR CSV 数据导出器

	支持 7 种数据类别的 CSV 导出:
	bus_voltages, system_totals, agent_actions, agent_rewards,
	restoration_states, action_masks, fault_data
	"""

	EXPORT_CATEGORIES: List[str] = [
		"bus_voltages",
		"system_totals",
		"agent_actions",
		"agent_rewards",
		"restoration_states",
		"action_masks",
		"fault_data",
	]

	def _get_generators(self) -> Dict[str, Callable]:
		"""获取类别名到生成函数的映射。

		Returns:
			{类别名: 生成函数(snapshots) -> str}
		"""
		return {
			"bus_voltages": self._generate_bus_voltages_csv,
			"system_totals": self._generate_system_totals_csv,
			"agent_actions": self._generate_agent_actions_csv,
			"agent_rewards": self._generate_agent_rewards_csv,
			"restoration_states": self._generate_restoration_states_csv,
			"action_masks": self._generate_action_masks_csv,
			"fault_data": self._generate_fault_data_csv,
		}

	@staticmethod
	def _generate_restoration_states_csv(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""生成恢复进度数据 CSV。

		包含 restoration_pct、total_restored_kw、各优先级负荷恢复状态。

		Args:
			snapshots: 快照列表

		Returns:
			CSV 字符串
		"""
		if not snapshots:
			return ""

		output = io.StringIO()
		writer = csv.writer(output)

		writer.writerow([
			"step", "restoration_pct", "total_restored_kw",
			"n_energized_buses", "n_deenergized_buses", "n_fault_lines",
			"critical_pct", "critical_restored_kw", "critical_total_kw",
			"high_pct", "high_restored_kw", "high_total_kw",
			"medium_pct", "medium_restored_kw", "medium_total_kw",
			"low_pct", "low_restored_kw", "low_total_kw",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			rest = snap.get("restoration_data", {})
			breakdown = rest.get("priority_breakdown", {})

			row = [
				step,
				_fmt(rest.get("restoration_pct", 0.0)),
				_fmt(rest.get("total_restored_kw", 0.0)),
				len(rest.get("energized_buses", [])),
				len(rest.get("de_energized_buses", [])),
				len(rest.get("fault_lines", [])),
			]

			for priority in ["critical", "high", "medium", "low"]:
				stats = breakdown.get(priority, {})
				row.extend([
					_fmt(stats.get("pct", 0.0)),
					_fmt(stats.get("restored_kw", 0.0)),
					_fmt(stats.get("total_kw", 0.0)),
				])

			writer.writerow(row)

		return output.getvalue()

	@staticmethod
	def _generate_action_masks_csv(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""生成动作掩码数据 CSV。

		每行记录一个 agent 在某个时间步的可用动作掩码向量。

		Args:
			snapshots: 快照列表

		Returns:
			CSV 字符串
		"""
		if not snapshots:
			return ""

		# 确定最大动作维度
		max_action_dim = 0
		for snap in snapshots:
			avail = snap.get("available_actions")
			if avail is None:
				continue
			if isinstance(avail, list):
				for agent_mask in avail:
					if isinstance(agent_mask, (list, tuple)):
						max_action_dim = max(max_action_dim, len(agent_mask))

		if max_action_dim == 0:
			return ""

		output = io.StringIO()
		writer = csv.writer(output)

		header = ["step", "agent_idx", "agent_type"]
		for d in range(max_action_dim):
			header.append(f"action_{d}_valid")
		header.append("n_valid_actions")
		writer.writerow(header)

		for snap in snapshots:
			step = snap.get("step", 0)
			avail = snap.get("available_actions")
			agent_types = snap.get("agent_types", [])

			if avail is None or not isinstance(avail, list):
				continue

			for agent_idx, agent_mask in enumerate(avail):
				agent_type = agent_types[agent_idx] if agent_idx < len(agent_types) else "unknown"

				row = [step, agent_idx, agent_type]
				if isinstance(agent_mask, (list, tuple)):
					n_valid = sum(1 for v in agent_mask if v)
					for v in agent_mask:
						row.append(1 if v else 0)
					# 补齐到 max_action_dim
					while len(row) < 3 + max_action_dim:
						row.append(0)
				else:
					n_valid = 0
					while len(row) < 3 + max_action_dim:
						row.append(0)

				row.append(n_valid)
				writer.writerow(row)

		return output.getvalue()

	@staticmethod
	def _generate_fault_data_csv(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""生成故障线路和开关操作数据 CSV。

		Args:
			snapshots: 快照列表

		Returns:
			CSV 字符串
		"""
		if not snapshots:
			return ""

		output = io.StringIO()
		writer = csv.writer(output)

		writer.writerow([
			"step", "line_name", "from_bus", "to_bus",
			"is_faulted", "is_open", "is_switch",
			"loading_pct", "is_overloaded",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			lines = snap.get("lines", {})

			for line_name, line_data in lines.items():
				writer.writerow([
					step,
					line_name,
					line_data.get("from_bus", ""),
					line_data.get("to_bus", ""),
					1 if line_data.get("is_faulted", False) else 0,
					1 if line_data.get("is_open", False) else 0,
					1 if line_data.get("is_switch", False) else 0,
					_fmt(line_data.get("loading_pct", 0.0)),
					1 if line_data.get("is_overloaded", False) else 0,
				])

		return output.getvalue()
