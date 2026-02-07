# -*- coding: utf-8 -*-
"""
SmartGrid CSV 数据导出器

继承 BaseCsvExporter，增加 SmartGrid 特有的导出类别:
- lagrangian_data: CMDP 拉格朗日乘子和约束违反数据
- component_states: 电容器/调压器/电池/PV 设备状态
"""

import csv
import io
from typing import Any, Callable, Dict, List

from envs.render_common.export.base_csv_exporter import BaseCsvExporter, _fmt


class SmartGridCsvExporter(BaseCsvExporter):
	"""SmartGrid CSV 数据导出器

	支持 6 种数据类别的 CSV 导出:
	bus_voltages, system_totals, agent_actions, agent_rewards,
	lagrangian_data, component_states
	"""

	EXPORT_CATEGORIES: List[str] = [
		"bus_voltages",
		"system_totals",
		"agent_actions",
		"agent_rewards",
		"lagrangian_data",
		"component_states",
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
			"lagrangian_data": self._generate_lagrangian_csv,
			"component_states": self._generate_component_states_csv,
		}

	@staticmethod
	def _generate_lagrangian_csv(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""生成 CMDP 拉格朗日乘子数据 CSV。

		包含 lambda、cost_voltage、voltage_violation_rate 等 CMDP 特有指标。

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
			"step", "lambda", "cost_voltage",
			"lagrangian_penalty", "reward_before_lagrangian",
			"voltage_violation_rate_buses", "voltage_violation_rate_phases",
			"voltage_violation_count",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			info = snap.get("info", {})
			rc = snap.get("reward_components", {})
			lagrangian = snap.get("lagrangian", {})
			merged = {**rc, **info, **lagrangian}

			writer.writerow([
				step,
				_fmt(merged.get("lambda", merged.get("lmbda"))),
				_fmt(merged.get("cost_voltage")),
				_fmt(merged.get("lagrangian_penalty")),
				_fmt(merged.get("reward_before_lagrangian")),
				_fmt(merged.get("voltage_violation_rate_buses")),
				_fmt(merged.get("voltage_violation_rate_phases")),
				_fmt(merged.get("voltage_violation_count")),
			])

		return output.getvalue()

	@staticmethod
	def _generate_component_states_csv(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""生成 SmartGrid 设备状态 CSV。

		包含电容器开关、调压器分接头、电池 SOC/功率、PV 输出/削减。

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
			"step", "device_type", "device_name",
			"bus", "status_or_value", "detail_1", "detail_2",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			devices = snap.get("devices", {})

			# 电容器
			caps = devices.get("capacitors", {})
			for name, cd in caps.items():
				writer.writerow([
					step, "capacitor", name,
					cd.get("bus", ""),
					cd.get("status", ""),
					_fmt(cd.get("kvar")),
					"",
				])

			# 调压器
			regs = devices.get("regulators", {})
			for name, rd in regs.items():
				writer.writerow([
					step, "regulator", name,
					rd.get("bus", ""),
					_fmt(rd.get("tap")),
					_fmt(rd.get("forward_vreg")),
					"",
				])

			# 电池
			bats = devices.get("batteries", {})
			for name, bd in bats.items():
				writer.writerow([
					step, "battery", name,
					bd.get("bus", ""),
					_fmt(bd.get("soc")),
					_fmt(bd.get("kw")),
					bd.get("state", ""),
				])

			# PV
			pvs = devices.get("pvs", devices.get("pvsystems", {}))
			for name, pd_data in pvs.items():
				writer.writerow([
					step, "pvsystem", name,
					pd_data.get("bus", ""),
					_fmt(pd_data.get("kw")),
					_fmt(pd_data.get("curtail_pct", pd_data.get("curtailment"))),
					_fmt(pd_data.get("pmpp")),
				])

		return output.getvalue()
