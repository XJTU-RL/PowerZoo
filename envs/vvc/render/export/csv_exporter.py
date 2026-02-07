# -*- coding: utf-8 -*-
"""
VVC CSV 数据导出器

继承 BaseCsvExporter，增加 VVC 特有的 device_states 导出:
电容器开关、调压器分接头、电池 SOC、PV 输出。
"""

import csv
import io
from typing import Any, Callable, Dict, List

from envs.render_common.export.base_csv_exporter import BaseCsvExporter, _fmt


class VVCCsvExporter(BaseCsvExporter):
	"""VVC CSV 数据导出器

	支持 5 种数据类别的 CSV 导出:
	bus_voltages, system_totals, agent_actions, agent_rewards, device_states
	"""

	EXPORT_CATEGORIES: List[str] = [
		"bus_voltages",
		"system_totals",
		"agent_actions",
		"agent_rewards",
		"device_states",
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
			"device_states": self._generate_device_states_csv,
		}

	@staticmethod
	def _generate_device_states_csv(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""生成 VVC 设备状态 CSV。

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
			"bus", "state_or_value", "detail_1", "detail_2",
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
					"ON" if cd.get("is_on", False) else "OFF",
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
			pvs = devices.get("pvsystems", {})
			for name, pd_data in pvs.items():
				writer.writerow([
					step, "pvsystem", name,
					pd_data.get("bus", ""),
					_fmt(pd_data.get("kw")),
					_fmt(pd_data.get("curtail_pct")),
					_fmt(pd_data.get("pmpp")),
				])

		return output.getvalue()
