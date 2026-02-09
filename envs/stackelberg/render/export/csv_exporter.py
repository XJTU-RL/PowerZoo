# -*- coding: utf-8 -*-
"""
Stackelberg CSV 数据导出器

继承 BaseCsvExporter，额外提供 Stackelberg 特有的导出类别：
- market_data: TOU 电价、有效电价、DR 信号
- leader_actions: UC Leader 5D 动作明细
- follower_actions: Consumer 3D 动作明细
"""

import csv
import io
from typing import Any, Dict, List

from envs.render_common.export.base_csv_exporter import BaseCsvExporter, _fmt


class StackelbergCsvExporter(BaseCsvExporter):
	"""Stackelberg CSV 数据导出器

	在通用类别（bus_voltages, system_totals, agent_actions, agent_rewards）
	基础上添加市场数据和分角色动作导出。
	"""

	EXPORT_CATEGORIES: List[str] = [
		"bus_voltages",
		"system_totals",
		"agent_actions",
		"agent_rewards",
		"market_data",
		"leader_actions",
		"follower_actions",
	]

	def _get_generators(self) -> Dict[str, Any]:
		"""获取类别名到生成函数的映射

		Returns:
			{类别名: 生成函数(snapshots) -> str}
		"""
		return {
			"bus_voltages": self._generate_bus_voltages_csv,
			"system_totals": self._generate_system_totals_csv,
			"agent_actions": self._generate_agent_actions_csv,
			"agent_rewards": self._generate_agent_rewards_csv,
			"market_data": self._generate_market_data_csv,
			"leader_actions": self._generate_leader_actions_csv,
			"follower_actions": self._generate_follower_actions_csv,
		}

	@staticmethod
	def _generate_market_data_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成市场数据 CSV

		列：step, hour, tou_period, tou_base_price, effective_price,
		    dr_signal, ess_charge, ess_discharge, uc_utility, avg_consumer_utility

		Args:
			snapshots: 快照列表

		Returns:
			CSV 字符串
		"""
		output = io.StringIO()
		writer = csv.writer(output)
		writer.writerow([
			"step", "hour", "tou_period", "tou_base_price",
			"effective_price", "dr_signal",
			"ess_charge", "ess_discharge",
			"uc_utility", "avg_consumer_utility",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			market = snap.get("market_data", {})
			uc_act = market.get("uc_actions", {})

			writer.writerow([
				step,
				market.get("hour", step % 24),
				market.get("tou_period", ""),
				_fmt(market.get("tou_base_price")),
				_fmt(uc_act.get("effective_price")),
				_fmt(uc_act.get("dr_signal_value")),
				_fmt(uc_act.get("ESS_charge")),
				_fmt(uc_act.get("ESS_discharge")),
				_fmt(market.get("uc_utility")),
				_fmt(market.get("avg_consumer_utility")),
			])

		return output.getvalue()

	@staticmethod
	def _generate_leader_actions_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成 UC Leader 动作明细 CSV

		列：step, price, DR_signal, ESS_charge, ESS_discharge, reserve

		Args:
			snapshots: 快照列表

		Returns:
			CSV 字符串
		"""
		output = io.StringIO()
		writer = csv.writer(output)
		writer.writerow([
			"step", "price", "DR_signal",
			"ESS_charge", "ESS_discharge", "reserve",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			actions = snap.get("actions", [])
			if not actions:
				continue

			uc_actions = actions[0] if actions else []
			if isinstance(uc_actions, (list, tuple)):
				row = [step]
				labels = ["price", "DR_signal", "ESS_charge", "ESS_discharge", "reserve"]
				for i in range(5):
					val = uc_actions[i] if i < len(uc_actions) else 0.0
					row.append(_fmt(val))
				writer.writerow(row)

		return output.getvalue()

	@staticmethod
	def _generate_follower_actions_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成 Consumer Follower 动作明细 CSV

		列：step, consumer_id, load_adjustment, DER_output, flexibility

		Args:
			snapshots: 快照列表

		Returns:
			CSV 字符串
		"""
		output = io.StringIO()
		writer = csv.writer(output)
		writer.writerow([
			"step", "consumer_id",
			"load_adjustment", "DER_output", "flexibility",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			actions = snap.get("actions", [])
			if not actions or len(actions) < 2:
				continue

			for c_idx in range(1, len(actions)):
				c_actions = actions[c_idx]
				if isinstance(c_actions, (list, tuple)):
					row = [step, c_idx]
					for i in range(3):
						val = c_actions[i] if i < len(c_actions) else 0.0
						row.append(_fmt(val))
					writer.writerow(row)

		return output.getvalue()
