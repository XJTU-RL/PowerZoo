# -*- coding: utf-8 -*-
"""
市场数据提取器 (Stackelberg 独有)

从 Stackelberg 博弈环境中提取 TOU 电价、DR 信号、
UC Leader 定价行为及 Consumer 响应等市场数据。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# TOU 时段默认定义 (24小时制)
TOU_PERIODS: Dict[str, Dict[str, Any]] = {
	"off_peak": {"hours": list(range(0, 7)) + list(range(22, 24)), "base_price": 0.04},
	"mid_peak": {"hours": list(range(7, 11)) + list(range(17, 22)), "base_price": 0.08},
	"on_peak": {"hours": list(range(11, 17)), "base_price": 0.15},
}

# 24 小时 TOU 基础电价序列
TOU_BASE_PRICES: List[float] = []
for hour in range(24):
	if hour in TOU_PERIODS["on_peak"]["hours"]:
		TOU_BASE_PRICES.append(TOU_PERIODS["on_peak"]["base_price"])
	elif hour in TOU_PERIODS["mid_peak"]["hours"]:
		TOU_BASE_PRICES.append(TOU_PERIODS["mid_peak"]["base_price"])
	else:
		TOU_BASE_PRICES.append(TOU_PERIODS["off_peak"]["base_price"])


class MarketDataExtractor:
	"""市场数据提取器

	从 Stackelberg 环境快照和动作中提取市场相关数据，
	包括 UC Leader 的定价、DR 信号和 Consumer 的响应。

	Args:
		n_consumers: Consumer 数量
	"""

	def __init__(self, n_consumers: int = 1):
		self.n_consumers = n_consumers

	def extract_from_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		infos: Any,
		env_state: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""从单步快照中提取市场数据。

		Args:
			step: 当前时间步 (0-23, 对应小时)
			actions: 动作数组 shape=(n_agents, action_dim)
			infos: 环境 info 字典
			env_state: 环境内部状态 (可选)

		Returns:
			市场数据字典
		"""
		hour = step % 24
		tou_base = TOU_BASE_PRICES[hour]

		market_data: Dict[str, Any] = {
			"hour": hour,
			"tou_base_price": tou_base,
			"tou_period": self._get_tou_period(hour),
			"uc_actions": {},
			"consumer_actions": [],
			"uc_utility": 0.0,
			"avg_consumer_utility": 0.0,
		}

		if actions is None:
			return market_data

		actions_arr = np.array(actions)

		# UC Leader 动作 (agent_id=0): [price, DR_signal, ESS_charge, ESS_discharge, reserve]
		if actions_arr.shape[0] > 0:
			uc_actions = actions_arr[0]
			n_uc = min(len(uc_actions), 5)
			uc_labels = ["price", "DR_signal", "ESS_charge", "ESS_discharge", "reserve"]
			uc_dict: Dict[str, float] = {}
			for i in range(n_uc):
				uc_dict[uc_labels[i]] = float(uc_actions[i])

			# 实际电价 = TOU 基础 + UC 价格调整 (归一化到合理范围)
			price_adj = float(uc_actions[0]) if n_uc > 0 else 0.0
			uc_dict["effective_price"] = tou_base * (1.0 + 0.5 * price_adj)
			uc_dict["dr_signal_value"] = float(uc_actions[1]) if n_uc > 1 else 0.0

			market_data["uc_actions"] = uc_dict

		# Consumer 动作 (agent_id>0): [load_adjustment, DER_output, flexibility]
		consumer_actions_list: List[Dict[str, float]] = []
		for c_idx in range(1, actions_arr.shape[0]):
			c_actions = actions_arr[c_idx]
			n_c = min(len(c_actions), 3)
			c_labels = ["load_adjustment", "DER_output", "flexibility"]
			c_dict: Dict[str, float] = {}
			for i in range(n_c):
				c_dict[c_labels[i]] = float(c_actions[i])
			consumer_actions_list.append(c_dict)

		market_data["consumer_actions"] = consumer_actions_list

		# 从 infos 中提取 utility 数据 (如果可用)
		if infos is not None:
			market_data.update(self._extract_utility_from_infos(infos))

		# 从环境状态中提取额外数据
		if env_state is not None:
			market_data.update(self._extract_from_env_state(env_state))

		return market_data

	def extract_timeseries(
		self,
		snapshots: List[Dict[str, Any]],
	) -> Dict[str, List[float]]:
		"""从快照序列提取市场数据时间序列。

		Args:
			snapshots: 快照列表

		Returns:
			{field_name: [values_per_step]}
		"""
		series: Dict[str, List[float]] = {
			"tou_base_price": [],
			"effective_price": [],
			"dr_signal": [],
			"ess_charge": [],
			"ess_discharge": [],
			"avg_load_adjustment": [],
			"avg_der_output": [],
			"avg_flexibility": [],
			"uc_utility": [],
			"avg_consumer_utility": [],
		}

		for snap in snapshots:
			market = snap.get("market_data", {})
			uc = market.get("uc_actions", {})
			consumers = market.get("consumer_actions", [])

			series["tou_base_price"].append(market.get("tou_base_price", 0.0))
			series["effective_price"].append(uc.get("effective_price", 0.0))
			series["dr_signal"].append(uc.get("dr_signal_value", 0.0))
			series["ess_charge"].append(uc.get("ESS_charge", 0.0))
			series["ess_discharge"].append(uc.get("ESS_discharge", 0.0))
			series["uc_utility"].append(market.get("uc_utility", 0.0))
			series["avg_consumer_utility"].append(market.get("avg_consumer_utility", 0.0))

			# Consumer 平均值
			if consumers:
				avg_la = np.mean([c.get("load_adjustment", 0.0) for c in consumers])
				avg_der = np.mean([c.get("DER_output", 0.0) for c in consumers])
				avg_flex = np.mean([c.get("flexibility", 0.0) for c in consumers])
			else:
				avg_la = avg_der = avg_flex = 0.0

			series["avg_load_adjustment"].append(float(avg_la))
			series["avg_der_output"].append(float(avg_der))
			series["avg_flexibility"].append(float(avg_flex))

		return series

	def _get_tou_period(self, hour: int) -> str:
		"""获取 TOU 时段名称。"""
		for period_name, info in TOU_PERIODS.items():
			if hour in info["hours"]:
				return period_name
		return "off_peak"

	def _extract_utility_from_infos(self, infos: Any) -> Dict[str, float]:
		"""从 env infos 提取 utility 数据。"""
		result: Dict[str, float] = {}

		if isinstance(infos, dict):
			result["uc_utility"] = infos.get("uc_utility", 0.0)
			result["avg_consumer_utility"] = infos.get("avg_consumer_utility", 0.0)
		elif isinstance(infos, (list, tuple)):
			# infos 为 per-agent list
			if len(infos) > 0 and isinstance(infos[0], dict):
				result["uc_utility"] = infos[0].get("utility", 0.0)
			if len(infos) > 1:
				consumer_utils = []
				for info in infos[1:]:
					if isinstance(info, dict):
						consumer_utils.append(info.get("utility", 0.0))
				if consumer_utils:
					result["avg_consumer_utility"] = float(np.mean(consumer_utils))

		return result

	def _extract_from_env_state(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
		"""从环境内部状态中提取额外数据。"""
		result: Dict[str, Any] = {}

		if "ess_soc" in env_state:
			result["ess_soc"] = float(env_state["ess_soc"])
		if "demand_response_signal" in env_state:
			result["demand_response_signal"] = float(env_state["demand_response_signal"])
		if "total_demand" in env_state:
			result["total_demand"] = float(env_state["total_demand"])
		if "total_supply" in env_state:
			result["total_supply"] = float(env_state["total_supply"])

		return result
