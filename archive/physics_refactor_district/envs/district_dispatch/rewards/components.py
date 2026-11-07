# -*- coding: utf-8 -*-
"""
District Dispatch Reward Components
区域调度奖励分量

各奖励分量独立实现，通过 DistrictDispatchReward 复合调用。
每个分量返回 per-agent 标量奖励值。
"""

import logging
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from envs.district_dispatch.constants import VOLTAGE, STORAGE, MARKET, EXCHANGE

logger = logging.getLogger(__name__)


class BaseRewardComponent:
	"""奖励分量基类"""

	def __init__(self, weight: float = 1.0):
		self.weight = weight

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		"""计算奖励分量

		Args:
			district_id: 台区索引
			env_state: 环境状态字典，包含所有必要信息

		Returns:
			标量奖励值（未加权）
		"""
		raise NotImplementedError


class EconomicDispatchReward(BaseRewardComponent):
	"""经济调度奖励 — 最小化购电成本

	r_econ = -(net_purchase_kw * price_yuan_kwh * dt_hours)

	净购电量 = 本地负荷 - PV实际出力 + 储能充电 - 储能放电 + 交换输出 - 交换输入
	正值表示需要从上级电网购电，负值表示有盈余（卖电收益）。
	"""

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		district = env_state['districts'][district_id]
		price = env_state.get('electricity_price', MARKET.BASE_PRICE_YUAN_KWH)
		dt_hours = env_state.get('dt_hours', 0.25)  # 15min = 0.25h

		# 计算净购电量
		total_pv = sum(pv.actual_power_kw for pv in district.pv_units)
		total_storage = sum(s.current_power_kw for s in district.storage_units)
		# storage: 正=放电(减少购电), 负=充电(增加购电)
		net_purchase = (
			district.total_load_kw
			- total_pv
			- total_storage  # 放电减少购电，充电增加购电
			+ district.exchange_out_kw
			- district.exchange_in_kw
		)

		# 成本（元）→ 归一化到合理范围
		cost = net_purchase * price * dt_hours
		# 用基础负荷归一化
		base_cost = max(district.total_load_kw, 1.0) * price * dt_hours
		return -cost / max(base_cost, 1e-6)


class VoltageComplianceReward(BaseRewardComponent):
	"""电压合规奖励 — 平方铰链损失

	r_volt = -Σ max(0, v_min - v)² + max(0, v - v_max)²

	对台区内所有母线电压进行越限惩罚，使用平方铰链损失
	保证梯度在越限附近连续可导。
	"""

	def __init__(self, weight: float = 2.0,
				 v_min: float = VOLTAGE.MIN_PU,
				 v_max: float = VOLTAGE.MAX_PU):
		super().__init__(weight)
		self.v_min = v_min
		self.v_max = v_max

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		district = env_state['districts'][district_id]
		total_violation = 0.0
		n_buses = 0

		for bus_name, v_phases in district.voltages.items():
			for v in v_phases:
				under = max(0.0, self.v_min - v)
				over = max(0.0, v - self.v_max)
				total_violation += under ** 2 + over ** 2
				n_buses += 1

		# 按母线数归一化
		if n_buses > 0:
			total_violation /= n_buses

		return -total_violation


class LossMinimizationReward(BaseRewardComponent):
	"""网损最小化奖励

	r_loss = -(total_losses / base_load)

	总网损除以基础负荷归一化，避免不同系统规模间的量纲问题。
	"""

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		total_loss_kw = env_state.get('total_loss_kw', 0.0)
		total_load_kw = env_state.get('total_load_kw', 1.0)

		# 全局网损按台区数均分（各台区共同承担）
		n_districts = env_state.get('n_districts', 1)
		loss_share = total_loss_kw / max(n_districts, 1)

		return -loss_share / max(total_load_kw / n_districts, 1.0)


class CarbonReductionReward(BaseRewardComponent):
	"""碳排放减少奖励

	r_carbon = pv_actual_kw * carbon_intensity * dt_hours * carbon_price

	每使用 1 kWh PV 发电替代电网购电，减少 carbon_intensity kgCO2，
	乘以碳价得到碳减排经济价值。归一化到 [0, 1] 范围。
	"""

	def __init__(self, weight: float = 0.3,
				 carbon_intensity: float = MARKET.CARBON_INTENSITY,
				 carbon_price: float = MARKET.CARBON_PRICE):
		super().__init__(weight)
		self.carbon_intensity = carbon_intensity
		self.carbon_price = carbon_price

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		district = env_state['districts'][district_id]
		dt_hours = env_state.get('dt_hours', 0.25)

		total_pv = sum(pv.actual_power_kw for pv in district.pv_units)
		total_pv_capacity = sum(pv.capacity_kw for pv in district.pv_units)

		# 碳减排量（kgCO2）
		carbon_reduced = total_pv * dt_hours * self.carbon_intensity
		# 碳价值（元）
		carbon_value = carbon_reduced * self.carbon_price

		# 按 PV 容量归一化
		max_value = total_pv_capacity * dt_hours * self.carbon_intensity * self.carbon_price
		return carbon_value / max(max_value, 1e-6)


class ExchangeBalanceReward(BaseRewardComponent):
	"""功率交换平衡奖励 — 联络线过载惩罚

	r_exchange = -Σ (|P_flow| / P_capacity)²

	惩罚联络线接近或超过额定容量的功率流，鼓励均衡分配。
	"""

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		config = env_state.get('config')
		if config is None:
			return 0.0

		exchange_records = env_state.get('exchange_records', {})
		total_penalty = 0.0
		n_ties = 0

		for tl in config.tie_lines:
			if tl.from_district == district_id or tl.to_district == district_id:
				key = (tl.from_district, tl.to_district)
				p_flow, _ = exchange_records.get(key, (0.0, 0.0))
				ratio = abs(p_flow) / max(tl.capacity_kw, 1.0)
				total_penalty += ratio ** 2
				n_ties += 1

		if n_ties > 0:
			total_penalty /= n_ties

		return -total_penalty


class StorageHealthReward(BaseRewardComponent):
	"""储能健康奖励 — SOC 超出健康范围惩罚

	r_health = -Σ penalty(SOC ∉ [soc_healthy_min, soc_healthy_max])

	SOC 在健康范围内不惩罚，超出范围按距离平方惩罚。
	"""

	def __init__(self, weight: float = 0.1,
				 soc_healthy_min: float = STORAGE.SOC_HEALTHY_MIN,
				 soc_healthy_max: float = STORAGE.SOC_HEALTHY_MAX):
		super().__init__(weight)
		self.soc_healthy_min = soc_healthy_min
		self.soc_healthy_max = soc_healthy_max

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> float:
		district = env_state['districts'][district_id]
		total_penalty = 0.0
		n_storage = len(district.storage_units)

		for storage in district.storage_units:
			under = max(0.0, self.soc_healthy_min - storage.soc)
			over = max(0.0, storage.soc - self.soc_healthy_max)
			total_penalty += under ** 2 + over ** 2

		if n_storage > 0:
			total_penalty /= n_storage

		return -total_penalty
