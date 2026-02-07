# -*- coding: utf-8 -*-
"""
District Model
单台区模型

封装每个台区的 DER 设备（PV、储能、EV 充电桩）和运行状态，
提供设备控制接口和观测向量构建。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from envs.district_dispatch.constants import STORAGE, VOLTAGE

logger = logging.getLogger(__name__)


@dataclass
class PVUnit:
	"""光伏单元"""
	name: str
	bus: str
	capacity_kw: float
	available_power_kw: float = 0.0		# 当前可用出力（由辐照决定）
	actual_power_kw: float = 0.0		# 实际出力（含削减后）
	curtailment_ratio: float = 0.0		# 削减率 (0=不削减, 1=全部削减)


@dataclass
class StorageUnit:
	"""储能单元"""
	name: str
	bus: str
	capacity_kwh: float
	max_power_kw: float
	soc: float = STORAGE.SOC_INIT
	current_power_kw: float = 0.0		# 当前功率（正=放电，负=充电）
	charge_efficiency: float = STORAGE.CHARGE_EFFICIENCY
	discharge_efficiency: float = STORAGE.DISCHARGE_EFFICIENCY


@dataclass
class EVCharger:
	"""EV 充电桩"""
	name: str
	bus: str
	max_power_kw: float
	current_load_kw: float = 0.0


class District:
	"""单台区模型

	管理一个配电台区内的所有 DER 设备和运行状态。
	每个台区作为 MARL 中的一个 agent。

	参数:
		district_id: 台区编号
		name: 台区名称
		buses: 台区包含的所有母线
		boundary_buses: 边界母线（用于功率交换）
		pv_units: 光伏单元列表
		storage_units: 储能单元列表
		ev_chargers: EV 充电桩列表
	"""

	def __init__(
		self,
		district_id: int,
		name: str,
		buses: List[str],
		boundary_buses: List[str],
		pv_units: List[PVUnit],
		storage_units: List[StorageUnit],
		ev_chargers: List[EVCharger],
	):
		self.district_id = district_id
		self.name = name
		self.buses = buses
		self.boundary_buses = boundary_buses
		self.pv_units = pv_units
		self.storage_units = storage_units
		self.ev_chargers = ev_chargers

		# 台区运行状态
		self.voltages: Dict[str, List[float]] = {}
		self.total_load_kw: float = 0.0
		self.total_load_kvar: float = 0.0
		self.exchange_in_kw: float = 0.0
		self.exchange_out_kw: float = 0.0

		self.logger = logging.getLogger(
			f"District.{district_id}"
		)

	def apply_controls(
		self,
		circuit_adapter,
		pv_curtailments: np.ndarray,
		storage_cmds: np.ndarray,
		ev_modulations: np.ndarray,
	) -> None:
		"""应用设备控制到电路

		参数:
			circuit_adapter: DistrictCircuitAdapter 实例
			pv_curtailments: PV 削减率数组, shape=(n_pv,), 范围 [0, 1]
			storage_cmds: 储能控制数组, shape=(n_storage,), 归一化 [-1, 1]
			ev_modulations: EV 调制数组, shape=(n_ev,), 归一化 [0, 1]
		"""
		# PV 控制: 削减率 -> 出力百分比
		for i, pv in enumerate(self.pv_units):
			if i < len(pv_curtailments):
				curtail = float(np.clip(pv_curtailments[i], 0.0, 1.0))
				pv.curtailment_ratio = curtail
				pct_pmpp = (1.0 - curtail) * 100.0
				circuit_adapter.set_pv_output(pv.name, pct_pmpp)
				pv.actual_power_kw = (
					pv.available_power_kw * (1.0 - curtail)
				)

		# 储能控制: 归一化功率 -> 实际功率
		for i, storage in enumerate(self.storage_units):
			if i < len(storage_cmds):
				cmd = float(np.clip(storage_cmds[i], -1.0, 1.0))
				target_kw = cmd * storage.max_power_kw

				# SOC 边界保护
				if target_kw > 0 and storage.soc <= STORAGE.SOC_MIN:
					target_kw = 0.0
				elif target_kw < 0 and storage.soc >= STORAGE.SOC_MAX:
					target_kw = 0.0

				storage.current_power_kw = target_kw
				circuit_adapter.set_storage_power(
					storage.name, target_kw
				)

		# EV 调制: 归一化负荷比例
		for i, ev in enumerate(self.ev_chargers):
			if i < len(ev_modulations):
				mod = float(np.clip(ev_modulations[i], 0.0, 1.0))
				ev.current_load_kw = mod * ev.max_power_kw
				circuit_adapter.set_load_power(
					ev.name, ev.current_load_kw
				)

	def update_state(self, circuit_adapter) -> None:
		"""从电路读取更新台区状态

		参数:
			circuit_adapter: DistrictCircuitAdapter 实例
		"""
		# 更新母线电压
		self.voltages = circuit_adapter.get_bus_voltages(self.buses)

		# 更新 PV 实际出力
		for pv in self.pv_units:
			kw, _ = circuit_adapter.get_pv_output(pv.name)
			pv.actual_power_kw = kw

		# 更新储能 SOC
		for storage in self.storage_units:
			storage.soc = circuit_adapter.get_storage_soc(
				storage.name
			)

		# 获取台区总负荷（近似：遍历台区母线上的负荷）
		# NOTE: 精确负荷需要从 loadprofile 获取，这里用电路状态近似
		total_kw, total_kvar = circuit_adapter.get_total_load()
		# 按台区母线数占比近似分配（简化处理）
		all_buses = list(
			circuit_adapter.get_all_bus_voltages().keys()
		)
		ratio = (
			len(self.buses) / max(len(all_buses), 1)
		)
		self.total_load_kw = total_kw * ratio
		self.total_load_kvar = total_kvar * ratio

	def get_obs_vector(
		self,
		time_of_day: float,
		electricity_price: float,
		neighbor_info: Optional[List[Dict]] = None,
	) -> np.ndarray:
		"""构建观测向量

		维度对齐 config.get_max_obs_dim():
		- 台区电压统计: 3 (v_min, v_max, v_mean)
		- 本地负荷: 2 (P_load, Q_load) -- 归一化
		- PV 状态: n_pv * 2 (available, actual) -- 归一化
		- 储能状态: n_storage * 2 (soc, current_power) -- 归一化
		- EV 负荷: n_ev_charger -- 归一化
		- 功率交换: 2 (exchange_in, exchange_out) -- 归一化
		- 邻居信息: n_neighbors * 2 (v_mean, net_load)
		- 时间: 2 (time_of_day, electricity_price)

		参数:
			time_of_day: 归一化时间 (0-1)
			electricity_price: 归一化电价 (0-1)
			neighbor_info: 邻居信息列表 [{v_mean, net_load}, ...]

		返回:
			观测向量 np.ndarray
		"""
		obs_parts = []

		# 1. 电压统计 (3)
		v_min, v_max, v_mean = self.get_voltage_stats()
		obs_parts.extend([v_min, v_max, v_mean])

		# 2. 本地负荷 (2) -- 归一化到 [0, 1] 范围
		load_norm = 1000.0  # 归一化基准 (kW)
		obs_parts.append(self.total_load_kw / load_norm)
		obs_parts.append(self.total_load_kvar / load_norm)

		# 3. PV 状态 (n_pv * 2)
		for pv in self.pv_units:
			obs_parts.append(
				pv.available_power_kw / max(pv.capacity_kw, 1.0)
			)
			obs_parts.append(
				pv.actual_power_kw / max(pv.capacity_kw, 1.0)
			)

		# 4. 储能状态 (n_storage * 2)
		for storage in self.storage_units:
			obs_parts.append(storage.soc)
			obs_parts.append(
				storage.current_power_kw / max(storage.max_power_kw, 1.0)
			)

		# 5. EV 负荷 (n_ev_charger)
		for ev in self.ev_chargers:
			obs_parts.append(
				ev.current_load_kw / max(ev.max_power_kw, 1.0)
			)

		# 6. 功率交换 (2)
		exchange_norm = 500.0
		obs_parts.append(self.exchange_in_kw / exchange_norm)
		obs_parts.append(self.exchange_out_kw / exchange_norm)

		# 7. 邻居信息 (n_neighbors * 2)
		if neighbor_info is not None:
			for nb in neighbor_info:
				obs_parts.append(nb.get("v_mean", 1.0))
				obs_parts.append(
					nb.get("net_load", 0.0) / load_norm
				)

		# 8. 时间 + 电价 (2)
		obs_parts.append(time_of_day)
		obs_parts.append(electricity_price)

		return np.array(obs_parts, dtype=np.float32)

	def get_voltage_stats(self) -> Tuple[float, float, float]:
		"""返回台区电压统计

		返回:
			(v_min, v_max, v_mean) 标幺值
		"""
		all_v = []
		for bus_name, v_list in self.voltages.items():
			all_v.extend(v_list)

		if len(all_v) == 0:
			return VOLTAGE.TARGET_PU, VOLTAGE.TARGET_PU, VOLTAGE.TARGET_PU

		return float(np.min(all_v)), float(np.max(all_v)), float(np.mean(all_v))

	def get_net_load(self) -> float:
		"""获取台区净负荷 (负荷 - PV 出力)

		返回:
			净负荷 (kW)，正值表示需要从电网取电
		"""
		pv_total = sum(pv.actual_power_kw for pv in self.pv_units)
		return self.total_load_kw - pv_total

	def reset(self, soc_init: float = STORAGE.SOC_INIT) -> None:
		"""重置台区状态

		参数:
			soc_init: 储能初始 SOC
		"""
		self.voltages = {}
		self.total_load_kw = 0.0
		self.total_load_kvar = 0.0
		self.exchange_in_kw = 0.0
		self.exchange_out_kw = 0.0

		for pv in self.pv_units:
			pv.available_power_kw = 0.0
			pv.actual_power_kw = 0.0
			pv.curtailment_ratio = 0.0

		for storage in self.storage_units:
			storage.soc = soc_init
			storage.current_power_kw = 0.0

		for ev in self.ev_chargers:
			ev.current_load_kw = 0.0

		self.logger.debug(
			f"台区 {self.name} 已重置, SOC_init={soc_init:.2f}"
		)
