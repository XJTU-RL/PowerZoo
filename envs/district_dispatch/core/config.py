# -*- coding: utf-8 -*-
"""
District Dispatch Configuration
区域调度环境配置数据类

采用 dataclass 模式，与 DSR/SmartGrid 保持一致。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from envs.district_dispatch.constants import (
	VOLTAGE, STORAGE, MARKET, EPISODE, EXCHANGE
)


@dataclass
class DistrictDeviceConfig:
	"""单台区设备配置"""
	n_pv: int = 2						# PV系统数量
	n_storage: int = 1					# 储能系统数量
	n_ev_charger: int = 1				# EV充电桩数量
	pv_capacity_kw: float = 200.0		# 单个PV容量(kW)
	storage_capacity_kwh: float = 500.0	# 单个储能容量(kWh)
	storage_max_power_kw: float = 100.0	# 储能最大功率(kW)
	ev_max_power_kw: float = 50.0		# EV充电桩最大功率(kW)


@dataclass
class TieLineConfig:
	"""联络线配置"""
	from_district: int = 0				# 源台区索引
	to_district: int = 1				# 目标台区索引
	from_bus: str = ""					# 源母线名称
	to_bus: str = ""					# 目标母线名称
	line_name: str = ""					# 联络线名称
	capacity_kw: float = EXCHANGE.DEFAULT_CAPACITY_KW
	connection_type: str = "transformer"	# "transformer" | "tieline"


@dataclass
class DispatchRewardWeights:
	"""调度奖励权重"""
	economic_dispatch: float = 1.0		# 经济调度
	voltage_compliance: float = 2.0		# 电压合规
	loss_minimization: float = 0.5		# 网损最小化
	carbon_reduction: float = 0.3		# 碳排放减少
	exchange_balance: float = 0.2		# 功率交换平衡
	storage_health: float = 0.1			# 储能健康


@dataclass
class DistrictDispatchConfig:
	"""区域调度环境主配置"""

	# === 基础配置 ===
	env_name: str = "district_dispatch"
	system_name: str = "District_34Bus_3Zone"
	dss_file: str = "district_master.dss"
	max_episode_steps: int = EPISODE.DEFAULT_MAX_STEPS
	seed: int = 42

	# === 台区配置 ===
	n_districts: int = 3
	connection_mode: str = "mixed"		# "transformer" | "tieline" | "mixed"

	# 各台区设备配置（按台区索引）
	district_devices: Dict[int, DistrictDeviceConfig] = field(
		default_factory=lambda: {
			0: DistrictDeviceConfig(n_pv=2, n_storage=1, n_ev_charger=1),
			1: DistrictDeviceConfig(n_pv=2, n_storage=1, n_ev_charger=1),
			2: DistrictDeviceConfig(n_pv=1, n_storage=1, n_ev_charger=0),
		}
	)

	# 联络线配置
	tie_lines: List[TieLineConfig] = field(
		default_factory=lambda: [
			TieLineConfig(
				from_district=0, to_district=1,
				from_bus="830", to_bus="854",
				line_name="tie_0_1",
				capacity_kw=500.0,
				connection_type="transformer"
			),
			TieLineConfig(
				from_district=1, to_district=2,
				from_bus="848", to_bus="890",
				line_name="tie_1_2",
				capacity_kw=300.0,
				connection_type="tieline"
			),
		]
	)

	# === 物理约束 ===
	v_min: float = VOLTAGE.MIN_PU
	v_max: float = VOLTAGE.MAX_PU
	soc_min: float = STORAGE.SOC_MIN
	soc_max: float = STORAGE.SOC_MAX
	soc_healthy_min: float = STORAGE.SOC_HEALTHY_MIN
	soc_healthy_max: float = STORAGE.SOC_HEALTHY_MAX
	soc_init: float = STORAGE.SOC_INIT
	charge_efficiency: float = STORAGE.CHARGE_EFFICIENCY
	discharge_efficiency: float = STORAGE.DISCHARGE_EFFICIENCY

	# === 市场参数 ===
	base_price_yuan_kwh: float = MARKET.BASE_PRICE_YUAN_KWH
	carbon_intensity: float = MARKET.CARBON_INTENSITY
	carbon_price: float = MARKET.CARBON_PRICE
	tou_multipliers: List[float] = field(
		default_factory=lambda: list(MARKET.TOU_MULTIPLIERS)
	)

	# === 奖励权重 ===
	reward_weights: DispatchRewardWeights = field(
		default_factory=DispatchRewardWeights
	)

	# === 观测空间配置 ===
	use_neighbor_obs: bool = True		# 是否在观测中包含邻居信息
	max_neighbors: int = 2				# 最大邻居数

	# === 运行时配置 ===
	worker_idx: Optional[int] = None
	load_noise: bool = True				# 负荷随机噪声
	noise_std: float = 0.05			# 噪声标准差
	use_render: bool = False
	debug_mode: bool = False

	def get_max_action_dim(self) -> int:
		"""
		计算所有台区中最大动作维度（用于零填充对齐）

		动作向量 per agent:
		- 与每个邻居的有功交换: n_neighbors
		- 与每个邻居的无功交换: n_neighbors
		- PV削减率: n_pv
		- 储能控制: n_storage
		- EV调制: n_ev_charger
		"""
		max_dim = 0
		for d_id in range(self.n_districts):
			dev = self.district_devices.get(d_id, DistrictDeviceConfig())
			n_neighbors = self._count_neighbors(d_id)
			dim = (
				n_neighbors * 2		# 有功 + 无功交换
				+ dev.n_pv			# PV削减
				+ dev.n_storage		# 储能控制
				+ dev.n_ev_charger	# EV调制
			)
			max_dim = max(max_dim, dim)
		return max_dim

	def get_max_obs_dim(self) -> int:
		"""
		计算所有台区中最大观测维度

		观测向量 per agent:
		- 台区电压统计: 3 (v_min, v_max, v_mean)
		- 本地负荷: 2 (P_load, Q_load)
		- PV状态: n_pv * 2 (available, actual)
		- 储能状态: n_storage * 2 (soc, current_power)
		- EV负荷: n_ev_charger
		- 功率交换: 2 (exchange_in, exchange_out)
		- 邻居信息: n_neighbors * 2 (v_mean, net_load)
		- 时间: 2 (time_of_day, electricity_price)
		"""
		max_dim = 0
		for d_id in range(self.n_districts):
			dev = self.district_devices.get(d_id, DistrictDeviceConfig())
			n_neighbors = self._count_neighbors(d_id)
			dim = (
				3						# 电压统计
				+ 2						# 本地负荷
				+ dev.n_pv * 2			# PV状态
				+ dev.n_storage * 2		# 储能状态
				+ dev.n_ev_charger		# EV负荷
				+ 2						# 功率交换汇总
				+ n_neighbors * 2		# 邻居信息
				+ 2						# 时间+电价
			)
			max_dim = max(max_dim, dim)
		return max_dim

	def get_share_obs_dim(self) -> int:
		"""
		计算共享观测维度 (centralized critic)

		全局观测 = 所有台区本地观测拼接 + 系统级聚合
		"""
		total_local = self.get_max_obs_dim() * self.n_districts
		system_level = 6	# total_load, total_gen, total_loss, sys_v_min, sys_v_max, sys_v_mean
		market_level = 2	# price, carbon_intensity
		return total_local + system_level + market_level

	def _count_neighbors(self, district_id: int) -> int:
		"""计算台区的邻居数"""
		neighbors = set()
		for tl in self.tie_lines:
			if tl.from_district == district_id:
				neighbors.add(tl.to_district)
			elif tl.to_district == district_id:
				neighbors.add(tl.from_district)
		return min(len(neighbors), self.max_neighbors)

	def get_neighbor_ids(self, district_id: int) -> List[int]:
		"""获取台区的邻居台区ID列表"""
		neighbors = set()
		for tl in self.tie_lines:
			if tl.from_district == district_id:
				neighbors.add(tl.to_district)
			elif tl.to_district == district_id:
				neighbors.add(tl.from_district)
		return sorted(list(neighbors))[:self.max_neighbors]

	def get_tie_line(self, from_id: int, to_id: int) -> Optional[TieLineConfig]:
		"""获取两台区之间的联络线配置"""
		for tl in self.tie_lines:
			if (tl.from_district == from_id and tl.to_district == to_id) or \
				(tl.from_district == to_id and tl.to_district == from_id):
				return tl
		return None

	def validate(self) -> bool:
		"""验证配置合理性"""
		assert self.n_districts >= 2, "至少需要2个台区"
		assert self.max_episode_steps > 0, "max_episode_steps必须为正"
		assert 0 < self.v_min < self.v_max < 2.0, "无效的电压边界"
		assert 0 <= self.soc_min < self.soc_max <= 1.0, "无效的SOC边界"
		assert self.connection_mode in ("transformer", "tieline", "mixed"), \
			f"无效的连接模式: {self.connection_mode}"

		# 验证联络线引用的台区存在
		for tl in self.tie_lines:
			assert 0 <= tl.from_district < self.n_districts, \
				f"联络线引用了无效台区: {tl.from_district}"
			assert 0 <= tl.to_district < self.n_districts, \
				f"联络线引用了无效台区: {tl.to_district}"
			assert tl.from_district != tl.to_district, \
				"联络线的两端不能是同一台区"

		return True


# === 预定义配置 ===

DEFAULT_DISTRICT_DISPATCH_CONFIG = DistrictDispatchConfig()

DISTRICT_34BUS_3ZONE_CONFIG = DistrictDispatchConfig(
	system_name="District_34Bus_3Zone",
	n_districts=3,
	max_episode_steps=96,
)
