# -*- coding: utf-8 -*-
"""
Power Exchange Manager
台区间功率交换管理器

管理台区间的虚拟功率交换，通过 OpenDSS 中的虚拟 Load/Generator 对实现。
A 台区向 B 台区输送 P kW:
  - A 侧边界母线: Load.exchange_{tie_name}_from kW=P (消耗)
  - B 侧边界母线: Generator.exchange_{tie_name}_to kW=P (注入)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from envs.district_dispatch.constants import EXCHANGE
from envs.district_dispatch.core.config import (
	DistrictDispatchConfig,
	TieLineConfig,
)

logger = logging.getLogger(__name__)


class PowerExchangeManager:
	"""台区间功率交换管理器

	管理台区间的虚拟功率交换。功率守恒原则：
	源台区消耗 P kW = 目标台区注入 P kW。

	参数:
		config: 区域调度配置
	"""

	def __init__(self, config: DistrictDispatchConfig):
		self.config = config
		self.tie_lines: List[TieLineConfig] = config.tie_lines
		# 当前交换状态: {(from_id, to_id): (p_kw, q_kvar)}
		self.exchange_records: Dict[
			Tuple[int, int], Tuple[float, float]
		] = {}
		self._initialized = False

	def initialize_exchange_elements(
		self, circuit_adapter
	) -> None:
		"""在电路中创建所有联络线的虚拟交换元素

		必须在首次 apply_exchange 之前调用。

		参数:
			circuit_adapter: DistrictCircuitAdapter 实例
		"""
		for tl in self.tie_lines:
			circuit_adapter.create_exchange_elements(
				tl.line_name, tl.from_bus, tl.to_bus
			)
			key = (tl.from_district, tl.to_district)
			self.exchange_records[key] = (0.0, 0.0)
		self._initialized = True
		logger.info(
			f"已初始化 {len(self.tie_lines)} 条联络线交换元素"
		)

	def apply_exchange(
		self,
		circuit_adapter,
		exchange_actions: Dict[Tuple[int, int], Tuple[float, float]],
	) -> None:
		"""应用所有台区间的功率交换

		参数:
			circuit_adapter: DistrictCircuitAdapter 实例
			exchange_actions: {(from_id, to_id): (p_kw, q_kvar)}
				p_kw 正值表示 from -> to 方向传输
				p_kw 负值表示 to -> from 方向传输
		"""
		if not self._initialized:
			logger.warning(
				"交换元素未初始化，先调用 initialize_exchange_elements()"
			)
			return

		for (from_id, to_id), (p_kw, q_kvar) in exchange_actions.items():
			tl = self.config.get_tie_line(from_id, to_id)
			if tl is None:
				logger.warning(
					f"未找到台区 {from_id}-{to_id} 间的联络线"
				)
				continue

			# 容量约束
			p_kw = float(
				np.clip(p_kw, -tl.capacity_kw, tl.capacity_kw)
			)
			# 无功按功率因数约束
			max_q = tl.capacity_kw * np.sqrt(
				1.0 - EXCHANGE.POWER_FACTOR ** 2
			) / EXCHANGE.POWER_FACTOR
			q_kvar = float(np.clip(q_kvar, -max_q, max_q))

			# 确定实际方向
			if p_kw >= 0:
				# from -> to: from 侧 Load 消耗, to 侧 Generator 注入
				circuit_adapter.set_exchange_power(
					tl.from_bus, tl.to_bus, tl.line_name,
					p_kw, q_kvar,
				)
			else:
				# to -> from: 反向传输
				circuit_adapter.set_exchange_power(
					tl.to_bus, tl.from_bus, tl.line_name,
					abs(p_kw), abs(q_kvar),
				)

			# 记录（始终以 from_id, to_id 的原始顺序记录，正值=正向）
			self.exchange_records[(from_id, to_id)] = (
				p_kw, q_kvar,
			)

	def decode_agent_exchange_actions(
		self,
		agent_id: int,
		action_slice: np.ndarray,
		neighbor_ids: List[int],
		config: DistrictDispatchConfig,
	) -> Dict[Tuple[int, int], Tuple[float, float]]:
		"""从 agent 的动作向量中解码功率交换指令

		动作布局: [p_exchange_0, p_exchange_1, ..., q_exchange_0, q_exchange_1, ...]
		前 n_neighbors 个为有功交换（归一化 [-1, 1]），
		后 n_neighbors 个为无功交换（归一化 [-1, 1]）。

		参数:
			agent_id: 当前 agent（台区）编号
			action_slice: 交换动作切片, shape=(n_neighbors * 2,)
			neighbor_ids: 邻居台区 ID 列表
			config: 配置

		返回:
			{(from_id, to_id): (p_kw, q_kvar)} 交换动作字典
		"""
		n_neighbors = len(neighbor_ids)
		if len(action_slice) < n_neighbors * 2:
			logger.warning(
				f"Agent {agent_id} 交换动作维度不足: "
				f"需要 {n_neighbors * 2}, 实际 {len(action_slice)}"
			)
			return {}

		exchange_actions = {}
		for i, nb_id in enumerate(neighbor_ids):
			# 有功和无功交换（归一化 [-1, 1]）
			p_norm = float(np.clip(action_slice[i], -1.0, 1.0))
			q_norm = float(
				np.clip(action_slice[n_neighbors + i], -1.0, 1.0)
			)

			# 获取联络线容量
			tl = config.get_tie_line(agent_id, nb_id)
			if tl is None:
				continue

			p_kw = p_norm * tl.capacity_kw
			max_q = tl.capacity_kw * np.sqrt(
				1.0 - EXCHANGE.POWER_FACTOR ** 2
			) / EXCHANGE.POWER_FACTOR
			q_kvar = q_norm * max_q

			# 确定方向键: 始终以 (小ID, 大ID) 记录
			if agent_id < nb_id:
				exchange_actions[(agent_id, nb_id)] = (
					p_kw, q_kvar,
				)
			else:
				# 反向: agent 是 to 端，翻转符号
				exchange_actions[(nb_id, agent_id)] = (
					-p_kw, -q_kvar,
				)

		return exchange_actions

	def get_exchange_summary(
		self, district_id: int
	) -> Tuple[float, float]:
		"""获取某台区的功率交换汇总

		参数:
			district_id: 台区编号

		返回:
			(exchange_in_kw, exchange_out_kw) 均为非负值
		"""
		exchange_in = 0.0
		exchange_out = 0.0

		for (from_id, to_id), (p_kw, _) in self.exchange_records.items():
			if from_id == district_id:
				if p_kw >= 0:
					exchange_out += p_kw		# 正向输出
				else:
					exchange_in += abs(p_kw)	# 反向接收
			elif to_id == district_id:
				if p_kw >= 0:
					exchange_in += p_kw			# 正向接收
				else:
					exchange_out += abs(p_kw)	# 反向输出

		return exchange_in, exchange_out

	def get_total_exchange_power(self) -> float:
		"""获取系统总交换功率（绝对值之和）

		返回:
			总交换功率 (kW)
		"""
		return sum(
			abs(p_kw)
			for (p_kw, _) in self.exchange_records.values()
		)

	def reset(self, circuit_adapter) -> None:
		"""重置所有交换为零

		参数:
			circuit_adapter: DistrictCircuitAdapter 实例
		"""
		zero_actions = {
			(tl.from_district, tl.to_district): (0.0, 0.0)
			for tl in self.tie_lines
		}
		if self._initialized:
			self.apply_exchange(circuit_adapter, zero_actions)
		else:
			# 初始化并设零
			self.initialize_exchange_elements(circuit_adapter)
			self.apply_exchange(circuit_adapter, zero_actions)

		logger.debug("功率交换已重置为零")
