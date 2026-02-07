# -*- coding: utf-8 -*-
"""
District Dispatch Composite Reward
区域调度复合奖励函数

将各奖励分量按权重组合为最终奖励。
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from envs.district_dispatch.core.config import DistrictDispatchConfig
from envs.district_dispatch.rewards.components import (
	BaseRewardComponent,
	CarbonReductionReward,
	EconomicDispatchReward,
	ExchangeBalanceReward,
	LossMinimizationReward,
	StorageHealthReward,
	VoltageComplianceReward,
)

logger = logging.getLogger(__name__)


class DistrictDispatchReward:
	"""区域调度复合奖励函数

	total_reward = Σ(weight_i × component_i.compute(district_id, env_state))

	所有权重通过 DispatchRewardWeights 配置，新增分量只需:
	1. 在 components.py 添加新类
	2. 在 DispatchRewardWeights 添加新权重
	3. 在此处 __init__ 注册
	"""

	def __init__(self, config: DistrictDispatchConfig):
		"""初始化奖励函数

		Args:
			config: 环境配置，包含奖励权重和物理约束
		"""
		self.config = config
		weights = config.reward_weights

		# 注册奖励分量
		self.components: Dict[str, BaseRewardComponent] = {
			'economic_dispatch': EconomicDispatchReward(
				weight=weights.economic_dispatch
			),
			'voltage_compliance': VoltageComplianceReward(
				weight=weights.voltage_compliance,
				v_min=config.v_min,
				v_max=config.v_max,
			),
			'loss_minimization': LossMinimizationReward(
				weight=weights.loss_minimization
			),
			'carbon_reduction': CarbonReductionReward(
				weight=weights.carbon_reduction,
				carbon_intensity=config.carbon_intensity,
				carbon_price=config.carbon_price,
			),
			'exchange_balance': ExchangeBalanceReward(
				weight=weights.exchange_balance
			),
			'storage_health': StorageHealthReward(
				weight=weights.storage_health,
				soc_healthy_min=config.soc_healthy_min,
				soc_healthy_max=config.soc_healthy_max,
			),
		}

	def compute(self, district_id: int, env_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
		"""计算复合奖励

		Args:
			district_id: 台区索引
			env_state: 环境状态字典

		Returns:
			(total_reward, info_dict): 总奖励和各分量详情
		"""
		total_reward = 0.0
		info = {}

		for name, component in self.components.items():
			raw_value = component.compute(district_id, env_state)
			weighted_value = component.weight * raw_value
			total_reward += weighted_value
			info[f'reward/{name}_raw'] = raw_value
			info[f'reward/{name}_weighted'] = weighted_value

		info['reward/total'] = total_reward
		return total_reward, info

	def compute_all_agents(self, env_state: Dict[str, Any]) -> Tuple[List[float], List[Dict[str, float]]]:
		"""计算所有台区的奖励

		Args:
			env_state: 环境状态字典

		Returns:
			(rewards_list, infos_list): 所有台区的奖励和详情
		"""
		rewards = []
		infos = []
		for d_id in range(self.config.n_districts):
			reward, info = self.compute(d_id, env_state)
			rewards.append(reward)
			infos.append(info)
		return rewards, infos
