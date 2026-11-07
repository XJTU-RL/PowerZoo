# -*- coding: utf-8 -*-
"""
Stackelberg 手动动作覆盖管理器

为异质智能体提供差异化的动作标签：
- UC Leader (agent_id=0): price, DR_signal, ESS_charge, ESS_discharge, reserve
- Consumer (agent_id>0): load_adjustment, DER_output, flexibility
"""

from typing import List

from envs.render_common.engine.base_manual_override import BaseManualOverride

# UC Leader 动作标签
UC_ACTION_LABELS: List[str] = [
	"price",
	"DR_signal",
	"ESS_charge",
	"ESS_discharge",
	"reserve",
]

# Consumer 动作标签
CONSUMER_ACTION_LABELS: List[str] = [
	"load_adjustment",
	"DER_output",
	"flexibility",
]


class StackelbergManualOverride(BaseManualOverride):
	"""Stackelberg 手动覆盖管理器

	处理异质动作维度：UC=5D, Consumer=3D。

	Args:
		n_agents: 智能体总数 (1 UC + N Consumers)
	"""

	def __init__(self, n_agents: int):
		action_dims = [5] + [3] * (n_agents - 1)
		super().__init__(n_agents=n_agents, action_dims=action_dims)

	def get_action_labels(self, agent_id: int) -> List[str]:
		"""获取指定智能体的动作维度语义标签

		Args:
			agent_id: 智能体索引

		Returns:
			动作维度标签列表
		"""
		if agent_id == 0:
			return UC_ACTION_LABELS
		return CONSUMER_ACTION_LABELS

	@property
	def uc_labels(self) -> List[str]:
		"""UC Leader 动作标签"""
		return UC_ACTION_LABELS

	@property
	def consumer_labels(self) -> List[str]:
		"""Consumer 动作标签"""
		return CONSUMER_ACTION_LABELS
