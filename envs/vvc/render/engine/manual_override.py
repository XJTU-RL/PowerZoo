# -*- coding: utf-8 -*-
"""
VVC 手动动作覆盖管理器

提供 VVC 环境下各设备类型的动作语义标签：
电容器开关、调压器分接头、电池功率、PV 削减。
"""

import logging
from typing import Any, Dict, List

from envs.render_common.engine.base_manual_override import BaseManualOverride

logger = logging.getLogger(__name__)


class VVCManualOverride(BaseManualOverride):
	"""VVC 手动动作覆盖管理器

	根据 VVC 环境的 CRBP 设备类型，为每个智能体提供
	语义化的动作标签。

	Args:
		n_agents: 智能体数量
		action_dims: 每个智能体的动作维度列表
		cap_num: 电容器数量
		reg_num: 调压器数量
		bat_num: 电池数量
		pv_num: 光伏数量
	"""

	def __init__(
		self,
		n_agents: int,
		action_dims: List[int],
		cap_num: int = 0,
		reg_num: int = 0,
		bat_num: int = 0,
		pv_num: int = 0,
	):
		super().__init__(n_agents, action_dims)
		self.cap_num = cap_num
		self.reg_num = reg_num
		self.bat_num = bat_num
		self.pv_num = pv_num

	def get_action_labels(self, agent_id: int) -> List[str]:
		"""获取指定智能体的动作维度语义标签。

		Args:
			agent_id: 智能体索引

		Returns:
			动作维度标签列表
		"""
		agent_type, type_idx = self._identify_agent(agent_id)

		if agent_type == "capacitor":
			return ["cap_switch"]

		elif agent_type == "regulator":
			return ["reg_tap"]

		elif agent_type == "battery":
			act_dim = self.action_dims[agent_id]
			if act_dim == 1:
				return ["bat_power"]
			else:
				# 离散电池动作
				return [f"bat_level_{i}" for i in range(act_dim)]

		elif agent_type == "pv":
			act_dim = self.action_dims[agent_id]
			if act_dim == 2:
				return ["pv_active_power", "pv_power_factor"]
			elif act_dim == 1:
				return ["pv_curtail"]
			else:
				return [f"pv_ctrl_{i}" for i in range(act_dim)]

		return [f"action_{i}" for i in range(self.action_dims[agent_id])]

	def _identify_agent(self, agent_id: int) -> tuple[str, int]:
		"""判断智能体的设备类型。

		Args:
			agent_id: 智能体索引

		Returns:
			(设备类型, 该类型内的索引)
		"""
		idx = 0

		if agent_id < idx + self.cap_num:
			return "capacitor", agent_id - idx
		idx += self.cap_num

		if agent_id < idx + self.reg_num:
			return "regulator", agent_id - idx
		idx += self.reg_num

		if agent_id < idx + self.bat_num:
			return "battery", agent_id - idx
		idx += self.bat_num

		if agent_id < idx + self.pv_num:
			return "pv", agent_id - idx

		return "unknown", agent_id
