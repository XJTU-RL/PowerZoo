"""
SmartGrid Manual Override
SmartGrid 手动动作覆盖管理器

继承 BaseManualOverride，为 SmartGrid 的 CRBP 动作空间
提供语义标签（电容器 on/off、调压器 tap、电池功率、PV 削减）。
"""

import logging
from typing import List

from envs.render_common.engine.base_manual_override import BaseManualOverride

logger = logging.getLogger(__name__)


class SmartGridManualOverride(BaseManualOverride):
	"""SmartGrid 手动动作覆盖管理器

	Args:
		n_agents: 智能体数量（SmartGrid 通常为 1）
		action_dims: 每个智能体的动作维度列表
		cap_num: 电容器数量
		reg_num: 调压器数量
		bat_num: 电池数量
		pv_num: 光伏数量
		pv_control_enabled: 是否启用 PV 控制
	"""

	def __init__(
		self,
		n_agents: int,
		action_dims: List[int],
		cap_num: int = 0,
		reg_num: int = 0,
		bat_num: int = 0,
		pv_num: int = 0,
		pv_control_enabled: bool = False,
	):
		super().__init__(n_agents, action_dims)
		self.cap_num = cap_num
		self.reg_num = reg_num
		self.bat_num = bat_num
		self.pv_num = pv_num
		self.pv_control_enabled = pv_control_enabled

	def get_action_labels(self, agent_id: int) -> List[str]:
		"""获取动作维度的语义标签

		SmartGrid 动作空间布局 (CRBP):
		[cap_0, ..., cap_N, reg_0, ..., reg_M, bat_0, ..., bat_K, pv_0, ..., pv_J]

		Args:
			agent_id: 智能体索引

		Returns:
			动作标签列表
		"""
		labels: List[str] = []

		# 电容器 (on/off)
		for i in range(self.cap_num):
			labels.append(f"Cap_{i} (on/off)")

		# 调压器 (tap position)
		for i in range(self.reg_num):
			labels.append(f"Reg_{i} (tap)")

		# 电池 (power)
		for i in range(self.bat_num):
			labels.append(f"Bat_{i} (power)")

		# PV (如果启用)
		if self.pv_control_enabled:
			for i in range(self.pv_num):
				labels.append(f"PV_{i} (P_ratio)")
				labels.append(f"PV_{i} (PF)")

		return labels

	@classmethod
	def from_env(cls, env) -> "SmartGridManualOverride":
		"""从 SmartGrid Env 实例创建

		Args:
			env: SmartGrid Env 实例

		Returns:
			SmartGridManualOverride 实例
		"""
		action_dim = env.ActionSpace.dim()
		cap_num = env.cap_num
		reg_num = env.reg_num
		bat_num = env.bat_num
		pv_num = env.pv_num
		pv_control = getattr(env, "pv_control_enabled", False)

		return cls(
			n_agents=1,
			action_dims=[action_dim],
			cap_num=cap_num,
			reg_num=reg_num,
			bat_num=bat_num,
			pv_num=pv_num,
			pv_control_enabled=pv_control,
		)
