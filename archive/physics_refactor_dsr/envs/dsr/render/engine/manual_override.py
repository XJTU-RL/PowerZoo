"""
DSR Manual Override
DSR 手动动作覆盖管理器

继承 BaseManualOverride，提供 DSR 环境的动作语义标签。
DSR 的三类异质智能体有不同的动作语义:
- Switch agent: restore_line_0, restore_line_1, ..., no_action
- PV agent: power_level_0, power_level_1, ..., shutdown
- Load agent: disconnect, connect
"""

import logging
from typing import Any, Dict, List, Optional

from envs.render_common.engine.base_manual_override import BaseManualOverride

logger = logging.getLogger(__name__)


class DSRManualOverride(BaseManualOverride):
	"""DSR 手动动作覆盖管理器

	根据 agent 类型生成对应的动作语义标签。

	Args:
		n_agents: 智能体数量
		action_dims: 每个智能体的动作维度列表
		agent_types: 智能体类型列表 ("switch", "pv", "load")
		env: 可选的 DSREnv 实例（用于获取更多信息）
	"""

	def __init__(
		self,
		n_agents: int,
		action_dims: List[int],
		agent_types: Optional[List[str]] = None,
		env: Any = None,
	):
		super().__init__(n_agents, action_dims)
		self._agent_types = agent_types or []
		self._env = env
		self._label_cache: Dict[int, List[str]] = {}

	def get_action_labels(self, agent_id: int) -> List[str]:
		"""获取指定智能体的动作维度语义标签

		Args:
			agent_id: 智能体索引

		Returns:
			动作维度标签列表
		"""
		if agent_id in self._label_cache:
			return self._label_cache[agent_id]

		if agent_id >= len(self._agent_types):
			labels = [f"action_{i}" for i in range(self.action_dims[agent_id])]
			self._label_cache[agent_id] = labels
			return labels

		agent_type = self._agent_types[agent_id]
		n_actions = self.action_dims[agent_id]

		if agent_type == "switch":
			labels = self._build_switch_labels(agent_id, n_actions)
		elif agent_type == "pv":
			labels = self._build_pv_labels(agent_id, n_actions)
		elif agent_type == "load":
			labels = self._build_load_labels(agent_id, n_actions)
		else:
			labels = [f"action_{i}" for i in range(n_actions)]

		self._label_cache[agent_id] = labels
		return labels

	def get_agent_type(self, agent_id: int) -> str:
		"""获取智能体类型

		Args:
			agent_id: 智能体索引

		Returns:
			智能体类型字符串
		"""
		if agent_id < len(self._agent_types):
			return self._agent_types[agent_id]
		return "unknown"

	def get_agents_by_type(self) -> Dict[str, List[int]]:
		"""按类型分组返回智能体 ID

		Returns:
			{"switch": [0], "pv": [1, 2], "load": [3, 4, ...]}
		"""
		groups: Dict[str, List[int]] = {"switch": [], "pv": [], "load": []}
		for i, t in enumerate(self._agent_types):
			if t in groups:
				groups[t].append(i)
			else:
				groups.setdefault(t, []).append(i)
		return groups

	def _build_switch_labels(self, agent_id: int, n_actions: int) -> List[str]:
		"""构建 Switch agent 的动作标签"""
		labels: List[str] = []

		# 获取故障线路名称
		faultable_lines: List[str] = []
		if self._env is not None:
			core = getattr(self._env, "core_env", None) or getattr(self._env, "dsr_core", None)
			if core is not None:
				faultable_lines = list(getattr(core, "faultable_lines", []))

		for i in range(n_actions):
			if i < len(faultable_lines):
				labels.append(f"restore_{faultable_lines[i]}")
			elif i == n_actions - 1:
				labels.append("no_action")
			else:
				labels.append(f"restore_line_{i}")

		return labels

	def _build_pv_labels(self, agent_id: int, n_actions: int) -> List[str]:
		"""构建 PV agent 的动作标签"""
		labels: List[str] = []

		for i in range(n_actions):
			if i == n_actions - 1:
				labels.append("shutdown")
			else:
				# 功率级别百分比
				pct = int(i / max(n_actions - 2, 1) * 100)
				labels.append(f"power_{pct}%")

		return labels

	def _build_load_labels(self, agent_id: int, n_actions: int) -> List[str]:
		"""构建 Load agent 的动作标签"""
		if n_actions == 2:
			return ["disconnect", "connect"]
		elif n_actions > 2:
			labels = ["disconnect"]
			for i in range(1, n_actions):
				labels.append(f"connect_level_{i}")
			return labels
		else:
			return ["toggle"]
