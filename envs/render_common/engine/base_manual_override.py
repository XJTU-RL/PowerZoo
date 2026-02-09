"""
Base Manual Override (Abstract)
手动动作覆盖管理器抽象基类

子类需实现 get_action_labels() 方法，提供环境特定的动作语义标签。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class BaseManualOverride(ABC):
	"""手动动作覆盖管理器

	在 Live Inference 模式下拦截模型输出的动作，
	将用户手动指定的值覆盖到对应的 agent/action_idx 上。

	Args:
		n_agents: 智能体数量
		action_dims: 每个智能体的动作维度列表
	"""

	def __init__(self, n_agents: int, action_dims: List[int]):
		self.n_agents = n_agents
		self.action_dims = action_dims
		self._overrides: Dict[int, Dict[int, float]] = {}
		self._enabled = False
		self._history: List[Dict[str, Any]] = []

	@property
	def enabled(self) -> bool:
		"""覆盖功能是否启用"""
		return self._enabled

	@enabled.setter
	def enabled(self, val: bool) -> None:
		self._enabled = val

	def set_override(
		self,
		agent_id: int,
		action_idx: int,
		value: float,
	) -> None:
		"""设置单个动作覆盖

		Args:
			agent_id: 智能体索引
			action_idx: 动作维度索引
			value: 覆盖值
		"""
		self._validate_indices(agent_id, action_idx)

		if agent_id not in self._overrides:
			self._overrides[agent_id] = {}

		self._overrides[agent_id][action_idx] = float(value)
		self._history.append({
			"action": "set",
			"agent_id": agent_id,
			"action_idx": action_idx,
			"value": float(value),
		})

	def clear_override(self, agent_id: int, action_idx: int) -> None:
		"""清除单个动作覆盖"""
		self._validate_indices(agent_id, action_idx)

		if agent_id in self._overrides:
			self._overrides[agent_id].pop(action_idx, None)
			if not self._overrides[agent_id]:
				del self._overrides[agent_id]

	def clear_agent(self, agent_id: int) -> None:
		"""清除指定智能体的所有覆盖"""
		if agent_id < 0 or agent_id >= self.n_agents:
			raise ValueError(f"agent_id={agent_id} out of range (n_agents={self.n_agents})")
		self._overrides.pop(agent_id, {})

	def clear_all(self) -> None:
		"""清除所有覆盖"""
		self._overrides.clear()

	def apply(self, actions: np.ndarray) -> np.ndarray:
		"""将覆盖应用到模型输出的动作上

		Args:
			actions: 模型输出的动作 shape=(n_agents, max_action_dim)

		Returns:
			应用覆盖后的动作 (新数组)
		"""
		modified = actions.copy()

		if not self._enabled or not self._overrides:
			return modified

		for agent_id, idx_map in self._overrides.items():
			for action_idx, value in idx_map.items():
				if (
					agent_id < modified.shape[0]
					and action_idx < modified.shape[1]
				):
					modified[agent_id, action_idx] = value

		return modified

	def get_override_summary(self) -> Dict[str, Any]:
		"""获取当前覆盖状态摘要

		Returns:
			摘要字典
		"""
		agents_detail: Dict[str, Any] = {}

		for agent_id, idx_map in self._overrides.items():
			labels = self.get_action_labels(agent_id)
			labeled_overrides = {}
			for action_idx, value in idx_map.items():
				label = labels[action_idx] if action_idx < len(labels) else f"dim_{action_idx}"
				labeled_overrides[f"idx_{action_idx} ({label})"] = round(value, 4)
			agents_detail[f"agent_{agent_id}"] = labeled_overrides

		return {
			"enabled": self._enabled,
			"total_overrides": sum(len(v) for v in self._overrides.values()),
			"agents": agents_detail,
			"history_length": len(self._history),
		}

	@abstractmethod
	def get_action_labels(self, agent_id: int) -> List[str]:
		"""获取指定智能体的动作维度语义标签

		Args:
			agent_id: 智能体索引

		Returns:
			动作维度标签列表
		"""
		...

	def _validate_indices(self, agent_id: int, action_idx: int) -> None:
		"""校验索引合法性"""
		if agent_id < 0 or agent_id >= self.n_agents:
			raise ValueError(
				f"agent_id={agent_id} out of range (n_agents={self.n_agents})"
			)
		if action_idx < 0 or action_idx >= self.action_dims[agent_id]:
			raise ValueError(
				f"action_idx={action_idx} out of range "
				f"(agent {agent_id} action_dim={self.action_dims[agent_id]})"
			)
