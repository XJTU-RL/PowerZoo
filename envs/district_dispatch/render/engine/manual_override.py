# -*- coding: utf-8 -*-
"""
手动动作覆盖管理器

允许用户在 Live Inference 模式下手动修改部分智能体的动作。
覆盖粒度为单个 agent 的单个动作维度，可随时开关。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 台区调度环境的动作语义标签
ACTION_LABELS = {
	"exchange_p": "Active Power Exchange",
	"exchange_q": "Reactive Power Exchange",
	"pv_curtail": "PV Curtailment",
	"storage_cmd": "Storage Command (-1=discharge, 1=charge)",
	"ev_mod": "EV Modulation",
}


class ManualOverride:
	"""手动动作覆盖管理器

	在 Live Inference 模式下拦截模型输出的动作，
	将用户手动指定的值覆盖到对应的 agent/action_idx 上。
	未覆盖的维度保持模型原始输出不变。

	参数:
		n_agents: 智能体数量
		action_dims: 每个智能体的动作维度列表
	"""

	def __init__(self, n_agents: int, action_dims: List[int]):
		self.n_agents = n_agents
		self.action_dims = action_dims
		# {agent_id: {action_idx: override_value}}
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
		if not val:
			logger.debug("ManualOverride 已禁用")
		else:
			logger.debug("ManualOverride 已启用")

	def set_override(
		self,
		agent_id: int,
		action_idx: int,
		value: float,
	) -> None:
		"""设置单个动作覆盖

		参数:
			agent_id: 智能体索引
			action_idx: 动作维度索引
			value: 覆盖值 (应在动作空间范围内, 通常 [-1, 1])

		异常:
			ValueError: agent_id 或 action_idx 越界
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

		logger.debug(
			f"设置覆盖: agent={agent_id}, "
			f"idx={action_idx}, value={value:.4f}"
		)

	def clear_override(self, agent_id: int, action_idx: int) -> None:
		"""清除单个动作覆盖

		参数:
			agent_id: 智能体索引
			action_idx: 动作维度索引
		"""
		self._validate_indices(agent_id, action_idx)

		if agent_id in self._overrides:
			self._overrides[agent_id].pop(action_idx, None)
			if not self._overrides[agent_id]:
				del self._overrides[agent_id]

		self._history.append({
			"action": "clear",
			"agent_id": agent_id,
			"action_idx": action_idx,
		})

	def clear_agent(self, agent_id: int) -> None:
		"""清除指定智能体的所有覆盖

		参数:
			agent_id: 智能体索引
		"""
		if agent_id < 0 or agent_id >= self.n_agents:
			raise ValueError(
				f"agent_id={agent_id} 越界 (n_agents={self.n_agents})"
			)

		removed = self._overrides.pop(agent_id, {})
		if removed:
			self._history.append({
				"action": "clear_agent",
				"agent_id": agent_id,
				"cleared_count": len(removed),
			})

	def clear_all(self) -> None:
		"""清除所有覆盖"""
		count = sum(len(v) for v in self._overrides.values())
		self._overrides.clear()

		if count > 0:
			self._history.append({
				"action": "clear_all",
				"cleared_count": count,
			})
			logger.debug(f"已清除全部 {count} 个覆盖")

	def apply(self, actions: np.ndarray) -> np.ndarray:
		"""将覆盖应用到模型输出的动作上

		若覆盖未启用或无覆盖项，直接返回原始动作的拷贝。

		参数:
			actions: 模型输出的动作 shape=(n_agents, max_action_dim)

		返回:
			modified_actions: 应用覆盖后的动作 (新数组)
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

		返回:
			摘要字典，包含:
			- enabled: 是否启用
			- total_overrides: 覆盖总数
			- agents: 各 agent 的覆盖详情
			- history_length: 历史操作数
		"""
		agents_detail: Dict[str, Any] = {}

		for agent_id, idx_map in self._overrides.items():
			labeled_overrides = {}
			for action_idx, value in idx_map.items():
				label = self._get_action_label(agent_id, action_idx)
				labeled_overrides[f"idx_{action_idx} ({label})"] = round(
					value, 4
				)
			agents_detail[f"agent_{agent_id}"] = labeled_overrides

		return {
			"enabled": self._enabled,
			"total_overrides": sum(
				len(v) for v in self._overrides.values()
			),
			"agents": agents_detail,
			"history_length": len(self._history),
		}

	def get_action_labels(self, agent_id: int) -> List[str]:
		"""获取指定智能体的动作维度语义标签

		基于台区调度环境的动作向量布局生成标签。
		布局: [exchange_p * n_nb, exchange_q * n_nb,
			   pv_curtail * n_pv, storage_cmd * n_st, ev_mod * n_ev]

		参数:
			agent_id: 智能体索引

		返回:
			动作维度标签列表
		"""
		if agent_id < 0 or agent_id >= self.n_agents:
			return []

		act_dim = self.action_dims[agent_id]
		labels: List[str] = []

		# NOTE: 这里假设环境的标准动作布局。
		# 实际布局取决于 config 中每个台区的设备数量和邻居数。
		# 此方法提供通用的索引标签。
		for idx in range(act_dim):
			labels.append(f"action_{idx}")

		return labels

	# ------------------------------------------------------------------
	# 内部方法
	# ------------------------------------------------------------------

	def _validate_indices(self, agent_id: int, action_idx: int) -> None:
		"""校验智能体和动作索引的合法性

		参数:
			agent_id: 智能体索引
			action_idx: 动作维度索引

		异常:
			ValueError: 索引越界
		"""
		if agent_id < 0 or agent_id >= self.n_agents:
			raise ValueError(
				f"agent_id={agent_id} 越界 "
				f"(n_agents={self.n_agents})"
			)
		if action_idx < 0 or action_idx >= self.action_dims[agent_id]:
			raise ValueError(
				f"action_idx={action_idx} 越界 "
				f"(agent {agent_id} action_dim={self.action_dims[agent_id]})"
			)

	@staticmethod
	def _get_action_label(agent_id: int, action_idx: int) -> str:
		"""获取动作维度的可读标签

		参数:
			agent_id: 智能体索引
			action_idx: 动作维度索引

		返回:
			人类可读的动作标签
		"""
		# 通用标签方案 -- 具体语义需结合环境配置
		label_keys = list(ACTION_LABELS.keys())
		if action_idx < len(label_keys):
			key = label_keys[action_idx]
			return ACTION_LABELS[key]
		return f"dim_{action_idx}"
