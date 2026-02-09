# -*- coding: utf-8 -*-
"""
快照组装器

将 BusDataExtractor、LineDataExtractor、DeviceDataExtractor、
CircuitDataExtractor 的输出组装为统一的快照字典格式。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.vvc.render.data.bus_data_extractor import BusDataExtractor
from envs.vvc.render.data.circuit_data_extractor import CircuitDataExtractor
from envs.vvc.render.data.device_data_extractor import DeviceDataExtractor
from envs.vvc.render.data.line_data_extractor import LineDataExtractor

logger = logging.getLogger(__name__)


class SnapshotAssembler:
	"""快照组装器

	持有四个子提取器的引用，调用 assemble() 输出完整快照。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any) -> None:
		self._bus_extractor = BusDataExtractor(dss)
		self._line_extractor = LineDataExtractor(dss)
		self._device_extractor = DeviceDataExtractor(dss)
		self._circuit_extractor = CircuitDataExtractor(dss)

	def assemble(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
	) -> Dict[str, Any]:
		"""组装当前步的完整快照。

		Args:
			step: 当前时间步编号
			actions: 智能体动作数组
			rewards: 智能体奖励数组
			infos: 环境返回的 info 列表

		Returns:
			快照字典，包含 buses, lines, devices, circuit, actions, rewards 等
		"""
		snapshot: Dict[str, Any] = {"step": step}

		# 母线数据
		try:
			snapshot["buses"] = self._bus_extractor.extract_all()
		except Exception as exc:
			logger.warning(f"Step {step}: bus extraction failed: {exc}")
			snapshot["buses"] = {}

		# 线路数据
		try:
			snapshot["lines"] = self._line_extractor.extract_all()
		except Exception as exc:
			logger.warning(f"Step {step}: line extraction failed: {exc}")
			snapshot["lines"] = {}

		# 设备数据
		try:
			snapshot["devices"] = self._device_extractor.extract_all()
		except Exception as exc:
			logger.warning(f"Step {step}: device extraction failed: {exc}")
			snapshot["devices"] = {}

		# 系统级数据
		try:
			snapshot["circuit"] = self._circuit_extractor.extract()
		except Exception as exc:
			logger.warning(f"Step {step}: circuit extraction failed: {exc}")
			snapshot["circuit"] = {}

		# 动作和奖励
		if actions is not None:
			snapshot["actions"] = (
				actions.tolist() if isinstance(actions, np.ndarray) else actions
			)

		if rewards is not None:
			snapshot["rewards"] = (
				rewards.tolist() if isinstance(rewards, np.ndarray) else rewards
			)

		# 奖励分量 (从 infos 中提取)
		reward_components = self._extract_reward_components(infos)
		if reward_components:
			snapshot["reward_components"] = reward_components

		return snapshot

	@staticmethod
	def _extract_reward_components(infos: Any) -> Dict[str, Any]:
		"""从环境 infos 中提取奖励分量。

		Args:
			infos: 环境 step() 返回的 info 列表

		Returns:
			{power_loss, voltage_violation, control_cost, ...}
		"""
		if infos is None:
			return {}

		# infos 是 n_agents 个相同 dict 的列表
		info = infos[0] if isinstance(infos, list) and infos else infos
		if not isinstance(info, dict):
			return {}

		components: Dict[str, Any] = {}

		# 常见的奖励分量键
		reward_keys = [
			"power_loss", "power_loss_ratio",
			"voltage_violation", "voltage_penalty",
			"control_cost", "switching_cost",
			"reward_breakdown",
		]

		for key in reward_keys:
			if key in info:
				components[key] = info[key]

		# 如果有 reward_breakdown 字典，展开
		breakdown = info.get("reward_breakdown", {})
		if isinstance(breakdown, dict):
			for k, v in breakdown.items():
				if k not in components:
					components[k] = v

		return components
