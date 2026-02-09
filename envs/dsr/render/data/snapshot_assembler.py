"""
DSR Snapshot Assembler
快照组装器

将 Bus / Line / Device / Circuit / Restoration 数据提取器的输出
组装为单个时间步的完整快照字典，供可视化层使用。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.dsr.render.data.bus_data_extractor import BusDataExtractor
from envs.dsr.render.data.line_data_extractor import LineDataExtractor
from envs.dsr.render.data.device_data_extractor import DeviceDataExtractor
from envs.dsr.render.data.circuit_data_extractor import CircuitDataExtractor
from envs.dsr.render.data.restoration_data_extractor import RestorationDataExtractor

logger = logging.getLogger(__name__)


class SnapshotAssembler:
	"""DSR 快照组装器

	将各数据提取器的输出组合为统一的快照格式。

	Args:
		env: DSREnv 实例
	"""

	def __init__(self, env: Any):
		self.env = env
		self.bus_extractor = BusDataExtractor(env)
		self.line_extractor = LineDataExtractor(env)
		self.device_extractor = DeviceDataExtractor(env)
		self.circuit_extractor = CircuitDataExtractor(env)
		self.restoration_extractor = RestorationDataExtractor(env)

	def assemble(
		self,
		step: int,
		actions: Optional[np.ndarray] = None,
		rewards: Optional[np.ndarray] = None,
		infos: Any = None,
		available_actions: Optional[List] = None,
	) -> Dict[str, Any]:
		"""组装完整快照

		Args:
			step: 当前时间步
			actions: 智能体动作
			rewards: 智能体奖励
			infos: 环境信息字典
			available_actions: 可用动作掩码 (DSR 关键数据)

		Returns:
			完整快照字典
		"""
		snapshot: Dict[str, Any] = {
			"step": step,
		}

		# 母线数据
		try:
			snapshot["buses"] = self.bus_extractor.extract()
		except Exception as exc:
			logger.warning(f"Bus extraction failed at step {step}: {exc}")
			snapshot["buses"] = {}

		# 线路数据
		try:
			snapshot["lines"] = self.line_extractor.extract()
		except Exception as exc:
			logger.warning(f"Line extraction failed at step {step}: {exc}")
			snapshot["lines"] = {}

		# 设备数据
		try:
			snapshot["devices"] = self.device_extractor.extract()
		except Exception as exc:
			logger.warning(f"Device extraction failed at step {step}: {exc}")
			snapshot["devices"] = {"switches": {}, "pvs": {}, "loads": {}}

		# 电路汇总
		try:
			snapshot["circuit"] = self.circuit_extractor.extract()
		except Exception as exc:
			logger.warning(f"Circuit extraction failed at step {step}: {exc}")
			snapshot["circuit"] = {}

		# 恢复状态 (DSR 独有)
		try:
			snapshot["restoration_data"] = self.restoration_extractor.extract()
		except Exception as exc:
			logger.warning(f"Restoration extraction failed at step {step}: {exc}")
			snapshot["restoration_data"] = {}

		# 动作
		if actions is not None:
			if isinstance(actions, np.ndarray):
				snapshot["actions"] = actions.tolist()
			else:
				snapshot["actions"] = list(actions)

		# 奖励
		if rewards is not None:
			if isinstance(rewards, np.ndarray):
				snapshot["rewards"] = rewards.flatten().tolist()
			else:
				snapshot["rewards"] = list(rewards)

		# 奖励分量
		if infos is not None:
			snapshot["reward_components"] = self._extract_reward_components(infos)
			snapshot["infos_summary"] = self._summarize_infos(infos)

		# 动作掩码 (DSR 关键数据)
		if available_actions is not None:
			snapshot["available_actions"] = self._serialize_avail_actions(available_actions)

		return snapshot

	def _extract_reward_components(self, infos: Any) -> Dict[str, Any]:
		"""从 infos 中提取奖励分量

		Args:
			infos: 环境信息（可能是列表或字典）

		Returns:
			奖励分量字典
		"""
		components: Dict[str, Any] = {}

		if isinstance(infos, list) and infos:
			# 取第一个 agent 的 info
			info = infos[0] if isinstance(infos[0], dict) else {}
		elif isinstance(infos, dict):
			info = infos
		else:
			return components

		reward_keys = [
			"restore_reward", "voltage_penalty", "overload_penalty",
			"done_reward", "total_reward", "voltage_violations",
			"overload_count", "severe_overload_count", "max_overload_ratio",
		]
		for key in reward_keys:
			if key in info:
				components[key] = info[key]

		return components

	def _summarize_infos(self, infos: Any) -> Dict[str, Any]:
		"""汇总 infos 关键信息

		Args:
			infos: 环境信息

		Returns:
			汇总字典
		"""
		summary: Dict[str, Any] = {}

		if isinstance(infos, list) and infos:
			info = infos[0] if isinstance(infos[0], dict) else {}
		elif isinstance(infos, dict):
			info = infos
		else:
			return summary

		for key in ["termination_reason", "TimeLimit.truncated", "bad_transition"]:
			if key in info:
				summary[key] = info[key]

		return summary

	def _serialize_avail_actions(self, available_actions: List) -> List[List[int]]:
		"""序列化可用动作掩码

		Args:
			available_actions: 可用动作列表

		Returns:
			二维整数列表
		"""
		result: List[List[int]] = []
		for agent_actions in available_actions:
			if isinstance(agent_actions, np.ndarray):
				result.append(agent_actions.astype(int).tolist())
			elif isinstance(agent_actions, list):
				result.append([int(a) for a in agent_actions])
			else:
				result.append([])
		return result
