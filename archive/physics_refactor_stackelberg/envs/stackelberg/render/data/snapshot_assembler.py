# -*- coding: utf-8 -*-
"""
快照组装器

将母线、线路、设备、电路级和市场数据组装为
统一的快照字典格式，供可视化层消费。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.stackelberg.render.data.bus_data_extractor import BusDataExtractor
from envs.stackelberg.render.data.circuit_data_extractor import CircuitDataExtractor
from envs.stackelberg.render.data.device_data_extractor import DeviceDataExtractor
from envs.stackelberg.render.data.line_data_extractor import LineDataExtractor
from envs.stackelberg.render.data.market_data_extractor import MarketDataExtractor

logger = logging.getLogger(__name__)


class SnapshotAssembler:
	"""快照组装器

	协调各数据提取器，在每个时间步收集完整快照。

	Args:
		dss: dss-python DSS 引擎实例
		n_consumers: Consumer 数量
	"""

	def __init__(self, dss: Any, n_consumers: int = 1):
		self._bus_extractor = BusDataExtractor(dss)
		self._line_extractor = LineDataExtractor(dss)
		self._device_extractor = DeviceDataExtractor(dss)
		self._circuit_extractor = CircuitDataExtractor(dss)
		self._market_extractor = MarketDataExtractor(n_consumers=n_consumers)

	def assemble(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
		env_state: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""组装完整快照。

		Args:
			step: 当前时间步
			actions: 动作数组 (n_agents, action_dim)
			rewards: 奖励数组 (n_agents, 1) or (n_agents,)
			infos: 环境 info
			env_state: 环境内部状态

		Returns:
			快照字典
		"""
		snapshot: Dict[str, Any] = {"step": step}

		# 电路级数据
		try:
			snapshot["circuit"] = self._circuit_extractor.extract()
		except Exception as exc:
			logger.warning(f"Step {step} circuit extraction failed: {exc}")
			snapshot["circuit"] = {}

		# 母线数据
		try:
			snapshot["buses"] = self._bus_extractor.extract_all()
			snapshot["voltage_summary"] = self._bus_extractor.get_voltage_summary()
		except Exception as exc:
			logger.warning(f"Step {step} bus extraction failed: {exc}")
			snapshot["buses"] = {}
			snapshot["voltage_summary"] = {}

		# 线路数据
		try:
			snapshot["lines"] = self._line_extractor.extract_all()
		except Exception as exc:
			logger.warning(f"Step {step} line extraction failed: {exc}")
			snapshot["lines"] = {}

		# 设备数据
		try:
			snapshot["devices"] = self._device_extractor.extract_all()
		except Exception as exc:
			logger.warning(f"Step {step} device extraction failed: {exc}")
			snapshot["devices"] = {}

		# 市场数据 (Stackelberg 独有)
		try:
			snapshot["market_data"] = self._market_extractor.extract_from_snapshot(
				step=step,
				actions=actions,
				infos=infos,
				env_state=env_state,
			)
		except Exception as exc:
			logger.warning(f"Step {step} market data extraction failed: {exc}")
			snapshot["market_data"] = {}

		# 动作和奖励
		if actions is not None:
			snapshot["actions"] = actions.tolist() if hasattr(actions, "tolist") else list(actions)
		if rewards is not None:
			snapshot["rewards"] = rewards.tolist() if hasattr(rewards, "tolist") else list(rewards)

		# 智能体分类奖励 (UC vs Consumers)
		if rewards is not None:
			rewards_arr = np.array(rewards).flatten()
			if len(rewards_arr) > 0:
				snapshot["uc_reward"] = float(rewards_arr[0])
				if len(rewards_arr) > 1:
					consumer_rewards = rewards_arr[1:]
					snapshot["avg_consumer_reward"] = float(np.mean(consumer_rewards))
					snapshot["consumer_rewards"] = consumer_rewards.tolist()

		return snapshot
