# -*- coding: utf-8 -*-
"""
快照组装器 -- 聚合所有提取器的数据为统一快照。

线程安全：所有 OpenDSS 访问通过全局 _snapshot_lock 保护，
防止 Gradio 异步事件处理器从不同线程并发访问 DSS 引擎。
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from envs.district_dispatch.render.data.bus_data_extractor import BusDataExtractor
from envs.district_dispatch.render.data.circuit_data_extractor import CircuitDataExtractor
from envs.district_dispatch.render.data.device_data_extractor import DeviceDataExtractor
from envs.district_dispatch.render.data.line_data_extractor import LineDataExtractor
from envs.district_dispatch.render.data.regulator_data_extractor import RegulatorDataExtractor
from envs.district_dispatch.render.data.transformer_data_extractor import TransformerDataExtractor

logger = logging.getLogger(__name__)

# 全局快照锁: OpenDSS 不是线程安全的，
# 整个提取过程必须原子化执行
_snapshot_lock = threading.Lock()


class SnapshotAssembler:
	"""快照组装器 -- 聚合所有提取器的数据为统一快照

	线程安全：所有 OpenDSS 访问通过 _snapshot_lock 保护。
	在一次 take_snapshot 调用中，6 个提取器依次执行，
	期间不释放锁，确保数据一致性。

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine
		self.bus_extractor = BusDataExtractor(dss_engine)
		self.line_extractor = LineDataExtractor(dss_engine)
		self.device_extractor = DeviceDataExtractor(dss_engine)
		self.transformer_extractor = TransformerDataExtractor(dss_engine)
		self.regulator_extractor = RegulatorDataExtractor(dss_engine)
		self.circuit_extractor = CircuitDataExtractor(dss_engine)

		# 快照计数器
		self._snapshot_count = 0

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray] = None,
		rewards: Optional[np.ndarray] = None,
		reward_components: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""拍摄当前时刻的完整系统快照

		在全局锁保护下依次调用 6 个提取器，
		收集 OpenDSS 电路的全部运行时数据。

		参数:
			step: 当前仿真步数 (0-based)
			actions: 智能体动作数组 (可选)
			rewards: 智能体奖励数组 (可选)
			reward_components: 奖励分量字典 (可选)

		返回:
			{
				step:              int,
				timestamp_h:       float,   -- 仿真时间 (h)，15 分钟分辨率
				snapshot_id:       int,     -- 快照序号
				wall_time:         float,   -- 拍摄耗时 (ms)
				buses:             Dict,    -- 母线数据
				lines:             Dict,    -- 线路数据
				devices:           Dict,    -- 设备数据 {pv, storage, ev}
				transformers:      Dict,    -- 变压器数据
				regulators:        Dict,    -- 调压器数据
				circuit:           Dict,    -- 系统级数据
				actions:           Optional[List],
				rewards:           Optional[List],
				reward_components: Optional[Dict],
			}
		"""
		with _snapshot_lock:
			t0 = time.perf_counter()
			self._snapshot_count += 1

			snapshot: Dict[str, Any] = {
				"step": step,
				"timestamp_h": step * 0.25,
				"snapshot_id": self._snapshot_count,
			}

			# -- 依次提取 --
			snapshot["buses"] = self._safe_extract(
				self.bus_extractor.extract_all, "buses"
			)
			snapshot["lines"] = self._safe_extract(
				self.line_extractor.extract_all, "lines"
			)
			snapshot["devices"] = self._safe_extract(
				self.device_extractor.extract_all, "devices"
			)
			snapshot["transformers"] = self._safe_extract(
				self.transformer_extractor.extract_all, "transformers"
			)
			snapshot["regulators"] = self._safe_extract(
				self.regulator_extractor.extract_all, "regulators"
			)
			snapshot["circuit"] = self._safe_extract(
				self.circuit_extractor.extract_all, "circuit"
			)

			# -- RL 附加数据 --
			snapshot["actions"] = (
				actions.tolist()
				if isinstance(actions, np.ndarray)
				else actions
			)
			snapshot["rewards"] = (
				rewards.tolist()
				if isinstance(rewards, np.ndarray)
				else rewards
			)
			snapshot["reward_components"] = reward_components

			# -- 计时 --
			elapsed_ms = (time.perf_counter() - t0) * 1000.0
			snapshot["wall_time_ms"] = round(elapsed_ms, 2)

			logger.debug(
				f"快照 #{self._snapshot_count} (step={step}) "
				f"耗时 {elapsed_ms:.1f}ms"
			)

			return snapshot

	def take_lightweight_snapshot(self, step: int) -> Dict[str, Any]:
		"""拍摄轻量级快照 -- 仅含系统级数据和电压摘要

		适用于高频监控场景，避免逐元件遍历的开销。

		参数:
			step: 当前仿真步数

		返回:
			{step, timestamp_h, circuit, voltage_summary}
		"""
		with _snapshot_lock:
			t0 = time.perf_counter()

			snapshot: Dict[str, Any] = {
				"step": step,
				"timestamp_h": step * 0.25,
			}

			snapshot["circuit"] = self._safe_extract(
				self.circuit_extractor.extract_all, "circuit"
			)
			snapshot["voltage_summary"] = self._safe_extract(
				self.bus_extractor.get_voltage_summary, "voltage_summary"
			)

			elapsed_ms = (time.perf_counter() - t0) * 1000.0
			snapshot["wall_time_ms"] = round(elapsed_ms, 2)

			return snapshot

	@property
	def snapshot_count(self) -> int:
		"""已拍摄的快照总数"""
		return self._snapshot_count

	def reset_counter(self) -> None:
		"""重置快照计数器（通常在 env.reset() 时调用）"""
		self._snapshot_count = 0

	# ------------------------------------------------------------------
	# 内部实现
	# ------------------------------------------------------------------

	@staticmethod
	def _safe_extract(extractor_fn, label: str) -> Any:
		"""安全执行提取器函数

		参数:
			extractor_fn: 提取器的 extract_all 方法
			label: 数据标签（用于日志）

		返回:
			提取结果，异常时返回空字典
		"""
		try:
			return extractor_fn()
		except Exception as exc:
			logger.error(
				f"提取 {label} 数据失败: {exc}", exc_info=True
			)
			return {}
