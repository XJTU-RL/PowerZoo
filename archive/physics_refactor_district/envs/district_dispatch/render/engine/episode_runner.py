# -*- coding: utf-8 -*-
"""
Episode 运行器

驱动 DistrictDispatchCore 运行完整 episode，
每步通过 BusDataExtractor 收集 OpenDSS 电路快照。
支持推理引擎驱动、随机动作、手动覆盖三种模式。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from envs.district_dispatch.core.config import DistrictDispatchConfig
from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore
from envs.district_dispatch.render.data.bus_data_extractor import BusDataExtractor
from envs.district_dispatch.render.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
	"""Episode 数据容器

	存储一个完整 episode 的所有快照、奖励和元数据。
	snapshots 列表中每个元素对应一个 step 的完整状态。
	"""
	snapshots: List[Dict[str, Any]] = field(default_factory=list)
	total_reward: float = 0.0
	episode_length: int = 0
	config_summary: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)

	def get_reward_curve(self) -> List[float]:
		"""提取每步奖励序列

		返回:
			每步总奖励 (所有 agent 之和) 的列表
		"""
		return [
			snap.get("step_reward", 0.0)
			for snap in self.snapshots
			if snap.get("step", -1) > 0
		]

	def get_voltage_curve(self, bus_name: str) -> List[float]:
		"""提取指定母线的电压时间序列

		参数:
			bus_name: 母线名称

		返回:
			该母线各步的平均 pu 电压列表
		"""
		curve = []
		for snap in self.snapshots:
			bus_data = snap.get("bus_data", {}).get(bus_name, {})
			curve.append(bus_data.get("v_mean", 1.0))
		return curve


class EpisodeRunner:
	"""Episode 运行器

	驱动 DistrictDispatchCore 运行完整 episode，
	每步收集 OpenDSS 快照。支持三种动作来源:
	1. InferenceEngine 推理
	2. 随机采样
	3. 外部手动覆盖

	参数:
		config: 环境配置
		inference_engine: 推理引擎 (None 则使用随机动作)
		collect_snapshots: 是否收集 OpenDSS 快照
	"""

	def __init__(
		self,
		config: DistrictDispatchConfig,
		inference_engine: Optional[InferenceEngine] = None,
		collect_snapshots: bool = True,
	):
		self.config = config
		self.inference_engine = inference_engine
		self.collect_snapshots = collect_snapshots

		# 环境核心和快照提取器在 run_episode 时惰性初始化
		self._core: Optional[DistrictDispatchCore] = None
		self._bus_extractor: Optional[BusDataExtractor] = None

		# 当前 episode 状态
		self._current_obs: Optional[List[np.ndarray]] = None
		self._current_step = 0
		self._episode_done = False
		self._cumulative_reward = 0.0

	def run_episode(
		self,
		seed: Optional[int] = None,
		max_steps: Optional[int] = None,
		action_overrides: Optional[Dict[int, np.ndarray]] = None,
		step_callback: Optional[Callable] = None,
	) -> EpisodeData:
		"""运行完整 episode

		参数:
			seed: 随机种子
			max_steps: 最大步数 (默认用 config 的 max_episode_steps)
			action_overrides: {step_idx: override_actions} 手动覆盖动作
			step_callback: 每步回调 fn(step, snapshot) 用于实时更新

		返回:
			EpisodeData 包含所有快照和元数据
		"""
		t_start = time.time()
		max_steps = max_steps or self.config.max_episode_steps
		action_overrides = action_overrides or {}

		# 初始化环境
		self._init_core(seed)

		# 重置推理引擎的 RNN 隐状态
		if self.inference_engine is not None:
			self.inference_engine.reset_rnn_states()

		episode_data = EpisodeData(
			config_summary=self._build_config_summary(),
			metadata={
				"seed": seed,
				"max_steps": max_steps,
				"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
				"has_inference_engine": self.inference_engine is not None,
			},
		)

		# 收集初始快照 (step=0)
		if self.collect_snapshots:
			init_snapshot = self._take_snapshot(step=0, actions=None, rewards=None, infos=None)
			episode_data.snapshots.append(init_snapshot)
			if step_callback:
				step_callback(0, init_snapshot)

		# 主循环
		for step in range(1, max_steps + 1):
			self._current_step = step

			# 确定动作
			if step in action_overrides:
				actions = action_overrides[step]
			elif self.inference_engine is not None:
				actions = self.inference_engine.infer(
					self._current_obs, deterministic=True,
				)
			else:
				actions = self._random_actions()

			# 执行单步
			obs, share_obs, rewards, dones, infos = self._core.step(actions)
			self._current_obs = obs

			step_reward = float(np.sum(rewards))
			self._cumulative_reward += step_reward

			# 收集快照
			snapshot = None
			if self.collect_snapshots:
				snapshot = self._take_snapshot(
					step=step, actions=actions,
					rewards=rewards, infos=infos,
				)
				snapshot["step_reward"] = step_reward
				snapshot["cumulative_reward"] = self._cumulative_reward
				episode_data.snapshots.append(snapshot)

			if step_callback:
				step_callback(step, snapshot)

			# 终止判断
			if np.all(dones):
				self._episode_done = True
				break

		# 汇总
		episode_data.total_reward = self._cumulative_reward
		episode_data.episode_length = self._current_step
		episode_data.metadata["elapsed_seconds"] = round(
			time.time() - t_start, 3
		)

		logger.info(
			f"Episode 完成: {self._current_step} steps, "
			f"total_reward={self._cumulative_reward:.4f}, "
			f"elapsed={episode_data.metadata['elapsed_seconds']}s"
		)

		return episode_data

	def run_step(
		self,
		step: int,
		actions: Optional[np.ndarray] = None,
	) -> Dict[str, Any]:
		"""运行单步 (支持 Live Inference 模式)

		逐步执行环境，适用于交互式调试。
		首次调用前需先调用 run_episode 的初始化逻辑，
		或手动调用 _init_core()。

		参数:
			step: 当前步编号
			actions: 外部提供的动作 (None 则通过推理引擎或随机生成)

		返回:
			当前步的快照字典
		"""
		if self._core is None:
			self._init_core(seed=None)
			if self.inference_engine is not None:
				self.inference_engine.reset_rnn_states()

		if self._episode_done:
			logger.warning("Episode 已结束，run_step 无效")
			return {"step": step, "done": True}

		self._current_step = step

		# 确定动作
		if actions is None:
			if self.inference_engine is not None:
				actions = self.inference_engine.infer(
					self._current_obs, deterministic=True,
				)
			else:
				actions = self._random_actions()

		# 执行
		obs, share_obs, rewards, dones, infos = self._core.step(actions)
		self._current_obs = obs

		step_reward = float(np.sum(rewards))
		self._cumulative_reward += step_reward

		if np.all(dones):
			self._episode_done = True

		# 快照
		snapshot = self._take_snapshot(
			step=step, actions=actions,
			rewards=rewards, infos=infos,
		)
		snapshot["step_reward"] = step_reward
		snapshot["cumulative_reward"] = self._cumulative_reward
		snapshot["done"] = self._episode_done

		return snapshot

	def close(self) -> None:
		"""释放资源"""
		if self._core is not None:
			self._core.close()
			self._core = None
		self._bus_extractor = None
		self._current_obs = None
		logger.debug("EpisodeRunner 资源已释放")

	# ------------------------------------------------------------------
	# 内部方法
	# ------------------------------------------------------------------

	def _init_core(self, seed: Optional[int] = None) -> None:
		"""初始化环境核心

		创建 DistrictDispatchCore 并执行 reset，
		同时获取 DSS 引擎用于构建 BusDataExtractor。

		参数:
			seed: 可选随机种子
		"""
		# 清理旧实例
		if self._core is not None:
			self._core.close()

		self._core = DistrictDispatchCore(self.config)
		obs, share_obs = self._core.reset(seed=seed)
		self._current_obs = obs
		self._current_step = 0
		self._episode_done = False
		self._cumulative_reward = 0.0

		# 从核心环境获取同一个 DSS 引擎实例
		if self.collect_snapshots:
			self._bus_extractor = BusDataExtractor(self._core.circuit.dss)

	def _take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Optional[List[Dict[str, Any]]],
	) -> Dict[str, Any]:
		"""收集当前步的完整快照

		参数:
			step: 步编号
			actions: 本步动作
			rewards: 本步奖励
			infos: 本步环境信息

		返回:
			快照字典，包含母线数据、台区状态、动作、奖励等
		"""
		snapshot: Dict[str, Any] = {"step": step}

		# 母线电压数据 (通过 BusDataExtractor)
		if self._bus_extractor is not None:
			try:
				snapshot["bus_data"] = self._bus_extractor.extract_all()
				snapshot["voltage_summary"] = (
					self._bus_extractor.get_voltage_summary()
				)
			except Exception as exc:
				logger.warning(f"Step {step} 快照提取失败: {exc}")
				snapshot["bus_data"] = {}
				snapshot["voltage_summary"] = {}

		# 台区状态
		if self._core is not None:
			snapshot["districts"] = []
			for d in self._core.districts:
				v_min, v_max, v_mean = d.get_voltage_stats()
				snapshot["districts"].append({
					"district_id": d.district_id,
					"name": d.name,
					"v_min": v_min,
					"v_max": v_max,
					"v_mean": v_mean,
					"net_load": d.get_net_load(),
					"exchange_in_kw": d.exchange_in_kw,
					"exchange_out_kw": d.exchange_out_kw,
				})

		# 动作和奖励
		if actions is not None:
			snapshot["actions"] = actions.tolist()
		if rewards is not None:
			snapshot["rewards"] = rewards.tolist()
		if infos is not None:
			# 过滤掉不可序列化的对象
			snapshot["infos"] = [
				{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, dict))}
				for info in infos
			]

		return snapshot

	def _random_actions(self) -> np.ndarray:
		"""生成随机动作

		返回:
			np.ndarray(n_agents, max_action_dim), 值域 [-1, 1]
		"""
		max_action_dim = self.config.get_max_action_dim()
		return np.random.uniform(
			-1.0, 1.0,
			size=(self._core.n_agents, max_action_dim),
		).astype(np.float32)

	def _build_config_summary(self) -> Dict[str, Any]:
		"""构建配置摘要

		返回:
			环境配置的关键参数摘要
		"""
		return {
			"env_name": self.config.env_name,
			"system_name": self.config.system_name,
			"n_districts": self.config.n_districts,
			"max_episode_steps": self.config.max_episode_steps,
			"max_action_dim": self.config.get_max_action_dim(),
			"max_obs_dim": self.config.get_max_obs_dim(),
			"share_obs_dim": self.config.get_share_obs_dim(),
			"v_limits": [self.config.v_min, self.config.v_max],
		}
