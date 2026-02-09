"""
Base Episode Runner (Abstract)
Episode 运行器抽象基类

定义环境无关的 episode 运行框架，子类需实现:
- _create_env(): 创建环境实例
- _env_step(): 执行单步
- _take_snapshot(): 收集快照
- _random_actions(): 生成随机动作
- _build_config_summary(): 构建配置摘要
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class BaseEpisodeRunner(ABC):
	"""Episode 运行器抽象基类

	提供 run_episode / run_step / close 的标准流程，
	子类只需实现环境创建和快照收集的钩子方法。

	Args:
		config: 环境配置对象 (类型取决于子类)
		inference_engine: 推理引擎
		collect_snapshots: 是否收集快照
	"""

	def __init__(
		self,
		config: Any,
		inference_engine: Optional[InferenceEngine] = None,
		collect_snapshots: bool = True,
	):
		self.config = config
		self.inference_engine = inference_engine
		self.collect_snapshots = collect_snapshots

		self._env: Any = None
		self._current_obs: Optional[List[np.ndarray]] = None
		self._current_share_obs: Optional[List[np.ndarray]] = None
		self._current_avail_actions: Optional[List[Optional[np.ndarray]]] = None
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

		Args:
			seed: 随机种子
			max_steps: 最大步数
			action_overrides: {step_idx: override_actions}
			step_callback: 每步回调 fn(step, snapshot)

		Returns:
			EpisodeData 包含所有快照和元数据
		"""
		t_start = time.time()
		max_steps = max_steps or self._get_max_steps()
		action_overrides = action_overrides or {}

		self._init_env(seed)

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

		# 初始快照 (step=0)
		if self.collect_snapshots:
			init_snapshot = self._take_snapshot(step=0, actions=None, rewards=None, infos=None)
			episode_data.snapshots.append(init_snapshot)
			if step_callback:
				step_callback(0, init_snapshot)

		# 主循环
		for step in range(1, max_steps + 1):
			self._current_step = step

			if step in action_overrides:
				actions = action_overrides[step]
			elif self.inference_engine is not None:
				actions = self.inference_engine.infer(
					self._current_obs,
					deterministic=True,
					available_actions=self._current_avail_actions,
				)
			else:
				actions = self._random_actions()

			obs, share_obs, rewards, dones, infos, avail_actions = self._env_step(actions)
			self._current_obs = obs
			self._current_share_obs = share_obs
			self._current_avail_actions = avail_actions

			step_reward = float(np.sum(rewards))
			self._cumulative_reward += step_reward

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

			if np.all(dones):
				self._episode_done = True
				break

		episode_data.total_reward = self._cumulative_reward
		episode_data.episode_length = self._current_step
		episode_data.metadata["elapsed_seconds"] = round(
			time.time() - t_start, 3
		)

		logger.info(
			f"Episode done: {self._current_step} steps, "
			f"total_reward={self._cumulative_reward:.4f}, "
			f"elapsed={episode_data.metadata['elapsed_seconds']}s"
		)

		return episode_data

	def run_step(
		self,
		step: int,
		actions: Optional[np.ndarray] = None,
	) -> Dict[str, Any]:
		"""运行单步 (Live Inference 模式)

		Args:
			step: 当前步编号
			actions: 外部提供的动作

		Returns:
			当前步的快照字典
		"""
		if self._env is None:
			self._init_env(seed=None)
			if self.inference_engine is not None:
				self.inference_engine.reset_rnn_states()

		if self._episode_done:
			logger.warning("Episode already done, run_step is no-op")
			return {"step": step, "done": True}

		self._current_step = step

		if actions is None:
			if self.inference_engine is not None:
				actions = self.inference_engine.infer(
					self._current_obs,
					deterministic=True,
					available_actions=self._current_avail_actions,
				)
			else:
				actions = self._random_actions()

		obs, share_obs, rewards, dones, infos, avail_actions = self._env_step(actions)
		self._current_obs = obs
		self._current_share_obs = share_obs
		self._current_avail_actions = avail_actions

		step_reward = float(np.sum(rewards))
		self._cumulative_reward += step_reward

		if np.all(dones):
			self._episode_done = True

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
		if self._env is not None:
			try:
				self._env.close()
			except Exception:
				pass
			self._env = None
		self._current_obs = None
		self._current_share_obs = None
		self._current_avail_actions = None
		logger.debug("EpisodeRunner resources released")

	# ------------------------------------------------------------------
	# 抽象方法 (子类必须实现)
	# ------------------------------------------------------------------

	@abstractmethod
	def _create_env(self, seed: Optional[int] = None) -> Any:
		"""创建环境实例并执行 reset

		Args:
			seed: 随机种子

		Returns:
			环境实例
		"""
		...

	@abstractmethod
	def _env_step(self, actions: np.ndarray) -> tuple:
		"""执行环境单步

		Args:
			actions: 动作数组

		Returns:
			(obs, share_obs, rewards, dones, infos, avail_actions)
		"""
		...

	@abstractmethod
	def _take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
	) -> Dict[str, Any]:
		"""收集当前步的完整快照

		Args:
			step: 步编号
			actions: 动作
			rewards: 奖励
			infos: 环境信息

		Returns:
			快照字典
		"""
		...

	@abstractmethod
	def _random_actions(self) -> np.ndarray:
		"""生成随机动作

		Returns:
			np.ndarray(n_agents, action_dim)
		"""
		...

	@abstractmethod
	def _build_config_summary(self) -> Dict[str, Any]:
		"""构建配置摘要

		Returns:
			配置摘要字典
		"""
		...

	# ------------------------------------------------------------------
	# 可选覆写方法
	# ------------------------------------------------------------------

	def _get_max_steps(self) -> int:
		"""获取默认最大步数，子类可覆写"""
		if hasattr(self.config, "max_episode_steps"):
			return self.config.max_episode_steps
		if hasattr(self.config, "episode_length"):
			return self.config.episode_length
		return 24

	def _init_env(self, seed: Optional[int] = None) -> None:
		"""初始化环境，调用 _create_env 并保存状态"""
		if self._env is not None:
			try:
				self._env.close()
			except Exception:
				pass

		self._env = self._create_env(seed)
		self._current_step = 0
		self._episode_done = False
		self._cumulative_reward = 0.0
