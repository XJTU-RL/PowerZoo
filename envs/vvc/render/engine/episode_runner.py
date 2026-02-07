# -*- coding: utf-8 -*-
"""
VVC Episode 运行器

继承 BaseEpisodeRunner，驱动 VVCEnv 运行完整 episode，
每步通过 SnapshotAssembler 收集 OpenDSS 电路快照。
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
	import gymnasium as gym
	from gymnasium.spaces import Box, Discrete
except ImportError:
	import gym
	from gym.spaces import Box, Discrete

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.vvc.render.data.snapshot_assembler import SnapshotAssembler

logger = logging.getLogger(__name__)


class VVCEpisodeRunner(BaseEpisodeRunner):
	"""VVC Episode 运行器

	驱动 VVCEnv 运行完整 episode，利用 SnapshotAssembler
	逐步收集母线电压、线路负载、设备状态等 OpenDSS 数据。

	Args:
		config: VVC 环境配置字典 (包含 env_name, seed 等)
		inference_engine: 推理引擎
		collect_snapshots: 是否收集 OpenDSS 快照
	"""

	def __init__(
		self,
		config: Dict[str, Any],
		inference_engine: Optional[InferenceEngine] = None,
		collect_snapshots: bool = True,
	):
		super().__init__(config, inference_engine, collect_snapshots)
		self._assembler: Optional[SnapshotAssembler] = None

	def _create_env(self, seed: Optional[int] = None) -> Any:
		"""创建 VVCEnv 实例并执行 reset。

		Args:
			seed: 随机种子

		Returns:
			VVCEnv 实例
		"""
		from envs.vvc.vvc_env import VVCEnv

		env_config = copy.deepcopy(self.config)
		if seed is not None:
			env_config["seed"] = seed

		# 确保必要的配置项存在
		env_config.setdefault("use_render", False)
		env_config.setdefault("useS", False)
		env_config.setdefault("record_node", False)

		env = VVCEnv(env_config, rank=0)
		obs, share_obs, avail_actions = env.reset()

		self._current_obs = obs
		self._current_share_obs = share_obs
		self._current_avail_actions = avail_actions

		# 创建快照组装器 (共享同一 DSS 引擎)
		if self.collect_snapshots and hasattr(env, "env") and hasattr(env.env, "dss"):
			try:
				self._assembler = SnapshotAssembler(env.env.dss)
			except Exception as exc:
				logger.warning(f"SnapshotAssembler init failed: {exc}")
				self._assembler = None

		return env

	def _env_step(self, actions: np.ndarray) -> tuple:
		"""执行 VVCEnv 单步。

		Args:
			actions: 动作数组 shape=(n_agents, max_action_dim)

		Returns:
			(obs, share_obs, rewards, dones, infos, avail_actions)
		"""
		# 将 (n_agents, action_dim) 转为扁平动作列表
		flat_actions = []
		for agent_id in range(self._env.n_agents):
			action_space = self._env.action_space[agent_id]
			if isinstance(action_space, Discrete):
				# 离散动作: 取第一维的值
				act_val = int(actions[agent_id, 0]) if actions.ndim > 1 else int(actions[agent_id])
				act_val = max(0, min(act_val, action_space.n - 1))
				flat_actions.append(act_val)
			elif isinstance(action_space, Box):
				# 连续动作: 取对应维度
				act_dim = action_space.shape[0]
				if actions.ndim > 1:
					act_val = actions[agent_id, :act_dim]
				else:
					act_val = actions[agent_id]
				flat_actions.append(act_val)
			else:
				flat_actions.append(actions[agent_id])

		obs, share_obs, rewards, dones, infos, avail_actions = self._env.step(flat_actions)

		return obs, share_obs, rewards, dones, infos, avail_actions

	def _take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
	) -> Dict[str, Any]:
		"""收集当前步的完整快照。

		Args:
			step: 时间步编号
			actions: 动作数组
			rewards: 奖励数组
			infos: 环境 info 列表

		Returns:
			快照字典
		"""
		if self._assembler is not None:
			try:
				return self._assembler.assemble(step, actions, rewards, infos)
			except Exception as exc:
				logger.warning(f"Step {step} snapshot assembly failed: {exc}")

		# 降级: 只返回基本信息
		snapshot: Dict[str, Any] = {"step": step}

		if actions is not None:
			snapshot["actions"] = (
				actions.tolist() if isinstance(actions, np.ndarray) else actions
			)
		if rewards is not None:
			snapshot["rewards"] = (
				rewards.tolist() if isinstance(rewards, np.ndarray) else rewards
			)

		return snapshot

	def _random_actions(self) -> np.ndarray:
		"""生成随机动作。

		Returns:
			np.ndarray(n_agents, max_action_dim)
		"""
		max_dim = 1
		for space in self._env.action_space:
			if isinstance(space, Discrete):
				max_dim = max(max_dim, 1)
			elif isinstance(space, Box):
				max_dim = max(max_dim, space.shape[0])

		actions = np.zeros((self._env.n_agents, max_dim), dtype=np.float32)

		for agent_id, space in enumerate(self._env.action_space):
			if isinstance(space, Discrete):
				actions[agent_id, 0] = float(np.random.randint(0, space.n))
			elif isinstance(space, Box):
				act_dim = space.shape[0]
				actions[agent_id, :act_dim] = np.random.uniform(
					space.low, space.high
				).astype(np.float32)

		return actions

	def _build_config_summary(self) -> Dict[str, Any]:
		"""构建 VVC 环境配置摘要。

		Returns:
			配置摘要字典
		"""
		summary: Dict[str, Any] = {
			"env_name": self.config.get("env_name", "vvc"),
			"n_agents": self._env.n_agents if self._env else 0,
			"episode_length": 24,
		}

		if self._env is not None:
			env_core = self._env.env
			summary.update({
				"cap_num": getattr(env_core, "cap_num", 0),
				"reg_num": getattr(env_core, "reg_num", 0),
				"bat_num": getattr(env_core, "bat_num", 0),
				"pv_num": getattr(env_core, "pv_num", 0),
				"pv_control": getattr(env_core, "pv_control_enabled", False),
				"cap_names": getattr(self._env, "cap_names", []),
				"reg_names": getattr(self._env, "reg_names", []),
				"bat_names": getattr(self._env, "bat_names", []),
				"pv_names": getattr(self._env, "pv_names", []),
			})

			# 动作空间信息
			action_info: List[Dict[str, Any]] = []
			for i, space in enumerate(self._env.action_space):
				if isinstance(space, Discrete):
					action_info.append({"type": "discrete", "n": space.n})
				elif isinstance(space, Box):
					action_info.append({
						"type": "continuous",
						"shape": list(space.shape),
					})
			summary["action_spaces"] = action_info

		return summary

	def _get_max_steps(self) -> int:
		"""VVC 环境默认 24 步/episode (逐小时)。"""
		return self.config.get("episode_length", 24)

	def close(self) -> None:
		"""释放环境和快照组装器资源。"""
		self._assembler = None
		super().close()
