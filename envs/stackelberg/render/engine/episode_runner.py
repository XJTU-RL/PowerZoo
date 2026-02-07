# -*- coding: utf-8 -*-
"""
Stackelberg Episode 运行器

驱动 Stackelberg 博弈环境运行完整 episode，
每步通过 SnapshotAssembler 收集 OpenDSS 快照和市场数据。
支持异质智能体：UC Leader (5D) + N Consumer Followers (3D)。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.stackelberg.render.data.snapshot_assembler import SnapshotAssembler

logger = logging.getLogger(__name__)

# UC Leader 动作维度 = 5, Consumer 动作维度 = 3
UC_ACTION_DIM = 5
CONSUMER_ACTION_DIM = 3


class StackelbergEpisodeRunner(BaseEpisodeRunner):
	"""Stackelberg Episode 运行器

	处理异质智能体：
	- Agent 0 = UC Leader (5D: price, DR_signal, ESS_charge, ESS_discharge, reserve)
	- Agent 1..N = Consumer Followers (3D: load_adj, DER_output, flexibility)

	Args:
		config: Stackelberg 环境配置字典
		inference_engine: 推理引擎
		collect_snapshots: 是否收集快照
	"""

	def __init__(
		self,
		config: Dict[str, Any],
		inference_engine: Optional[InferenceEngine] = None,
		collect_snapshots: bool = True,
	):
		super().__init__(config, inference_engine, collect_snapshots)
		self._snapshot_assembler: Optional[SnapshotAssembler] = None
		self._n_agents: int = 0
		self._n_consumers: int = 0
		self._max_action_dim: int = UC_ACTION_DIM

	# ------------------------------------------------------------------
	# 抽象方法实现
	# ------------------------------------------------------------------

	def _create_env(self, seed: Optional[int] = None) -> Any:
		"""创建 Stackelberg 环境实例并执行 reset

		Args:
			seed: 随机种子

		Returns:
			Stackelberg 环境实例
		"""
		from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv

		env = StackelbergVVCEnv(self.config)
		obs, share_obs, avail_actions = env.reset()

		self._current_obs = obs
		self._current_share_obs = share_obs
		self._current_avail_actions = avail_actions
		self._n_agents = env.n_agents
		self._n_consumers = max(0, self._n_agents - 1)

		# 初始化快照组装器
		if self.collect_snapshots:
			dss = self._get_dss_engine(env)
			if dss is not None:
				self._snapshot_assembler = SnapshotAssembler(
					dss, n_consumers=self._n_consumers
				)

		return env

	def _env_step(self, actions: np.ndarray) -> tuple:
		"""执行环境单步

		Args:
			actions: 动作数组 shape=(n_agents, max_action_dim)

		Returns:
			(obs, share_obs, rewards, dones, infos, avail_actions)
		"""
		obs, share_obs, rewards, dones, infos, avail_actions = self._env.step(
			actions
		)
		return obs, share_obs, rewards, dones, infos, avail_actions

	def _take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
	) -> Dict[str, Any]:
		"""收集当前步的完整快照

		使用 SnapshotAssembler 提取电路 + 市场数据，
		额外添加 agent 角色标注。

		Args:
			step: 步编号
			actions: 动作
			rewards: 奖励
			infos: 环境信息

		Returns:
			快照字典
		"""
		# 获取环境内部状态
		env_state = self._get_env_state()

		if self._snapshot_assembler is not None:
			snapshot = self._snapshot_assembler.assemble(
				step=step,
				actions=actions,
				rewards=rewards,
				infos=infos,
				env_state=env_state,
			)
		else:
			snapshot = {"step": step}
			if actions is not None:
				snapshot["actions"] = (
					actions.tolist() if hasattr(actions, "tolist") else list(actions)
				)
			if rewards is not None:
				snapshot["rewards"] = (
					rewards.tolist() if hasattr(rewards, "tolist") else list(rewards)
				)

		# 添加 agent 角色标注
		snapshot["agent_roles"] = self._get_agent_roles()
		snapshot["n_agents"] = self._n_agents
		snapshot["n_consumers"] = self._n_consumers

		return snapshot

	def _random_actions(self) -> np.ndarray:
		"""生成异质随机动作

		UC Leader: 5D, Consumer: 3D，统一填充为 max_action_dim 宽度。
		Consumer 多余维度填 0（环境 step 会截断）。

		Returns:
			np.ndarray(n_agents, max_action_dim)
		"""
		actions = np.zeros(
			(self._n_agents, self._max_action_dim), dtype=np.float32
		)
		# UC Leader: 全 5 维随机
		actions[0, :UC_ACTION_DIM] = np.random.uniform(-1.0, 1.0, UC_ACTION_DIM)
		# Consumers: 前 3 维随机
		for i in range(1, self._n_agents):
			actions[i, :CONSUMER_ACTION_DIM] = np.random.uniform(
				-1.0, 1.0, CONSUMER_ACTION_DIM
			)
		return actions

	def _build_config_summary(self) -> Dict[str, Any]:
		"""构建配置摘要

		Returns:
			配置摘要字典
		"""
		cfg = self.config if isinstance(self.config, dict) else {}
		env_args = cfg.get("env_args", cfg)

		return {
			"env_name": "stackelberg",
			"system_name": env_args.get("system_name", "13Bus"),
			"n_agents": self._n_agents,
			"n_consumers": self._n_consumers,
			"uc_action_dim": UC_ACTION_DIM,
			"consumer_action_dim": CONSUMER_ACTION_DIM,
			"max_action_dim": self._max_action_dim,
			"episode_length": env_args.get("episode_length", 24),
		}

	def _get_max_steps(self) -> int:
		"""Stackelberg 默认 24 步/episode"""
		if isinstance(self.config, dict):
			env_args = self.config.get("env_args", self.config)
			return env_args.get("episode_length", 24)
		return 24

	# ------------------------------------------------------------------
	# 辅助方法
	# ------------------------------------------------------------------

	def _get_dss_engine(self, env: Any) -> Any:
		"""从环境中提取 DSS 引擎实例

		Args:
			env: Stackelberg 环境

		Returns:
			dss-python 引擎实例，或 None
		"""
		# StackelbergVVCEnv → base_env → circuit_adapter → dss
		for attr_chain in [
			["base_env", "circuit_adapter", "dss"],
			["base_env", "circuit", "dss"],
			["circuit_adapter", "dss"],
		]:
			obj = env
			try:
				for attr in attr_chain:
					obj = getattr(obj, attr)
				return obj
			except AttributeError:
				continue

		logger.warning("无法从环境中提取 DSS 引擎，快照将缺少电路数据")
		return None

	def _get_env_state(self) -> Optional[Dict[str, Any]]:
		"""从环境中提取内部状态

		Returns:
			环境内部状态字典，或 None
		"""
		if self._env is None:
			return None

		state: Dict[str, Any] = {}

		# 尝试获取 ESS SoC
		for attr_chain in [
			["base_env", "ess_soc"],
			["base_env", "circuit_adapter", "ess_soc"],
		]:
			obj = self._env
			try:
				for attr in attr_chain:
					obj = getattr(obj, attr)
				state["ess_soc"] = float(obj)
				break
			except (AttributeError, TypeError):
				continue

		# 尝试获取总需求和供给
		for key in ["total_demand", "total_supply", "demand_response_signal"]:
			for prefix in ["base_env", ""]:
				try:
					obj = getattr(self._env, prefix) if prefix else self._env
					if prefix:
						obj = getattr(obj, key)
					else:
						obj = getattr(self._env, key)
					state[key] = float(obj)
					break
				except (AttributeError, TypeError):
					continue

		return state if state else None

	def _get_agent_roles(self) -> List[Dict[str, str]]:
		"""获取 agent 角色标注

		Returns:
			[{id, role, label}, ...]
		"""
		roles = [{"id": 0, "role": "uc_leader", "label": "UC Leader"}]
		for i in range(1, self._n_agents):
			roles.append({
				"id": i,
				"role": "consumer",
				"label": f"Consumer {i}",
			})
		return roles
