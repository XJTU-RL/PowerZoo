"""
DSR Episode Runner
DSR 环境 Episode 运行器

继承 BaseEpisodeRunner，实现 DSR 环境的创建、步进、快照收集。
关键特性:
- _env_step 返回的 avail_actions 不是 None，必须传递给快照
- _take_snapshot 中包含 available_actions 和 restoration_state
- _random_actions 遵守 avail_actions 掩码（只选合法动作）
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.dsr.render.data.snapshot_assembler import SnapshotAssembler

logger = logging.getLogger(__name__)


class DSREpisodeRunner(BaseEpisodeRunner):
	"""DSR 环境 Episode 运行器

	处理 DSR 环境的异质智能体 (Switch/PV/Load)，
	在每步中正确传递动作掩码并收集恢复状态快照。

	Args:
		config: DSR 环境配置字典
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
		self._assembler: Optional[SnapshotAssembler] = None
		self._last_infos: Any = None
		self._last_actions: Optional[np.ndarray] = None
		self._last_rewards: Optional[np.ndarray] = None

	def _create_env(self, seed: Optional[int] = None) -> Any:
		"""创建 DSR 环境实例并执行 reset

		Args:
			seed: 随机种子

		Returns:
			DSREnv 实例
		"""
		from envs.dsr.dsr_env import DSREnv

		env_args = copy.deepcopy(self.config)

		if seed is not None:
			env_args["seed"] = seed

		env = DSREnv(env_args, rank=0)

		if seed is not None:
			env.seed(seed)

		obs, share_obs, avail_actions = env.reset()

		self._current_obs = obs
		self._current_share_obs = share_obs
		self._current_avail_actions = avail_actions
		self._assembler = SnapshotAssembler(env)

		logger.info(
			f"DSR env created: n_agents={env.n_agents}, "
			f"agent_types={getattr(env, 'agent_types', 'unknown')}, "
			f"system={env_args.get('system_name', 'unknown')}"
		)

		return env

	def _env_step(self, actions: np.ndarray) -> tuple:
		"""执行环境单步

		Args:
			actions: 动作数组 shape=(n_agents, max_action_dim)

		Returns:
			(obs, share_obs, rewards, dones, infos, avail_actions)
		"""
		# DSR 使用离散动作，需要从 action 数组中提取离散选择
		discrete_actions = self._convert_to_discrete(actions)

		obs, share_obs, rewards, dones, infos, avail_actions = self._env.step(
			discrete_actions
		)

		# 缓存用于快照
		self._last_actions = actions
		self._last_rewards = rewards
		self._last_infos = infos

		return obs, share_obs, rewards, dones, infos, avail_actions

	def _take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
	) -> Dict[str, Any]:
		"""收集当前步的完整快照

		包含 DSR 特有的 available_actions 和 restoration_data。

		Args:
			step: 步编号
			actions: 动作
			rewards: 奖励
			infos: 环境信息

		Returns:
			快照字典
		"""
		if self._assembler is None:
			return {"step": step}

		snapshot = self._assembler.assemble(
			step=step,
			actions=actions,
			rewards=rewards,
			infos=infos,
			available_actions=self._current_avail_actions,
		)

		return snapshot

	def _random_actions(self) -> np.ndarray:
		"""生成遵守动作掩码的随机动作

		DSR 环境使用离散动作空间 + 动作掩码，
		随机动作必须只从合法动作中选择。

		Returns:
			np.ndarray(n_agents, max_action_dim)
		"""
		n_agents = self._env.n_agents
		action_spaces = self._env.action_space
		max_action_dim = max(sp.n for sp in action_spaces)

		actions = np.zeros((n_agents, max_action_dim), dtype=np.float32)

		avail = self._current_avail_actions

		for i in range(n_agents):
			n_actions = action_spaces[i].n
			if avail is not None and i < len(avail) and avail[i] is not None:
				# 只从合法动作中随机选择
				mask = np.array(avail[i], dtype=bool)
				valid_indices = np.where(mask[:n_actions])[0]
				if len(valid_indices) > 0:
					chosen = np.random.choice(valid_indices)
				else:
					# 所有动作被掩码禁用，选择第一个动作 (安全后备)
					chosen = 0
			else:
				chosen = np.random.randint(0, n_actions)

			actions[i, 0] = float(chosen)

		return actions

	def _build_config_summary(self) -> Dict[str, Any]:
		"""构建配置摘要

		Returns:
			配置摘要字典
		"""
		summary: Dict[str, Any] = {
			"env_name": "dsr",
			"env_type": "DSR (Service Restoration)",
		}

		if self._env is not None:
			summary["n_agents"] = self._env.n_agents
			summary["agent_types"] = list(getattr(self._env, "agent_types", []))
			summary["discrete"] = getattr(self._env, "discrete", True)

			core = getattr(self._env, "core_env", None)
			if core is not None:
				config = getattr(core, "config", None)
				if config is not None:
					summary["system_name"] = getattr(config, "system_name", "unknown")
					summary["max_episode_steps"] = getattr(config, "max_episode_steps", 20)
					summary["n_pv"] = getattr(config, "n_pv", 0)
					summary["n_switch"] = getattr(config, "n_switch", 0)
					summary["v_min"] = getattr(config, "v_min", 0.95)
					summary["v_max"] = getattr(config, "v_max", 1.05)

				summary["n_bus"] = getattr(core, "n_bus", 0)
				summary["faultable_lines"] = list(getattr(core, "faultable_lines", []))

			# 动作空间信息
			summary["action_spaces"] = [
				{"type": "Discrete", "n": sp.n} for sp in self._env.action_space
			]

		# 源配置参数
		for key in ["system_name", "max_episode_steps", "seed", "use_action_mask"]:
			if key in self.config:
				summary[key] = self.config[key]

		return summary

	def _get_max_steps(self) -> int:
		"""获取默认最大步数

		DSR 环境通常 10-20 步/episode。
		"""
		if "max_episode_steps" in self.config:
			return self.config["max_episode_steps"]
		env_args = self.config.get("env_args", {})
		if "max_episode_steps" in env_args:
			return env_args["max_episode_steps"]
		return 20  # DSR 默认

	def _convert_to_discrete(self, actions: np.ndarray) -> List[int]:
		"""将连续动作数组转换为离散动作列表

		Args:
			actions: shape=(n_agents, max_action_dim)

		Returns:
			离散动作列表
		"""
		discrete_actions: List[int] = []
		n_agents = self._env.n_agents

		for i in range(n_agents):
			n_actions = self._env.action_space[i].n
			# 取第一维作为离散选择（已经是 index）
			raw = actions[i, 0] if actions.ndim > 1 else actions[i]
			action_idx = int(np.clip(round(float(raw)), 0, n_actions - 1))

			# 动作掩码校验
			avail = self._current_avail_actions
			if avail is not None and i < len(avail) and avail[i] is not None:
				mask = np.array(avail[i], dtype=bool)
				if action_idx < len(mask) and not mask[action_idx]:
					# 非法动作，选择第一个合法的
					valid = np.where(mask[:n_actions])[0]
					if len(valid) > 0:
						action_idx = int(valid[0])
					else:
						action_idx = 0

			discrete_actions.append(action_idx)

		return discrete_actions
