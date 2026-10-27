# -*- coding: utf-8 -*-
"""
District Dispatch MARL Environment
区域调度多智能体强化学习环境包装器

兼容 PowerZoo 的 MARL 接口 (ShareVecEnv 格式)，
封装 DispatchCore 并转换返回值为 Runner 期望的格式。
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
	import gymnasium as gym
	from gymnasium.spaces import Box
except ImportError:
	import gym
	from gym.spaces import Box

from envs.district_dispatch.core.config import (
	DistrictDeviceConfig,
	DistrictDispatchConfig,
	DispatchRewardWeights,
	TieLineConfig,
)
from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore

logger = logging.getLogger(__name__)


class DistrictDispatchEnv:
	"""区域调度 MARL 环境

	PowerZoo MARL 接口:
	- n_agents: int
	- observation_space: list[Box]       (per-agent)
	- share_observation_space: list[Box]  (centralized critic)
	- action_space: list[Box]             (per-agent, continuous)

	- step(actions) -> (obs, share_obs, rewards, dones, infos, avail_actions)
	- reset()       -> (obs, share_obs, avail_actions)
	- seed(seed)
	- close()

	参数:
		args: 环境参数字典
		rank: 并行进程编号 (worker_idx)
	"""

	def __init__(self, args: Dict[str, Any], rank: Optional[int] = None):
		self.args = copy.deepcopy(args)
		self.rank = rank

		# 解析配置
		self.config = self._parse_config(args)

		# 创建核心环境
		self.core = DistrictDispatchCore(self.config, worker_idx=rank)

		# 智能体信息
		self.n_agents = self.core.n_agents

		# 动作/观测空间维度
		self.max_action_dim = self.config.get_max_action_dim()
		self.max_obs_dim = self.config.get_max_obs_dim()
		self.share_obs_dim = self.config.get_share_obs_dim()

		# 定义空间
		self._setup_spaces()

		# PowerZoo 兼容属性
		self.env_name = args.get("env_name", "district_dispatch")
		self.discrete = False  # 连续动作空间

		# Stackelberg 模式的顺序更新不适用于本环境
		self.ordered_agents_pairs = None
		self.agents_bus = None

		logger.info(
			f"DistrictDispatchEnv 初始化: "
			f"n_agents={self.n_agents}, "
			f"obs={self.max_obs_dim}, share_obs={self.share_obs_dim}, "
			f"action={self.max_action_dim}, worker={rank}"
		)

	def _parse_config(self, args):
		"""唯一 dataclass 边界，所有市场、约束和嵌套设备参数均传递。"""
		from utils.config_validation import deep_merge, build_dataclass, normalize_env
		values = deep_merge(args.get('env_args', {}), {k: v for k, v in args.items() if k != 'env_args'})
		values['worker_idx'] = self.rank
		return build_dataclass(DistrictDispatchConfig, normalize_env(values, 'district_dispatch'), strict=False)

	def _setup_spaces(self) -> None:
		"""定义观测和动作空间"""
		# Per-agent 观测空间
		self.observation_space = [
			Box(
				low=-np.inf,
				high=np.inf,
				shape=(self.max_obs_dim,),
				dtype=np.float32,
			)
			for _ in range(self.n_agents)
		]

		# Centralized critic 共享观测空间
		self.share_observation_space = [
			Box(
				low=-np.inf,
				high=np.inf,
				shape=(self.share_obs_dim,),
				dtype=np.float32,
			)
			for _ in range(self.n_agents)
		]

		# Per-agent 连续动作空间 [-1, 1]
		self.action_space = [
			Box(
				low=-1.0,
				high=1.0,
				shape=(self.max_action_dim,),
				dtype=np.float32,
			)
			for _ in range(self.n_agents)
		]

	# ===== MARL 接口 =====

	def step(self, actions) -> Tuple[
		List[np.ndarray],
		List[np.ndarray],
		np.ndarray,
		np.ndarray,
		List[Dict[str, Any]],
		List[np.ndarray],
	]:
		"""执行一步

		参数:
			actions: 动作数组，shape=(n_agents, max_action_dim)

		返回:
			(obs, share_obs, rewards, dones, infos, available_actions)
		"""
		# 确保 actions 格式正确
		if isinstance(actions, (list, tuple)):
			actions = np.array(actions, dtype=np.float32)
		if actions.ndim == 1:
			actions = actions.reshape(self.n_agents, -1)

		# 裁剪到 [-1, 1]
		actions = np.clip(actions, -1.0, 1.0)

		obs, share_obs, rewards, dones, infos = self.core.step(actions)

		# 连续动作空间没有 available_actions 约束
		avail_actions = self._get_available_actions()

		return obs, share_obs, rewards, dones, infos, avail_actions

	def reset(self) -> Tuple[
		List[np.ndarray],
		List[np.ndarray],
		List[np.ndarray],
	]:
		"""重置环境

		返回:
			(obs, share_obs, available_actions)
		"""
		obs, share_obs = self.core.reset()
		avail_actions = self._get_available_actions()
		return obs, share_obs, avail_actions

	def _get_available_actions(self):
		"""连续动作空间的 available_actions

		连续 Box 空间不需要动作 mask。
		Runner 在 insert 时通过 available_actions[0] is None
		判断是否跳过 buffer 写入，因此必须返回 None。
		"""
		return None

	def seed(self, seed: int) -> None:
		"""设置随机种子"""
		self.core.seed(seed)

	def close(self) -> None:
		"""关闭环境"""
		self.core.close()
		logger.info("DistrictDispatchEnv closed")

	def render(self, mode: str = "human") -> None:
		"""渲染（预留接口）"""
		pass
