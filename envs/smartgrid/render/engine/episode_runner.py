"""
SmartGrid Episode Runner
SmartGrid 环境 episode 运行器

继承 BaseEpisodeRunner，实现 SmartGrid 特定的环境创建、
步进、快照收集和随机动作生成。
"""

import copy
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.smartgrid.render.data.snapshot_assembler import assemble_snapshot

logger = logging.getLogger(__name__)


class SmartGridEpisodeRunner(BaseEpisodeRunner):
	"""SmartGrid Episode 运行器

	Args:
		config: SmartGridConfig 或等效 info 字典
		inference_engine: 推理引擎
		collect_snapshots: 是否收集快照
		project_root: 项目根目录路径
	"""

	def __init__(
		self,
		config: Any,
		inference_engine: Optional[InferenceEngine] = None,
		collect_snapshots: bool = True,
		project_root: str = "",
	):
		super().__init__(config, inference_engine, collect_snapshots)
		self.project_root = project_root or os.getcwd()
		self._n_agents: int = 0
		self._action_dim: int = 0

	def _create_env(self, seed: Optional[int] = None) -> Any:
		"""创建 SmartGrid 环境并 reset

		Args:
			seed: 随机种子

		Returns:
			SmartGrid Env 实例
		"""
		from envs.smartgrid.base_env.env import Env as SmartGridBaseEnv
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		if isinstance(self.config, SmartGridConfig):
			cfg = self.config
		elif isinstance(self.config, dict):
			cfg = SmartGridConfig.from_dict(self.config)
		else:
			cfg = self.config

		env = SmartGridBaseEnv(
			folder_path=self.project_root,
			config_or_info=cfg,
		)

		if seed is not None:
			env.seed(seed)
			np.random.seed(seed)

		obs = env.reset(load_profile_idx=0)

		# 保存状态
		self._n_agents = 1  # SmartGrid 基础环境是单环境，MARL wrapper 会拆成多 agent
		self._action_dim = env.ActionSpace.dim()

		# BaseEpisodeRunner 需要的 obs 列表
		if isinstance(obs, np.ndarray):
			self._current_obs = [obs]
			self._current_share_obs = [obs]
		else:
			wrapped = env.wrap_obs(env.obs).astype(np.float32)
			self._current_obs = [wrapped]
			self._current_share_obs = [wrapped]

		self._current_avail_actions = [None]

		logger.info(
			f"SmartGrid env created: system={getattr(cfg, 'system_name', 'unknown')}, "
			f"horizon={getattr(cfg, 'max_episode_steps', 360)}, "
			f"action_dim={self._action_dim}"
		)

		return env

	def _env_step(self, actions: np.ndarray) -> tuple:
		"""执行环境单步

		Args:
			actions: 动作数组 shape=(1, action_dim) 或 (action_dim,)

		Returns:
			(obs, share_obs, rewards, dones, infos, avail_actions)
		"""
		# 转换动作格式
		if actions.ndim == 2:
			action = actions[0]
		else:
			action = actions

		# 对离散动作取整
		action_int = np.round(action).astype(int)

		obs, reward, done, info = self._env.step(action_int)

		# 转为 MARL 格式
		if isinstance(obs, np.ndarray):
			obs_list = [obs.astype(np.float32)]
		else:
			obs_list = [self._env.wrap_obs(self._env.obs).astype(np.float32)]

		rewards = np.array([[reward]], dtype=np.float32)
		dones = np.array([done], dtype=bool)
		infos = [info]

		return obs_list, obs_list, rewards, dones, infos, [None]

	def _take_snapshot(
		self,
		step: int,
		actions: Optional[np.ndarray],
		rewards: Optional[np.ndarray],
		infos: Any,
	) -> Dict[str, Any]:
		"""收集快照

		Args:
			step: 步编号
			actions: 动作
			rewards: 奖励
			infos: 环境信息

		Returns:
			快照字典
		"""
		info_dict = {}
		if isinstance(infos, list) and infos:
			info_dict = infos[0] if isinstance(infos[0], dict) else {}
		elif isinstance(infos, dict):
			info_dict = infos

		snapshot = assemble_snapshot(
			env=self._env,
			step=step,
			actions=actions,
			rewards=rewards,
			infos=info_dict,
		)

		return snapshot

	def _random_actions(self) -> np.ndarray:
		"""生成随机动作

		Returns:
			np.ndarray shape=(1, action_dim)
		"""
		sample = self._env.ActionSpace.sample()
		if isinstance(sample, list):
			# 混合动作空间: [discrete_arr, continuous_arr]
			flat = np.concatenate([np.asarray(s, dtype=np.float32).flatten() for s in sample])
		elif isinstance(sample, np.ndarray):
			flat = sample.astype(np.float32)
		else:
			flat = np.array(sample, dtype=np.float32).flatten()

		return flat.reshape(1, -1)

	def _build_config_summary(self) -> Dict[str, Any]:
		"""构建配置摘要

		Returns:
			配置摘要字典
		"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		if isinstance(self.config, SmartGridConfig):
			return {
				"env_name": self.config.env_name,
				"system_name": self.config.system_name,
				"max_episode_steps": self.config.max_episode_steps,
				"use_cmdp": self.config.use_cmdp,
				"pv_control": self.config.pv_control_enabled,
				"voltage_range": [self.config.voltage_min, self.config.voltage_max],
				"action_dim": self._action_dim,
			}

		return {
			"env_name": "smartgrid",
			"action_dim": self._action_dim,
		}

	def _get_max_steps(self) -> int:
		"""获取默认最大步数"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		if isinstance(self.config, SmartGridConfig):
			return self.config.max_episode_steps
		if isinstance(self.config, dict):
			return self.config.get("max_episode_steps", 360)
		return 360
