"""实际设备空间上的随机/策略回放；团队奖励在回放中只统计一次。"""

from pathlib import Path
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.powergrid.snapshot import assemble_snapshot

logger = logging.getLogger(__name__)


class DeviceEpisodeRunner(BaseEpisodeRunner):
    """设备运行任务的Episode运行器

    Args:
            config: 当前任务的配置对象或正式环境字典
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
        self.project_root = project_root or str(Path(__file__).resolve().parents[2])
        self._n_agents: int = 0
        self._action_dim: int = 0

    def _create_env(self, seed: Optional[int] = None) -> Any:
        """创建具体任务并reset

        Args:
                seed: 随机种子

        Returns:
                具体任务内核
        """
        env = self._make_core()
        cfg = env.config
        if seed is not None:
            env.seed(seed)
            np.random.seed(seed)

        obs = env.reset(load_profile_idx=0)

        # 保存状态
        self._n_agents = len(env.agent_spaces)
        self._sizes = [1 if hasattr(s, "n") else s.shape[0] for s in env.agent_spaces]
        self._action_dim = sum(self._sizes)
        if self.inference_engine is not None:
            from utils.physics_manifest import validate_checkpoint

            validate_checkpoint(
                self.inference_engine.checkpoint_dir, env.get_physics_manifest()
            )

        # BaseEpisodeRunner 需要的 obs 列表
        if isinstance(obs, np.ndarray):
            self._current_obs = self._actor_observations(env, obs)
            self._current_share_obs = [obs] * self._n_agents
        else:
            wrapped = env.wrap_obs(env.obs).astype(np.float32)
            self._current_obs = [wrapped]
            self._current_share_obs = [wrapped]

        self._current_avail_actions = self._available_actions(env)

        logger.info(
            f"Device task created: system={getattr(cfg, 'system_name', 'unknown')}, "
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
        values = np.asarray(actions)
        if values.ndim == 2 and values.shape[0] == self._n_agents:
            from envs.powergrid.adapters import actor_action_frame

            action = actor_action_frame(self._env.agent_spaces, values)
        else:
            action = values.reshape(-1)
        obs, reward, done, info = self._env.step(action)

        # 转为 MARL 格式
        if isinstance(obs, np.ndarray):
            obs_list = self._actor_observations(self._env, obs.astype(np.float32))
        else:
            obs_list = [self._env.wrap_obs(self._env.obs).astype(np.float32)]

        rewards = np.array([[reward]], dtype=np.float32)
        dones = np.array([done], dtype=bool)
        infos = [info]

        return (
            obs_list,
            [obs.astype(np.float32)] * self._n_agents,
            rewards,
            dones,
            infos,
            self._available_actions(self._env),
        )

    @staticmethod
    def _available_actions(env):
        return (
            env.available_actions()
            if hasattr(env, "available_actions")
            else [None] * len(env.agent_spaces)
        )

    def _actor_observations(self, env, obs):
        return [obs.copy() for _ in env.agent_spaces]

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

        if info_dict.get("physical_failure"):
            # 物理失败步没有可读取的有效物理状态（失败会话禁止读取，
            # 两个快照组装器的 read_state 都会抛错）；返回显式失败标记
            # 快照，回放与导出据此展示终止原因而非中断。
            return {
                "step": step,
                "physical_failure": True,
                "failure_reason": info_dict.get("failure_reason", ""),
                "reward": float(np.asarray(rewards).reshape(-1)[0])
                if rewards is not None
                else None,
                "buses": {},
                "lines": [],
            }

        snapshot = self._snapshot(self._env, step, actions, rewards, info_dict)

        return snapshot

    def _random_actions(self) -> np.ndarray:
        """生成随机动作

        Returns:
                np.ndarray shape=(1, action_dim)
        """
        actions = np.zeros((self._n_agents, max(self._sizes)), dtype=np.float32)
        for i, (space, size) in enumerate(zip(self._env.agent_spaces, self._sizes)):
            mask = self._available_actions(self._env)[i]
            value = (
                self._env.rng.choice(np.flatnonzero(mask))
                if mask is not None
                else space.sample()
            )
            actions[i, :size] = np.asarray(value).reshape(-1)
        return actions

    def _build_config_summary(self):
        return {
            "task": type(self._env.config).__name__,
            "system_name": self._env.config.system_name,
            "max_episode_steps": self._env.horizon,
            "step_seconds": self._env.circuit.step_seconds,
            "action_schema": self._env.get_physics_manifest()["action_schema"],
        }

    def _get_max_steps(self):
        if isinstance(self.config, dict):
            return self.config.get("max_episode_steps", 24)
        return self.config.max_episode_steps

    def _make_core(self):
        raise NotImplementedError

    def _snapshot(self, env, step, actions, rewards, infos):
        return assemble_snapshot(env, step, actions, rewards, infos)
