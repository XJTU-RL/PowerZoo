"""Gymnasium 集中控制基线：共享内核，仅转换动作和返回值。"""

import numpy as np
import gymnasium as gym
from envs.smartgrid.base_env.runtime_env import Env


class SingleAgentVVCEnv(gym.Env):
    def __init__(self, config, action_space_type="continuous", folder_path=None):
        from pathlib import Path
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        if not isinstance(config, SmartGridConfig):
            raise TypeError(
                "请使用公共 ConfigLoader/SmartGridConfig；旧 SingleAgentConfig 已停止支持"
            )
        root = Path(__file__).resolve().parents[3]
        self.core = Env(str(folder_path or root), config)
        self.observation_space = self.core.observation_space
        self.mode = action_space_type
        self._sizes = [
            1 if isinstance(s, gym.spaces.Discrete) else s.shape[0]
            for s in self.core.agent_spaces
        ]
        if self.mode == "continuous":
            # 集中式 Box 基线会将离散设备动作等距量化；物理执行仍只发生在内核。
            self.action_space = gym.spaces.Box(
                -1, 1, (sum(self._sizes),), dtype=np.float32
            )
        elif self.mode == "discrete" and all(
            isinstance(s, gym.spaces.Discrete) for s in self.core.agent_spaces
        ):
            self.action_space = gym.spaces.MultiDiscrete(
                [s.n for s in self.core.agent_spaces]
            )
        else:
            self.close()
            raise ValueError(
                "discrete 基线要求所有设备为离散动作；混合动作请用 continuous"
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("scenario_index")
        obs = self.core.reset(load_profile_idx=index, seed=seed)
        return obs, self.core._info()

    def step(self, action):
        values = np.asarray(action).reshape(-1)
        if len(values) != sum(self._sizes) or not np.isfinite(values).all():
            raise ValueError("集中控制动作形状或数值不符")
        flat, offset = [], 0
        for space, size in zip(self.core.agent_spaces, self._sizes):
            part = values[offset : offset + size]
            offset += size
            if self.mode == "continuous" and isinstance(space, gym.spaces.Discrete):
                part = [
                    int(
                        np.clip(
                            np.floor((float(part[0]) + 1) / 2 * space.n), 0, space.n - 1
                        )
                    )
                ]
            flat.extend(part)
        obs, reward, done, info = self.core.step(flat)
        return obs, reward, done, False, info

    def close(self):
        self.core.close()
