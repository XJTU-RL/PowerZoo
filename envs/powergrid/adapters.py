"""公共 MARL 接口：只转换设备动作和数组，不执行仿真。"""

import numpy as np
import gymnasium as gym


class DeviceMARLAdapter:
    def __init__(self, env, rank=None):
        self.env, self.config, self.rank = env, env.config, rank
        self.action_space = env.agent_spaces
        self.n_agents = len(self.action_space)
        self.observation_space = [env.observation_space] * self.n_agents
        self.share_observation_space = self.observation_space
        self.ordered_agents_pairs = None
        self.agents_bus = None

    def get_avail_actions(self):
        return [
            (
                np.ones(s.n, dtype=np.float32)
                if isinstance(s, gym.spaces.Discrete)
                else None
            )
            for s in self.action_space
        ]

    def get_physics_manifest(self):
        return self.env.get_physics_manifest()

    def _observations(self, obs):
        return [obs.copy() for _ in self.action_space]

    def step(self, actions):
        if len(actions) != self.n_agents:
            raise ValueError(f"需要 {self.n_agents} 个智能体动作")
        flat = []
        for space, value in zip(self.action_space, actions):
            dimension = 1 if isinstance(space, gym.spaces.Discrete) else space.shape[0]
            array = np.asarray(value).reshape(-1)
            if len(array) < dimension:
                raise ValueError("智能体动作维度不足")
            part = array[:dimension]
            if isinstance(space, gym.spaces.Box):
                # HAPPO 的高斯策略可以输出界外动作，执行前投影到设备空间。
                part = np.clip(part, space.low, space.high)
            flat.extend(part)
        obs, reward, done, info = self.env.step(flat)
        if info.get("TimeLimit.truncated"):
            info["bad_transition"] = True
        return (
            self._observations(obs),
            self._observations(obs),
            np.full((self.n_agents, 1), reward, dtype=np.float32),
            np.full(self.n_agents, done, dtype=bool),
            [dict(info) for _ in self.action_space],
            self.get_avail_actions(),
        )

    def reset(self):
        obs = self.env.reset()
        return (
            self._observations(obs),
            self._observations(obs),
            self.get_avail_actions(),
        )

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.obs


class CentralizedAdapter(gym.Env):
    def __init__(self, core, action_space_type="continuous"):
        self.core = core
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
