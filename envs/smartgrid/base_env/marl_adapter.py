"""公共 MARL 接口：只转换设备动作和数组，不执行仿真。"""

import numpy as np
import gymnasium as gym


class VVCEnv:
    def __init__(self, env, config_or_args=None, rank=None):
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        if isinstance(env, SmartGridConfig):
            from envs.smartgrid.base_env.env_register import make_base_env

            env = make_base_env(env)
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


OptimizedVVCEnv = VVCEnv
