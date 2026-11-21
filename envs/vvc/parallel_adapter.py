"""VVC 同步设备环境的 PettingZoo Parallel 协议边界。"""

from utils.capabilities import device_layout


class VVCParallelAdapter:
    """接收一个 VVCEnv，由该适配器独占并关闭该环境。"""

    metadata = {"name": "powerzoo_vvc_parallel_v0", "render_modes": []}
    render_mode = None

    def __init__(self, wrapper):
        self._wrapper = wrapper
        self._ids = device_layout(wrapper).device_ids
        self.possible_agents = list(self._ids)
        self.agents = []
        self.observation_spaces = dict(zip(self._ids, wrapper.observation_space))
        self.action_spaces = dict(zip(self._ids, wrapper.action_space))
        self.state_space = wrapper.share_observation_space[0]
        self._state = None
        self._closed = False

    @property
    def unwrapped(self):
        return self

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self._ids)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def state(self):
        if self._state is None:
            raise RuntimeError("请先 reset")
        return self._state.copy()

    def reset(self, seed=None, options=None):
        if self._closed:
            raise RuntimeError("环境已关闭")
        scenario = (options or {}).get("scenario_index")
        if scenario is None:
            if seed is not None:
                self._wrapper.seed(seed)
            observations, states, masks = self._wrapper.reset()
        else:
            obs = self._wrapper.env.reset(load_profile_idx=scenario, seed=seed)
            observations = self._wrapper._observations(obs)
            states = self._wrapper._shared_observations(obs)
            masks = self._wrapper.get_avail_actions()
        self._state = states[0]
        self.agents = list(self._ids)
        infos = {agent: {} for agent in self._ids}
        return dict(zip(self._ids, observations)), self._with_masks(infos, masks)

    def _with_masks(self, infos, masks):
        return {
            agent: dict(infos[agent], structural_action_mask=mask.copy())
            if mask is not None
            else dict(infos[agent])
            for agent, mask in zip(self._ids, masks)
        }

    def step(self, actions):
        if not self.agents:
            if actions:
                raise ValueError("没有在场设备，请先 reset")
            return {}, {}, {}, {}, {}
        if set(actions) != set(self._ids):
            raise ValueError("必须一次提供全部在场设备的动作")
        frame = [actions[agent] for agent in self._ids]
        if any(
            not self.action_spaces[agent].contains(action)
            for agent, action in zip(self._ids, frame)
        ):
            raise ValueError("动作超出设备声明空间")

        observations, states, rewards, done, info, masks = self._wrapper.step(frame)
        pairs = [(item["terminated"], item["truncated"]) for item in info]
        terminated, truncated = pairs[0]
        if (
            len(pairs) != len(self._ids)
            or any(pair != pairs[0] for pair in pairs)
            or len(done) != len(self._ids)
            or any(bool(value) != (terminated or truncated) for value in done)
        ):
            raise RuntimeError("VVC 环境必须同步结束且显式提供终止/截断")
        self._state = states[0]
        if terminated or truncated:
            self.agents = []
        return (
            dict(zip(self._ids, observations)),
            {agent: float(value[0]) for agent, value in zip(self._ids, rewards)},
            dict.fromkeys(self._ids, terminated),
            dict.fromkeys(self._ids, truncated),
            self._with_masks(dict(zip(self._ids, info)), masks),
        )

    def render(self):
        return None

    def close(self):
        if not self._closed:
            self._wrapper.close()
            self.agents = []
            self._state = None
            self._closed = True
