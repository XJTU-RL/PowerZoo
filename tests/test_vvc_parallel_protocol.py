"""PettingZoo Parallel 协议使用同步的假设备环境，不运行潮流。"""

import gymnasium
import numpy as np
import pytest

from envs.powergrid.adapters import DeviceMARLAdapter
from envs.vvc.parallel_adapter import VVCParallelAdapter


class FakeCore:
    config = object()
    agent_spaces = [gymnasium.spaces.Discrete(2), gymnasium.spaces.Box(-1, 1, (1,))]
    observation_space = gymnasium.spaces.Box(-1, 1, (2,))

    def __init__(self):
        self.calls = []
        self.flags = (True, False)

    def get_physics_manifest(self):
        return {
            "action_schema": [
                {"element": name, "space": str(space)}
                for name, space in zip(("Capacitor.a", "Storage.b"), self.agent_spaces)
            ]
        }

    def reset(self, load_profile_idx=None, seed=None):
        self.calls.append(("reset", load_profile_idx, seed))
        return np.array([0.2, 0.3], dtype=np.float32)

    def seed(self, seed):
        self.calls.append(("seed", seed))

    def step(self, action):
        self.calls.append(("step", action))
        terminated, truncated = self.flags
        return (
            np.array([0.3, 0.4], dtype=np.float32),
            -1.0,
            terminated or truncated,
            {
                "terminated": terminated,
                "truncated": truncated,
                "TimeLimit.truncated": truncated,
            },
        )

    def close(self):
        self.calls.append(("close",))


@pytest.mark.parametrize("flags", [(True, False), (False, True)])
def test_atomic_mixed_frames_and_terminal_lifecycle(flags):
    core = FakeCore()
    core.flags = flags
    env = VVCParallelAdapter(DeviceMARLAdapter(core))
    cap, storage = env.possible_agents
    observations, infos = env.reset(seed=17, options={"scenario_index": 3})
    assert tuple(observations) == (cap, storage)
    assert core.calls == [("reset", 3, 17)]
    assert infos[cap]["structural_action_mask"].tolist() == [1, 1]
    assert "structural_action_mask" not in infos[storage]
    with pytest.raises(ValueError, match="全部"):
        env.step({cap: 1})
    with pytest.raises(ValueError, match="超出"):
        env.step({cap: 3, storage: np.array([0.25], dtype=np.float32)})
    assert len(core.calls) == 1
    _, rewards, terminated, truncated, info = env.step(
        {storage: np.array([0.25], dtype=np.float32), cap: 1}
    )
    assert core.calls[-1] == ("step", [1, 0.25])
    assert rewards == {cap: -1.0, storage: -1.0}
    assert terminated == dict.fromkeys(env.possible_agents, flags[0])
    assert truncated == dict.fromkeys(env.possible_agents, flags[1])
    assert bool(info[cap].get("bad_transition")) is flags[1]
    assert not env.agents
    assert env.step({}) == ({}, {}, {}, {}, {})
    np.testing.assert_array_equal(env.state(), [np.float32(0.3), np.float32(0.4)])
    env.state()[:] = 0
    assert env.state().any()
    env.reset(seed=23)
    assert tuple(env.agents) == (cap, storage)
    assert core.calls[-2:] == [("seed", 23), ("reset", None, None)]
    env.close()
    env.close()
    assert core.calls.count(("close",)) == 1


def test_optional_pettingzoo_parallel_api():
    pettingzoo_test = pytest.importorskip("pettingzoo.test")
    env = VVCParallelAdapter(DeviceMARLAdapter(FakeCore()))
    try:
        pettingzoo_test.parallel_api_test(env, num_cycles=3)
    finally:
        env.close()
