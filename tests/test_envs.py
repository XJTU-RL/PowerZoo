"""活动环境入口的真实构造与转移；替换猜测类名和空实例化断言。"""

import numpy as np
import pytest
from utils.config_pipeline import ConfigLoader
from utils.environment_registry import ENVIRONMENTS
from utils.envs_tools import make_power_env


def test_package_registry_is_the_single_spec_mapping():
    from envs import ENV_REGISTRY

    assert ENV_REGISTRY is ENVIRONMENTS
    assert set(ENV_REGISTRY) == {
        "vvc",
        "smartgrid",
        "dsr",
        "district_dispatch",
        "stackelberg",
    }
    assert ENV_REGISTRY["stackelberg"].interaction == "sequential"


@pytest.mark.parametrize(
    "case", ["vvc", "smartgrid", "dsr", "dsr_13bus", "district_dispatch"]
)
def test_actual_synchronous_environment_reset_and_native_transition(case):
    cfg = ConfigLoader().load("happo", case)
    env = make_power_env(case, cfg.env_args, 0, 0)
    try:
        obs, state, available = env.reset()
        assert np.isfinite(obs).all() and np.isfinite(state).all()
        assert len(obs) == env.n_agents
        actions = []
        for i, space in enumerate(env.action_space):
            if space.__class__.__name__ == "Discrete":
                actions.append(
                    int(np.flatnonzero(available[i])[0]) if available is not None else 0
                )
            else:
                actions.append(np.zeros(space.shape))
        start = env.env.circuit.time_s
        result = env.step(actions)
        assert np.isfinite(result[0]).all() and np.isfinite(result[2]).all()
        assert np.allclose(result[2], result[2][0])
        assert env.env.circuit.time_s == pytest.approx(
            start + env.env.config.data.step_seconds
        )
    finally:
        env.close()


@pytest.mark.parametrize("case", ["stackelberg", "stackelberg_13bus"])
def test_actual_sequential_reset_and_transition(case):
    from envs.stackelberg.pricing_env import SequentialPricingEnv

    env = SequentialPricingEnv(ConfigLoader().load("sequential_ppo", case).env_args)
    try:
        obs = env.reset(load_profile_idx=0)
        followers = env.publish_price(0)
        assert len(followers) == 2 and np.isfinite(obs).all()
        _, utilities, done, info = env.respond([1, 2])
        assert len(utilities) == 3 and np.isfinite(utilities).all()
        assert not done and info["dss_converged"]
    finally:
        env.close()
