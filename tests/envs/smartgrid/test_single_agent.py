"""活动集中控制接口：共享配置、真实 DSS 和 Gymnasium 的回合契约。"""

from dataclasses import asdict
import gymnasium as gym
import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass
from envs.smartgrid.single_agent import SingleAgentVVCEnv, SmartGridConfig


@pytest.fixture
def config():
    cfg = ConfigLoader().load(
        experiment="configs/experiments/smartgrid_synthetic_smoke.yaml",
        overrides={"env.max_episode_steps": 4},
    )
    return build_dataclass(SmartGridConfig, cfg.env_args)


def test_seeded_reset_and_gymnasium_episode(config):
    env = SingleAgentVVCEnv(config)
    try:
        a, info_a = env.reset(seed=42)
        b, info_b = env.reset(seed=42)
        np.testing.assert_allclose(a, b, atol=1e-6)
        for key, value in info_a.items():
            if isinstance(value, (float, int)):
                assert info_b[key] == pytest.approx(value, rel=1e-5, abs=1e-4)
            else:
                assert info_b[key] == value
        for i in range(4):
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape)
            )
            assert env.observation_space.contains(obs) and np.isfinite(reward)
            assert terminated == (i == 3) and not truncated
            assert info["dss_converged"]
        with pytest.raises(RuntimeError):
            env.step(np.zeros(env.action_space.shape))
    finally:
        env.close()


def test_discrete_baseline_uses_product_free_multidiscrete(config):
    config.pv.action_num = 5
    env = SingleAgentVVCEnv(config, action_space_type="discrete")
    try:
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(env.action_space.sample())
        assert np.isfinite(reward) and info["dss_converged"]
    finally:
        env.close()


def test_public_adapter_rejects_independent_legacy_config(config):
    with pytest.raises(TypeError, match="SmartGridConfig"):
        SingleAgentVVCEnv(asdict(config))
