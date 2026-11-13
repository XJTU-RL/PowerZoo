"""大系统（8500/9500-Node）环境可构造性验收；慢测试，需要 OpenDSS。

8500/9500 单步潮流约 0.2~0.4s，构造含 compile 约 1~3s；测试步数刻意截短。
"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.envs_tools import make_power_env

pytestmark = [pytest.mark.slow, pytest.mark.requires_opendss]


def _build(system_ref, extra=None):
    kwargs = {"system_ref": system_ref}
    if extra:
        kwargs["values"] = {"env": extra}
    cfg = ConfigLoader().load("happo", "vvc", **kwargs)
    return make_power_env("vvc", cfg.env_args, 0, 0)


def _baseline_actions(env, available):
    actions = []
    for i, space in enumerate(env.action_space):
        if space.__class__.__name__ == "Discrete":
            actions.append(int(np.flatnonzero(available[i])[0]))
        else:
            actions.append(np.zeros(space.shape))
    return actions


@pytest.mark.parametrize("system_ref", ["8500Node", "9500Node", "9500Node_bal"])
def test_large_system_construct_reset_and_step(system_ref):
    env = _build(system_ref)
    try:
        obs, state, available = env.reset()
        assert np.isfinite(obs).all() and np.isfinite(state).all()
        manifest = env.env.get_physics_manifest()
        assert len(manifest["observation_schema"]) == obs[0].shape[0]
        actions = _baseline_actions(env, available)
        result = env.step(actions)
        assert np.isfinite(result[0]).all() and np.isfinite(result[2]).all()
    finally:
        env.close()


def test_9500_observation_projection_cuts_dimension_without_reward_change():
    full = _build("9500Node")
    projected = _build("9500Node", {"observation_neighborhood": 2})
    try:
        obs_full, _, available = full.reset()
        obs_proj, _, _ = projected.reset()
        assert obs_proj[0].shape[0] < obs_full[0].shape[0] / 4, (
            "邻域 k=2 投影应把观测降到全量的四分之一以下"
        )
        actions = _baseline_actions(full, available)
        reward_full = float(np.mean(full.step(actions)[2]))
        reward_proj = float(np.mean(projected.step(actions)[2]))
        assert reward_full == pytest.approx(reward_proj, rel=1e-12), (
            "投影不应改变奖励链"
        )
    finally:
        full.close()
        projected.close()
