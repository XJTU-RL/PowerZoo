"""关键母线观测投影验收：默认逐位兼容、schema 稳定、奖励链不受影响。"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.envs_tools import make_power_env

pytestmark = pytest.mark.requires_opendss


def _build(values=None, system_ref="34Bus_PV"):
    kwargs = {"system_ref": system_ref}
    if values:
        kwargs["values"] = {"env": values}
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


def test_default_mode_is_full_bus_expansion():
    env = _build()
    try:
        obs, _, _ = env.reset()
        manifest = env.env.get_physics_manifest()
        assert manifest["observation_projection"] is None
        assert len(manifest["observation_schema"]) == obs[0].shape[0]
        bus_entries = sum(
            1 for n in manifest["observation_schema"] if ".voltage_pu" in n
        )
        assert bus_entries > 20, "全母线展开不应只剩个位数母线"
    finally:
        env.close()


def test_neighborhood_projection_shape_and_schema_stability():
    env = _build({"observation_neighborhood": 1})
    try:
        obs, _, _ = env.reset()
        manifest = env.env.get_physics_manifest()
        projection = manifest["observation_projection"]
        assert projection["mode"] == "neighborhood" and projection["hops"] == 1
        assert len(manifest["observation_schema"]) == obs[0].shape[0]
        stat_names = [n for n in manifest["observation_schema"] if "voltage_stat" in n]
        assert stat_names == [
            "bus.voltage_stat.min_pu",
            "bus.voltage_stat.max_pu",
            "bus.voltage_stat.mean_pu",
            "bus.voltage_stat.violation_low_ratio",
            "bus.voltage_stat.violation_high_ratio",
        ]
        sha = manifest["observation_schema_sha256"]
        env.env.reset(seed=99)
        again = env.env.get_physics_manifest()
        assert again["observation_schema_sha256"] == sha, "投影 schema 跨 reset 漂移"
    finally:
        env.close()


def test_named_bus_projection_reports_unknown_buses():
    with pytest.raises(ValueError, match="不存在的母线"):
        _build({"observation_buses": ["NO_SUCH_BUS"]})


def test_projection_and_list_are_mutually_exclusive():
    with pytest.raises(ValueError, match="互斥"):
        _build({"observation_buses": ["800"], "observation_neighborhood": 1})


def test_projection_does_not_change_reward_or_info():
    def rollout(values):
        env = _build(values)
        try:
            obs, _, available = env.reset()
            actions = _baseline_actions(env, available)
            rewards, infos = [], []
            for _ in range(3):
                result = env.step(actions)
                rewards.append(float(np.mean(result[2])))
                infos.append(
                    (result[4][0]["voltage_violation_count"],
                     round(result[4][0]["power_loss_ratio"], 12))
                )
            return rewards, infos
        finally:
            env.close()

    full_rewards, full_infos = rollout(None)
    projected_rewards, projected_infos = rollout({"observation_neighborhood": 2})
    assert np.allclose(full_rewards, projected_rewards, atol=0)
    assert full_infos == projected_infos
