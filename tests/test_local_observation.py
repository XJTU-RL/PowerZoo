"""局部 actor 观测（CTDE）测试：维度/边界/manifest 与集中状态分离。"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.envs_tools import make_power_env

pytestmark = [pytest.mark.requires_opendss]


def _build(mode="local", extra=None):
    values = {"env": {"observation_mode": mode, "max_episode_steps": 8}}
    if extra:
        values["env"].update(extra)
    cfg = ConfigLoader().load("happo", "smartgrid", system_ref="34Bus_PV", values=values)
    return make_power_env("smartgrid", cfg.env_args, 0, 0)


def _actions(env):
    return [
        s.n // 2 if s.__class__.__name__ == "Discrete"
        else np.zeros(s.shape, dtype=np.float32) + 0.3
        for s in env.action_space
    ]


def test_local_mode_shapes_and_shared_state():
    env = _build()
    try:
        obs, state, _ = env.reset()
        assert len(env.observation_space) == len(env.action_space)
        dims = {space.shape[0] for space in env.observation_space}
        assert dims == {12, 13, 18}  # 电容/调压 12、储能/SVG 13、PV 18
        assert all(space.shape != state[0].shape for space in env.observation_space)
        # 集中 critic 状态=全局 wrap_obs，与 global 模式同源同维。
        assert state[0].shape == env.share_observation_space[0].shape
        result = env.step(_actions(env))
        assert all(np.isfinite(o).all() for o in result[0])
        manifest = env.env.get_physics_manifest()
        assert manifest["observation_mode"] == "local"
        assert len(manifest["local_observation_schema"]) == len(env.action_space)
        assert {s["element"].split(".")[0] for s in manifest["local_observation_schema"]} == {
            "capacitor", "regcontrol", "storage", "pvsystem", "generator"
        } & {s["element"].split(".")[0] for s in manifest["local_observation_schema"]}
    finally:
        env.close()


def test_local_information_boundary():
    """屏蔽邻域外母线电压不得改变任何局部观测；邻域内篡改必须改变。"""
    env = _build()
    try:
        env.reset()
        core = env.env
        base = [core.agent_local_observation(i) for i in range(len(core.action_space))]
        views = core._local_neighborhoods()
        allowed = set()
        for _, neighborhood in views:
            allowed |= neighborhood
        source = core.physical_state.voltages_pu
        outside = {
            bus for bus in source if bus not in allowed
        }
        assert outside, "测试假设存在邻域外母线"

        def rebuild(tamper_inside):
            return type(core.physical_state)(
                core.physical_state.time_s,
                {
                    bus: (
                        {p: 0.5 for p in phases}
                        if (bus in allowed) == tamper_inside
                        else dict(phases)
                    )
                    for bus, phases in source.items()
                },
                core.physical_state.injections,
                core.physical_state.grid_import,
                core.physical_state.losses,
                core.physical_state.storage_soc,
                core.physical_state.branches,
            )

        core.physical_state = rebuild(tamper_inside=False)
        outside_tampered = [
            core.agent_local_observation(i) for i in range(len(core.action_space))
        ]
        for i, (a, b) in enumerate(zip(base, outside_tampered)):
            np.testing.assert_array_equal(
                a, b, err_msg=f"智能体{i}的局部观测泄漏了邻域外电压"
            )
        core.physical_state = rebuild(tamper_inside=True)
        inside_tampered = [
            core.agent_local_observation(i) for i in range(len(core.action_space))
        ]
        assert any(
            not np.array_equal(a, b) for a, b in zip(base, inside_tampered)
        ), "阳性对照失败：邻域内篡改应改变局部观测"
    finally:
        env.close()


def test_local_mode_stable_across_reset():
    """局部观测维度与邻域跨 reset 稳定（schema 对齐前提）。"""
    env = _build()
    try:
        env.reset()
        dims_before = [s.shape[0] for s in env.observation_space]
        views_before = [sorted(n) for _, n in env.env._local_neighborhoods()]
        env.reset()
        assert [s.shape[0] for s in env.observation_space] == dims_before
        assert [sorted(n) for _, n in env.env._local_neighborhoods()] == views_before
    finally:
        env.close()


def test_global_mode_unchanged_by_local_infrastructure():
    """global 模式：actor 观测与 critic 状态同源（旧行为），局部方法仍可用。"""
    env = _build(mode="global")
    try:
        obs, state, _ = env.reset()
        assert obs[0].shape == state[0].shape
        manifest = env.env.get_physics_manifest()
        assert manifest["observation_mode"] == "global"
        assert manifest["local_observation_schema"] is None
        local = env.env.agent_local_observation(0)  # 基础设施常在，不改变 global 行为
        assert local.shape == (env.env.local_observation_dim(0),)
    finally:
        env.close()


def test_invalid_observation_mode_rejected():
    from utils.config_pipeline import ConfigError

    with pytest.raises(Exception) as error:
        _build(mode="everywhere")
    assert "观测模式" in str(error.value) or isinstance(error.value, ConfigError)
