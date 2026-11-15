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
        assert dims == {23}  # 统一维度：类型 one-hot+定长槽位（异维无法堆叠）
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


@pytest.mark.parametrize("restore_kind", ["checkpoint", "sidecar"])
def test_restore_rejects_changed_local_semantics(tmp_path, restore_kind):
    """真实同维局部观测：邻域变化和缺失元数据均不得静默恢复。"""
    import json
    from copy import deepcopy
    from utils.physics_manifest import validate_checkpoint, validate_sidecar

    envs = []
    try:
        manifests = []
        for hops in (0, 2):
            env = _build(extra={"local_observation_neighborhood": hops})
            envs.append(env)
            env.reset()
            manifests.append(env.env.get_physics_manifest())
        saved, changed = manifests
        assert envs[0].observation_space[0].shape == envs[1].observation_space[0].shape
        assert saved["observation_schema_sha256"] == changed["observation_schema_sha256"]
        assert saved["local_observation_schema"] != changed["local_observation_schema"]
        model = tmp_path / "actor.pt"
        path = (tmp_path / "physics.manifest.json" if restore_kind == "checkpoint"
                else model.with_suffix(".physics.json"))
        path.write_text(json.dumps(saved))
        restore = (lambda current: validate_checkpoint(tmp_path, current)
                   if restore_kind == "checkpoint" else validate_sidecar(model, current))
        restore(deepcopy(saved))
        with pytest.raises(ValueError, match="local_observation_schema"):
            restore(changed)
        for key in ("local_observation_schema", "observation_mode"):
            legacy = deepcopy(saved)
            del legacy[key]
            path.write_text(json.dumps(legacy))
            with pytest.raises(ValueError, match=key):
                restore(saved)
            path.write_text(json.dumps(saved))
        mode_changed = deepcopy(saved)
        mode_changed["observation_mode"] = "global"
        with pytest.raises(ValueError, match="observation_mode"):
            restore(mode_changed)
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("monitor,tap,remote,expected", [
    (2, 2, "", "814r"),
    (1, 2, "", "814"),
    (2, 1, "", "814r"),
    (2, 1, "850.2", "850"),
])
def test_regulator_observes_monitor_bus_and_actuates_tap_winding(
    tmp_path, monitor, tap, remote, expected
):
    """原始34Bus与独立监测/调档/远端配置：零跳电压、动作和清单一致。"""
    from copy import deepcopy
    from pathlib import Path
    import shutil
    from utils.physics_manifest import compare_manifests

    source = Path(__file__).resolve().parents[1] / "node_systems" / "34Bus_PV"
    model = tmp_path / "model"
    shutil.copytree(source, model)
    if (monitor, tap, remote) != (2, 2, ""):
        with (model / "network.dss").open("a") as stream:
            stream.write(f'\nEdit RegControl.creg1a winding={monitor} '
                         f'tapwinding={tap} bus="{remote}"\n')
    env = _build(extra={"dss_folder": str(model), "local_observation_neighborhood": 0})
    try:
        env.reset()
        core = env.env
        dev = core.circuit.regulators["creg1a"]
        assert (dev.winding, dev.tap_winding, dev.monitored_bus) == (monitor, tap, remote)
        index = core._agent_devices().index(("regcontrol", "creg1a"))
        assert core._device_bus("regcontrol", "creg1a") == expected
        assert core._local_neighborhoods()[index] == (expected, {expected})
        voltage = core.physical_state.voltages_pu[expected]
        vector = core.agent_local_observation(index)
        np.testing.assert_allclose(vector[5:8], [voltage.get(p, 0) for p in (1, 2, 3)])
        assert vector[8] == pytest.approx(min(voltage.values()))
        assert vector[9] == pytest.approx(max(voltage.values()))
        saved = core.get_physics_manifest()
        schema = saved["local_observation_schema"][index]
        action_entry = saved["action_schema"][index]
        assert action_entry["element"] == "RegControl.creg1a"
        assert action_entry["actuator"] == {
            "element": "Transformer.reg1a", "winding": tap,
        }
        # 旧摘要只绑定动作空间，不能证明调档目标；独立构造旧格式以验证拒绝。
        import json
        from hashlib import sha256
        legacy = deepcopy(saved)
        for entry in legacy["action_schema"]:
            entry.pop("actuator", None)
        legacy["action_schema_sha256"] = sha256(
            json.dumps(legacy["action_schema"], sort_keys=True).encode()
        ).hexdigest()
        with pytest.raises(ValueError, match="action_schema_sha256"):
            compare_manifests(legacy, saved)
        assert schema["voltage_bus"] == expected
        assert schema["neighborhood_buses"] == [expected]
        assert schema["regcontrol_reference"] == {
            "monitor_winding": monitor, "tap_winding": tap, "remote_bus": remote,
        }
        incompatible = deepcopy(saved)
        incompatible["local_observation_schema"][index]["voltage_bus"] = "wrong-side"
        with pytest.raises(ValueError, match="local_observation_schema"):
            compare_manifests(saved, incompatible)
        core.circuit.begin_interval(core.circuit.time_s)
        native = core.circuit.native.Transformers
        native.Name, native.Wdg = dev.transformer, 3 - tap
        other_before = native.Tap
        target = dev.initial + (dev.maxtap - dev.mintap) / dev.numtaps
        core.circuit.apply({"RegControl.creg1a": {"tap": target}})
        assert core.circuit.regulator_tap("creg1a") == pytest.approx(target)
        native.Name, native.Wdg = dev.transformer, 3 - tap
        assert native.Tap == other_before
        core.circuit.solve()
        core.circuit.finish_interval()
        assert core.circuit.trace["RegControl.creg1a"]["measured"]["tap"] == pytest.approx(target)
        env.reset()
        assert core.circuit.regulator_tap("creg1a") == dev.initial
        assert core.local_observation_schema()[index] == schema
    finally:
        env.close()
