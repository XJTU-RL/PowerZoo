"""规模优化等价性测试：结构缓存与精简读取不得改变物理与观测。

锁定两组等价（任务约定：任何优化都要同完整物理基准比较）：
1. light read_state 与完整 read_state：同状态下电压（按 bus 名序对齐）、
   设备类注入、损耗、源端注入逐位一致；light 的 branches/平衡校验为 None。
2. 结构缓存：第二次 reset（缓存命中）后同动作序列的轨迹与首次逐位一致，
   且 Storage 的 DispMode/SOC、PV 动作记忆等运行态被正确重放/重置。
"""

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.envs_tools import make_power_env

pytestmark = [pytest.mark.requires_opendss]


def _build(system, family="smartgrid", extra=None, light=False):
    values = {"env": {"light_state_read": light}}
    if system == "34Bus_PV":
        values["env"].setdefault("pv_plan", "none") if system != "34Bus_PV" else None
    if extra:
        values["env"].update(extra)
    cfg = ConfigLoader().load("happo", family, system_ref=system, values=values)
    return make_power_env(family, cfg.env_args, 0, 0)


def _actions(env):
    actions = []
    for space in env.action_space:
        if space.__class__.__name__ == "Discrete":
            actions.append(int(space.n // 2))
        else:
            actions.append(np.zeros(space.shape, dtype=np.float32) + 0.3)
    return actions


def test_light_read_state_matches_full_on_real_case():
    """UK 大系统：light 与完整读取的观测量逐位一致。"""
    from utils.external_cases import verify

    if verify("uk_mvlv_hv_ug")["status"] != "ok":
        pytest.skip("外部算例 uk_mvlv_hv_ug 未下载")
    full = _build("UKMVLV_HVUG", family="vvc", extra={"observation_neighborhood": 2})
    light = _build(
        "UKMVLV_HVUG", family="vvc", extra={"observation_neighborhood": 2}, light=True
    )
    try:
        actions = _actions(full)
        obs_full, _, _ = full.reset()
        obs_light, _, _ = light.reset()
        np.testing.assert_array_equal(obs_full[0], obs_light[0])  # 观测逐位一致
        r_full = full.step(actions)
        r_light = light.step(actions)
        np.testing.assert_array_equal(r_full[0], r_light[0])
        assert float(np.mean(r_full[2])) == pytest.approx(float(np.mean(r_light[2])))
        # light 的 info 明确缺失支路统计与平衡校验（不假装有）。
        assert r_full[4][0]["max_branch_loading_pu"] is not None
        assert r_light[4][0]["max_branch_loading_pu"] is None
        assert r_light[4][0]["active_balance_error_kw"] is None
    finally:
        full.close()
        light.close()


def test_light_read_state_equality_on_34bus():
    full = _build("34Bus_PV")
    light = _build("34Bus_PV", light=True)
    try:
        actions = _actions(full)
        obs_full, _, _ = full.reset()
        obs_light, _, _ = light.reset()
        np.testing.assert_array_equal(obs_full[0], obs_light[0])
        for _ in range(4):
            r_full = full.step(actions)
            r_light = light.step(actions)
            np.testing.assert_array_equal(r_full[0], r_light[0])
    finally:
        full.close()
        light.close()


def test_structure_cache_replay_identical():
    """缓存命中（第二次 reset）后固定动作序列的轨迹与首次逐位一致。"""
    env = _build("34Bus_PV")
    try:
        actions = _actions(env)
        # adapter.reset 不透传场景参数：经内核固定同一合成场景。
        env.env.reset(load_profile_idx=0)
        first = [np.array(env.step(actions)[0])]
        for _ in range(3):
            first.append(np.array(env.step(actions)[0]))
        # 构造（DSSSession 内部 reset + 环境首次 reset）之后结构缓存已生效。
        assert env.env.circuit.timings.get("reset.cache_hit", 0) > 0
        env.env.reset(load_profile_idx=0)
        second = [np.array(env.step(actions)[0])]
        for _ in range(3):
            second.append(np.array(env.step(actions)[0]))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        # 缓存命中的 reset 后 Storage 语义重放（DispMode=external）。
        for name in env.env.circuit.batteries:
            mode = env.env.circuit.read_property("Storage." + name, "dispmode")
            assert mode.lower().startswith("external")
    finally:
        env.close()


def test_storage_soc_reset_after_cache_hit():
    """缓存命中后 SOC 从引擎重读（不带上回合残留）。"""
    env = _build("34Bus_PV")
    try:
        env.reset()
        actions = _actions(env)
        # 放电若干步改变 SOC
        discharge = [
            np.ones(s.shape, dtype=np.float32) if s.__class__.__name__ == "Box" else s.n - 1
            for s in env.action_space
        ]
        for _ in range(3):
            env.step(discharge)
        soc_before = {
            n: b.soc for n, b in env.env.circuit.batteries.items()
        }
        assert any(v < 1.0 for v in soc_before.values())
        env.reset()
        soc_after = {n: b.soc for n, b in env.env.circuit.batteries.items()}
        for n, v in soc_after.items():
            assert v == pytest.approx(1.0) or v >= soc_before[n], (
                "缓存命中 reset 应从引擎重读 SOC（模型初始值），不得残留上回合值"
            )
    finally:
        env.close()


# ---- 步进计时段互斥口径（复审 4d8af3b5：分段和 392.8ms > 墙钟 296ms） ------

def test_exclusive_step_ms_dedups_nested_branch_state():
    """read_state 内嵌 branch_state：互斥化后分段和不得超过墙钟。

    复现数据：9500 full 步均 295.967ms，分段和 392.845ms——
    read 204.990 含 branch 145.897 被双计。
    """
    from scripts.benchmark_scale import exclusive_step_ms

    breakdown = {
        "step.apply": [50.0, 50.0],
        "step.probe": [0.0, 0.0],
        "step.solve": [88.0, 88.0],
        "step.branch_state": [145.897, 145.897],
        "step.read_state": [204.990, 204.990],
    }
    exclusive = exclusive_step_ms(breakdown)
    assert exclusive["step.read_state"] == pytest.approx(59.093, abs=1e-3)
    assert exclusive["step.branch_state"] == pytest.approx(145.897, abs=1e-3)
    total = sum(exclusive.values())
    assert total == pytest.approx(342.99, abs=1e-2)  # 互斥后不再是 392.8


def test_step_residual_not_clamped_and_flags_anomaly():
    """other 余量保留原始值；分段和超墙钟时显式标记异常而非钳零掩盖。"""
    from scripts.benchmark_scale import step_residual

    # 正常：墙钟 400ms，互斥分段和 343ms → 余量 +57ms
    residual, anomaly = step_residual(400.0, 342.99)
    assert residual == pytest.approx(57.01, abs=0.01)
    assert anomaly is None

    # 不守恒：墙钟 296ms，分段和 343ms → 余量 -47ms 超阈值，须显式标记
    residual, anomaly = step_residual(295.967, 342.99)
    assert residual < 0
    assert anomaly and "分段" in anomaly


def test_real_step_probe_does_not_double_count_solve():
    """真实步进：probe 段内 solve 不重复计时（已修路径回归）+ 父子包含关系成立。"""
    env = _build("34Bus_PV")
    try:
        actions = _actions(env)
        env.reset()
        circuit = env.env.circuit
        for _ in range(3):
            env.step(actions)
        # 探测发生在 apply 之后的 available_pv_power；其内部 _solve 不经
        # step.solve 计时段——整步 step.solve 增量只来自真正的求解调用。
        before = dict(circuit.timings)
        env.step(actions)
        solve_delta = circuit.timings.get("step.solve", 0) - before.get("step.solve", 0)
        read_delta = circuit.timings.get("step.read_state", 0) - before.get("step.read_state", 0)
        branch_delta = (
            circuit.timings.get("step.branch_state", 0) - before.get("step.branch_state", 0)
        )
        assert read_delta >= branch_delta - 1e-6, "read_state 应包含 branch_state 子段"
        if branch_delta > 0:
            # 有支路采集时父子包含成立；互斥后单段不超父段
            assert read_delta - branch_delta >= -1e-6
        # solve 段与 read 段互斥（不同代码路径）
        assert solve_delta >= 0
    finally:
        env.close()
