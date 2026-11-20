"""PISTG-LDCPPO 论文复现（Huang et al., IEEE TSG 2026）的算例与物理层测试。

M1：IEEE 33-bus OpenDSS 模型与文献基准潮流对拍；
参考值为 Baran-Wu (1989) 系统的公开文献值（Vmin=0.9131 p.u. @ bus18、
线路损耗 202.7 kW、总负荷 3715 kW + j2300 kvar），与 pandapower
case33bw 验证解一致（生成期对拍，见 build_case.py）。
"""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.powergrid.session import DSSSession  # noqa: E402

CASE_FOLDER = REPO_ROOT / "node_systems" / "33Bus_CBESS"
NETWORK = CASE_FOLDER / "network.dss"
OVERLAY = CASE_FOLDER / "der_overlay.dss"

# 文献基准（经典编号）：p.u. 电压与线路总损耗。
REFERENCE_VOLTAGES = {
    1: 1.0000,
    2: 0.9970,
    6: 0.9497,
    12: 0.9269,
    18: 0.9131,
    25: 0.9694,
    33: 0.9166,
}
REFERENCE_LINE_LOSS_KW = 202.7
REFERENCE_TOTAL_LOAD_KW = 3715.0


def _solve_case33(overlay=False):
    import dss

    ctx = dss.DSS.NewContext()
    ctx.Text.Command = f'compile "{NETWORK}"'
    if overlay:
        ctx.Text.Command = f'redirect "{OVERLAY}"'
    ctx.Text.Command = "set mode=snapshot controlmode=off"
    # 模型文件里的 voltagebases 设置在 compile 后不存活，须重设再算基值。
    ctx.Text.Command = "set voltagebases=[12.66]"
    ctx.Text.Command = "calcvoltagebases"
    ctx.ActiveCircuit.Solution.SolveSnap()
    if not ctx.ActiveCircuit.Solution.Converged:
        raise RuntimeError("33-bus 基准潮流未收敛")
    return ctx


def _bus_pu_voltages(ckt):
    values = np.zeros(33)
    for i in range(1, 34):
        ckt.SetActiveBus(f"bus{i}")
        values[i - 1] = float(np.asarray(ckt.ActiveBus.puVmagAngle)[::2].mean())
    return values


def _line_loss_kw(ckt):
    total_w = 0.0
    for element in ckt.AllElementNames:
        if element.lower().startswith("line."):
            ckt.SetActiveElement(element)
            total_w += float(np.asarray(ckt.ActiveCktElement.Losses)[0])
    return total_w / 1000.0


def test_case33_base_powerflow_matches_reference():
    """基准潮流与文献值对拍：逐母线 p.u. 电压 ±0.002，损耗 ±1 kW。"""
    ctx = _solve_case33()
    ckt = ctx.ActiveCircuit
    voltages = _bus_pu_voltages(ckt)
    for bus, expected in REFERENCE_VOLTAGES.items():
        assert voltages[bus - 1] == pytest.approx(expected, abs=2e-3), (
            f"bus{bus} 电压 {voltages[bus - 1]:.4f} 偏离文献值 {expected:.4f}"
        )
    assert int(np.argmin(voltages)) + 1 == 18, "最低电压应出现在 bus18"
    assert _line_loss_kw(ckt) == pytest.approx(REFERENCE_LINE_LOSS_KW, abs=1.0)


def test_case33_with_der_overlay_elements():
    """DER 覆盖层：26 台 PV + 4 台 CBESS Generator，求解正常。"""
    ctx = _solve_case33(overlay=True)
    ckt = ctx.ActiveCircuit
    generators = [n for n in ckt.AllElementNames if n.lower().startswith("generator.")]
    pv = [n for n in generators if ".pv_b" in n]
    cbess = [n for n in generators if ".cbes_b" in n]
    assert len(pv) == 26, f"PV Generator 应为 26 台，实际 {len(pv)}"
    assert len(cbess) == 4, f"CBESS Generator 应为 4 台，实际 {len(cbess)}"
    voltages = _bus_pu_voltages(ckt)
    assert (voltages > 0.85).all() and (voltages < 1.1).all()


def test_case33_static_meta_consistency():
    """case_meta.json 与模型文件一致：线路数、DER 母线、CBESS 母线。"""
    import json

    meta = json.loads((CASE_FOLDER / "case_meta.json").read_text(encoding="utf-8"))
    assert len(meta["lines"]) == 32
    assert meta["cbess_buses"] == [6, 11, 16, 31]
    assert len(meta["der_buses"]) == 26
    # 论文口径：PV 部署率 80%（26/33 = 78.8% ≈ 80%）。
    assert meta["der_deployment_ratio"] == pytest.approx(26 / 33, abs=1e-4)
    text = NETWORK.read_text(encoding="utf-8")
    assert text.count("new line.") == 32
    assert text.count("new load.") == 32


def test_dss_session_two_step_compatibility():
    """研究环境组合 DSSSession：两步 5-min 区间求解与时钟推进正常。"""
    from research.pistg_ldcpo.environment.session_utils import rebase_voltage

    session = DSSSession(
        CASE_FOLDER, "network.dss", overlays=("der_overlay.dss",), step_seconds=300
    )
    try:
        rebase_voltage(session)
        for step, second in enumerate((0.0, 300.0)):
            session.begin_interval(second)
            session.command("Edit Load.load_b2 kW=50 kvar=20")
            session.command("Edit Generator.pv_b6 kW=10 kvar=0")
            session.solve()
            state = session.read_state(light=True)
            # 基值正确时变电站母线应≈1.0 p.u.（牛顿容差内；基值吸附错误时
            # 会读到 1.0152，特征相差三个数量级）。
            assert np.mean(list(state.voltages_pu["bus1"].values())) == pytest.approx(
                1.0, abs=5e-5
            )
            assert 0.85 < np.mean(list(state.voltages_pu["bus18"].values())) < 1.05
            session.finish_interval()
            assert session.time_s == pytest.approx((step + 1) * 300.0, abs=1e-6)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# M2：剖面数据契约（依赖 data/external 的 Ausgrid 数据；未下载时跳过，
# 下载方式见 research/pistg_ldcpo/data/prepare_profiles.py 文档字符串）。


def _profiles_or_skip():
    path = REPO_ROOT / "data" / "external" / "ausgrid-solar-home" / "profiles.npz"
    if not path.exists():
        pytest.skip("Ausgrid 剖面未生成（需下载并运行 prepare_profiles.py）")
    import json

    data = np.load(path)
    manifest = json.loads(
        (path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    return data, manifest


def test_profiles_shapes_and_energy_targets():
    data, manifest = _profiles_or_skip()
    p, q, pv, price = (
        data["p_load_kw"],
        data["q_load_kvar"],
        data["pv_available_kw"],
        data["price_aud_per_mwh"],
    )
    assert p.shape == (data["day_keys"].shape[0], 288, 33)
    assert pv.shape[0] == p.shape[0] and pv.shape[2] == 26
    assert (p >= 0).all() and (pv >= 0).all() and (price > 0).all()
    # DER 母线日能量 = 基荷 1600 + EV 300 kWh（每母线每日，能量重标定；
    # M2 零控制标定冻结值，依据记入 docs/pistg-repro-deviations.md）。
    der = np.array(manifest["der_buses"]) - 1
    energy = p[:, :, der].sum(axis=1).mean(axis=1) * (5.0 / 60.0)
    assert energy.mean() == pytest.approx(1900.0, rel=0.01)
    # EV 日因子截断 [0.7,1.2] → 单日能量 [1810, 1960] kWh。
    assert (energy > 1800.0).all() and (energy < 1970.0).all()
    # 无功 = 有功 × tanφ（pf=0.85）。
    mask = p > 0
    ratio = (q[mask] / p[mask])
    assert np.allclose(ratio, np.tan(np.arccos(0.85)), atol=1e-3)
    # PV 静态峰值 = 0.9 × PV_BOOST(1.5) × 母线典型日峰负荷（跨日一致）。
    pv_peak = pv.max(axis=(0, 1))
    assert pv_peak.min() > 150.0 and pv_peak.max() < 230.0


def test_profiles_day_split_disjoint():
    data, manifest = _profiles_or_skip()
    split = manifest["split"]
    keys = [str(k) for k in data["day_keys"]]
    assert set(keys) == set(split.keys())
    assert set(split.values()) == {"train", "validation", "test"}
    counts = {s: sum(1 for v in split.values() if v == s)
              for s in ("train", "validation", "test")}
    assert sum(counts.values()) == len(keys)
    assert counts["train"] > 0.7 * len(keys)


def test_profile_store_roundtrip():
    _profiles_or_skip()
    from research.pistg_ldcpo.data.profile_store import ProfileStore

    store = ProfileStore(REPO_ROOT / "data" / "external" / "ausgrid-solar-home")
    rng = np.random.default_rng(0)
    for split in ("train", "validation", "test"):
        assert store.episode_count(split) > 0
    episode = store.sample_episode(rng, "train")
    assert episode.p_load_kw.shape == (288, 33)
    assert episode.pv_available_kw.shape == (288, 26)
    assert episode.price_aud_per_kwh_5min.shape == (288,)
    assert episode.price_aud_per_kwh_5min.max() < 1.0  # AUD/kWh 合理量级


# ---------------------------------------------------------------------------
# M4：PISTG 编码器与 PIAE（式(10)-(22)）。


def test_piam_values_and_encoder_shapes():
    """PIAM 数值口径（式(10)）与编码器维度（d_STG=80）。"""
    import json as _json

    import torch

    from research.pistg_ldcpo.models.pistg import (
        PISTGEncoder,
        physics_informed_adjacency,
    )

    meta = _json.loads((CASE_FOLDER / "case_meta.json").read_text(encoding="utf-8"))
    lines = meta["lines"]
    adjacency = physics_informed_adjacency(lines, 33, float(meta["base_kv_ll"]))
    # 手工核对第一条线路 line_1_2（r=0.0922, x=0.047，额定取自 case_meta）。
    impedance = float(np.hypot(0.0922, 0.0470))
    s_mva = lines[0]["normamps"] * 12.66 * np.sqrt(3.0) / 1000.0
    psi = -np.log(1e-6 + impedance) - 1.0 / s_mva
    expected = float(1.0 / (1.0 + np.exp(-psi)))
    assert float(adjacency[0, 1]) == pytest.approx(expected, rel=1e-6)
    assert float(adjacency[1, 0]) == pytest.approx(expected, rel=1e-6)  # 对称
    assert float(adjacency[0, 0]) == 0.0  # 无自环
    assert (adjacency == 0).sum() == 33 * 33 - 2 * 32  # 仅 32 条边有权重

    encoder = PISTGEncoder(num_buses=33)
    bus_window = torch.randn(2, 12, 33, 5)
    price_window = torch.randn(2, 12)
    degree = adjacency.sum(-1).clamp(min=1e-6)
    norm_adj = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
    h = encoder(bus_window, price_window, norm_adj)
    assert tuple(h.shape) == (2, 80)
    assert encoder.d_stg == 80


def test_piae_losses_overfit_single_batch():
    """PIAE 能在单个小批上把三项损失同时压低（物理损失量级正确性）。"""
    import json
    import torch

    from research.pistg_ldcpo.models.piae import PIAE, ReconstructionHead
    from research.pistg_ldcpo.models.pistg import (
        PISTGEncoder,
        physics_informed_adjacency,
    )

    meta = json.loads((CASE_FOLDER / "case_meta.json").read_text(encoding="utf-8"))
    lines = meta["lines"]
    adjacency = physics_informed_adjacency(lines, 33, float(meta["base_kv_ll"]))
    degree = adjacency.sum(-1).clamp(min=1e-6)
    norm_adj = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
    encoder = PISTGEncoder(num_buses=33)
    head = ReconstructionHead(80, 33, 32)
    stats = {"mean": np.zeros(5), "std": np.ones(5)}
    target = {k: (0.97 if k == "v_mean" else 0.02 if k == "v_std" else 300.0)
              for k in ("v_mean", "v_std", "p_mean", "p_std", "q_mean", "q_std")}
    piae = PIAE(encoder, head, lines, 33, 12.66, stats, target)

    torch.manual_seed(0)
    bw = torch.randn(16, 12, 33, 5)
    pw = torch.randn(16, 12)
    v_t = torch.randn(16, 33)
    p_t = torch.randn(16, 32)
    q_t = torch.randn(16, 32)
    inj = torch.randn(16, 33) * 0.05
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=1e-3
    )
    first = piae.loss(bw, pw, norm_adj, v_t, p_t, q_t, inj)[0]
    for _ in range(200):
        loss, _ = piae.loss(bw, pw, norm_adj, v_t, p_t, q_t, inj)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = piae.loss(bw, pw, norm_adj, v_t, p_t, q_t, inj)[0]
    # 随机注入下的 KCL/VDrop 残差有不可压下限，验证 85% 量级下降即可。
    assert float(final) < 0.15 * float(first), "单批 200 步应显著过拟合下降"
    assert float(first) < 100.0, "物理损失量级应在 O(1)-O(10)（非 1e6）"


# ---------------------------------------------------------------------------
# M5：LDCPPO（式(27)-(36)）。


def test_gae_hand_computed():
    """式(30)(31)：两步手工算例。r=[1,0]，V=[0.5,0.5]，V(T)=0，γ=1，λ=1。

    δ1 = 0 + 1×0 − 0.5 = −0.5；δ0 = 1 + 1×0.5 − 0.5 = 1；
    A1 = δ1 = −0.5；A0 = δ0 + γλ·A1 = 0.5。
    """
    from research.pistg_ldcpo.algo.ldcppo import compute_gae

    adv = compute_gae(
        np.array([1.0, 0.0]), np.array([0.5, 0.5]), 0.0, gamma=1.0, lam=1.0
    )
    assert adv[1] == pytest.approx(-0.5, abs=1e-12)
    assert adv[0] == pytest.approx(0.5, abs=1e-12)


def test_dual_ascent_direction():
    """式(36d)：超预算 λ 增、低于预算 λ 减且非负钳制。"""
    from research.pistg_ldcpo.algo.ldcppo import LDCPPO
    from research.pistg_ldcpo.config import LdcpoConfig

    agent = LDCPPO(state_dim=4, action_dim=2, cfg=LdcpoConfig())
    agent.set_budgets({"VD": 1.0, "VN": 1.0, "LO": 1.0})
    zeros = np.zeros(5)
    states = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    actions = np.zeros((5, 2), np.float32)
    # 成本 0.1/步 × 5 步（γ<1）→ J<1 → λ 保持 0（初值 0，非负钳制）。
    agent.update(
        states, actions, np.full(5, -1.0), np.ones(5) * 0.0,
        {"VD": np.full(5, 0.1), "VN": zeros, "LO": zeros},
        {"R": 0.0, "VD": 0.0, "VN": 0.0, "LO": 0.0},
    )
    assert agent.multipliers["VD"] == 0.0
    # 成本 0.5/步 → J≈2.4 > 1 → λ 增。
    agent.multipliers["VD"] = 0.1
    agent.update(
        states, actions, np.full(5, -1.0), np.ones(5) * 0.0,
        {"VD": np.full(5, 0.5), "VN": zeros, "LO": zeros},
        {"R": 0.0, "VD": 0.0, "VN": 0.0, "LO": 0.0},
    )
    assert agent.multipliers["VD"] > 0.1


def test_ldcpo_reduces_cost_in_synthetic_cmdp():
    """对抗性合成 CMDP：奖励 −(a−1)² 拉向 a=1，约束 |a| 预算 0.2 反压。

    拉格朗日机制的教科书行为：先奖励主导（成本升），λ 积累后约束反压
    （成本回落）。论文 lr 为 288 步大任务尺度，合成小任务放大 lr/对偶步长。
    """
    from research.pistg_ldcpo.algo.ldcppo import LDCPPO
    from research.pistg_ldcpo.config import LdcpoConfig

    rng = np.random.default_rng(3)
    cfg = LdcpoConfig()
    cfg.ppo_epochs = 5
    cfg.actor_lr = cfg.critic_lr = 3e-4
    cfg.dual_lr = 5e-3
    agent = LDCPPO(state_dim=2, action_dim=1, cfg=cfg)
    agent.set_budgets({"VD": 0.2, "VN": 1e9, "LO": 1e9})  # 只激活 VD 约束

    def rollout():
        states = rng.normal(size=(16, 2)).astype(np.float32)
        actions, logps, rewards, costs = [], [], [], []
        for s in states:
            a, logp = agent.act(s)
            actions.append(a)
            logps.append(logp)
            rewards.append(-(float(a) - 1.0) ** 2)
            costs.append(abs(float(a)))
        agent.update(
            states, np.stack(actions), np.array(logps), np.array(rewards),
            {"VD": np.array(costs), "VN": np.zeros(16), "LO": np.zeros(16)},
            {"R": 0.0, "VD": 0.0, "VN": 0.0, "LO": 0.0},
        )
        return float(np.mean(costs))

    costs = [rollout() for _ in range(80)]
    peak = float(np.mean(costs[30:40]))
    final = float(np.mean(costs[70:80]))
    assert agent.multipliers["VD"] > 1.0, "乘子应显著激活"
    assert final < peak, "约束应在乘子积累后压制成本（V 形轨迹）"


def test_flat_state_encoder_dimension():
    from research.pistg_ldcpo.train import StateEncoder

    encoder = StateEncoder("flat", None)
    obs = {
        "bus_features": np.ones((33, 5), np.float32),
        "price": np.float32(60.0),
    }
    encoder.reset(obs)
    state = encoder.encode()
    assert state.shape == (12 * (33 * 5 + 1),)  # 平铺窗口 1992 维


# ---------------------------------------------------------------------------
# M3：动作映射（式(1)(25)(26)）与 CBESS 环境。


def test_action_mapping_bounds():
    """α=±1 恰好触及 SOC 感知界限；式(1e) 圆约束与 ±50 kVAr 硬界成立。"""
    from research.pistg_ldcpo.environment.physics import (
        CBESSParams,
        cbess_energy_update,
        map_cbess_action,
        map_pv_action,
        soc_aware_limits,
    )

    params = CBESSParams()  # 论文默认：1000 kWh、±250 kW、±50 kVAr、η=0.95
    dt = 300.0 / 3600.0
    energy = 500.0  # SOC 50%（居中，充放均不受 SOC 限制）
    p_max, q_max = map_cbess_action(1.0, 1.0, energy, params, dt)
    p_min, _ = map_cbess_action(-1.0, 0.0, energy, params, dt)
    assert p_max == pytest.approx(250.0, abs=1e-9)
    assert p_min == pytest.approx(-250.0, abs=1e-9)
    assert abs(q_max) == pytest.approx(50.0, abs=1e-6)  # 圆约束或硬界取小

    # SOC 感知：接近下界时放电受限、充电满额。
    energy_low = params.soc_min * params.capacity_kwh + 10.0
    p_ch, p_dis = soc_aware_limits(energy_low, params, dt)
    assert p_ch == pytest.approx(250.0, abs=1e-9)
    assert p_dis == pytest.approx(10.0 * 0.95 / dt, abs=1e-6)
    # α_P=-1 在 SOC 上界附近只能充 0（无空间）。
    energy_high = params.soc_max * params.capacity_kwh
    p_zero, _ = map_cbess_action(-1.0, 0.0, energy_high, params, dt)
    assert p_zero == pytest.approx(0.0, abs=1e-9)

    # 圆约束：任意 α 组合 p²+q² ≤ S_CB²（q 硬界 50 更紧时亦然）。
    for a_p in (-1, -0.3, 0, 0.7, 1):
        for a_q in (-1, 0, 1):
            p_cb, q_cb = map_cbess_action(a_p, a_q, energy, params, dt)
            assert p_cb**2 + q_cb**2 <= params.s_rating_kva**2 + 1e-6
            assert abs(q_cb) <= 50.0 + 1e-9

    # 式(1a)：250 kW 充一步（5/60 h），能量增加 η×250×5/60。
    e_next = cbess_energy_update(500.0, -250.0, params, dt)
    assert e_next == pytest.approx(500.0 + 0.95 * 250 * 5 / 60, rel=1e-9)

    # 式(2)(3)：κ∈[0,1] 线性于 α_κ；q_inv 受视在功率圆约束。
    p_pv, q_inv, kappa = map_pv_action(-1.0, 0.0, 100.0, 110.0)
    assert kappa == 0.0 and p_pv == 100.0 and q_inv == 0.0
    p_pv, q_inv, kappa = map_pv_action(1.0, 0.0, 100.0, 110.0)
    assert kappa == 1.0 and p_pv == 0.0
    p_pv, q_inv, kappa = map_pv_action(0.0, 1.0, 100.0, 110.0)
    assert kappa == 0.5 and p_pv == 50.0
    assert p_pv**2 + q_inv**2 <= 110.0**2 + 1e-6


def test_constraint_costs_and_reward_values():
    """式(24c)-(24f) 手工算例：单位与口径逐项核对。"""
    from research.pistg_ldcpo.environment.physics import (
        constraint_costs,
        economic_reward,
    )

    v = np.array([1.06, 0.94, 1.00, 0.95, 1.05])  # 越上界 0.01、越下界 0.01
    s = np.array([110.0, 90.0, 100.0])  # 10% 过载一条
    costs = constraint_costs(v, s, np.full(3, 100.0), 0.95, 1.05)
    assert costs["VD"] == pytest.approx(0.02, abs=1e-12)
    assert costs["VN"] == 2.0
    assert costs["LO"] == pytest.approx(10.0, abs=1E-12)  # (110/100-1)×100

    reward = economic_reward(
        p_load_kw=np.array([100.0, 50.0]),
        p_pv_kw=30.0,
        p_cbess_kw=np.array([10.0, -10.0]),  # 净 0
        price_aud_per_mwh=60.0,
        degradation_aud_per_kwh=0.03,
        dt_hours=5.0 / 60.0,
    )
    # 能量成本 = 120 kW × 5/60 h × 0.06 AUD/kWh = 0.6；退化 = 20 kW×5/60×0.03。
    assert reward == pytest.approx(-(0.6 + 0.05), abs=1e-9)


def test_zero_control_day_reproduces_paper_phenomena():
    """零控制（κ=0、q=0、CBESS 静止）全天回放：白天过压 + 夜间欠压 + 成本量级。

    M2 标定冻结的验收实验；参数依据见 docs/pistg-repro-deviations.md。
    """
    _profiles_or_skip()
    from research.pistg_ldcpo.config import CASE_FOLDER, RunConfig
    from research.pistg_ldcpo.data.profile_store import ProfileStore
    from research.pistg_ldcpo.environment.cbes_env import CBESEnv

    cfg = RunConfig()
    store = ProfileStore(cfg.data_dir)
    env = CBESEnv(CASE_FOLDER, store, cfg.scenario)
    try:
        assert env.action_dim == 60  # 4×2 CBESS + 26×2 PV
        rng = np.random.default_rng(7)
        env.reset(rng, "test")
        action = np.zeros(env.action_dim)
        action[8:34] = -1.0  # κ=0 完全不弃光
        vmax_day, vmin_night = 0.0, 1.0
        for t in range(288):
            obs, reward, costs, done, info = env.step(action)
            assert set(costs) == {"VD", "VN", "LO"}
            if 96 <= t < 192:
                vmax_day = max(vmax_day, float(info["v_pu"].max()))
            if t < 72 or t >= 264:
                vmin_night = min(vmin_night, float(info["v_pu"].min()))
        metrics = env.day_metrics
        assert vmax_day > 1.03, "白天应出现论文量级的过压现象"
        assert vmin_night < 0.93, "夜间应出现欠压现象"
        cost_aud = -metrics["reward"]
        assert 500 < cost_aud < 3000, "零控制日成本应在论文表 III 量级附近"
        assert metrics["curtailed_kwh"] < 1e-6  # 零控制不弃光
    finally:
        env.close()
