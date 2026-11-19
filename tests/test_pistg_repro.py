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

CASE_FOLDER = REPO_ROOT / "research" / "pistg_ldcpo" / "case" / "ieee33bus"
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
    from research.pistg_ldcpo.env.session_utils import rebase_voltage

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
