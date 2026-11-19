"""生成 IEEE 33-bus（Baran-Wu 1989）OpenDSS 模型与场景静态描述。

数据源为 pandapower 的验证版 ``case33bw``（母线号 = pandapower 索引 + 1，
即经典 Baran-Wu 编号：主馈线 1-18，支线 2-19-22、3-23-25、6-26-33；
基准潮流 Vmin=0.9131 p.u. @ bus18、线路损耗 202.68 kW 与文献一致）。
pandapower 仅是本脚本的生成期依赖，运行时只需 dss-python。

论文场景（Huang et al., IEEE TSG 2026, Sec. IV-A）：
- 4 台 CBESS 置于母线 6/11/16/31，1000 kWh、±250 kW、q∈[-50,50] kVAr、η=0.95；
- DER 母线（PV+EV）共 26/33（≈80% 部署率），围绕 CBESS 分布：
  主馈线 4-18 + 支线 19-22 + 支线 27-33；
- PV 渗透率 90%（每 DER 母线 PV 峰值 = 0.9×该母线峰荷，由数据管线标定）。

线路额定（normamps）论文未给出，默认主干 400 A、支线 200 A，
在 M2 零控制标定中按论文 Fig.8/9 现象校准后冻结；详见
docs/pistg-repro-deviations.md。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandapower as pp
import pandapower.networks as pn

BASE_KV = 12.66
# 主干路径（经典编号 1..18）上的线路给 400 A，其余支线 200 A。
TRUNK_BUSES = set(range(1, 19))
TRUNK_NORMAMPS = 400.0
LATERAL_NORMAMPS = 200.0
# DER 母线集合：主馈线 4-18 ∪ 支线 19-22 ∪ 支线 27-33（含 4 个 CBESS 母线）。
DER_BUSES = sorted(set(range(4, 19)) | {19, 20, 21, 22} | set(range(27, 34)))
CBESS_BUSES = [6, 11, 16, 31]


def build_net():
    """返回验证过的 case33bw 与经典编号映射（classic = pp_idx + 1）。"""
    net = pn.case33bw()
    pp.runpp(net, algorithm="nr")
    if not net.converged:
        raise RuntimeError("case33bw 基准潮流未收敛，数据源异常")
    return net


def emit_network_dss(net, path: Path) -> list[dict]:
    """写出 network.dss，返回线路静态表（供 case_meta 与物理损失使用）。"""
    lines = []
    blocks = [
        "clear",
        # 电源直接建于变电站母线 bus1（论文 V0 = 1 p.u.，mvasc 置大消除内阻抗）。
        # 频率不显式设置：线路 R/X 为固定欧姆、负荷为恒 PQ，频率对潮流无影响；
        # 且本引擎版本在电路创建后 set frequency 会使求解归零（已实测）。
        f"new circuit.ieee33bus bus1=bus1.1.2.3 basekV={BASE_KV} pu=1.0"
        " angle=0 mvasc3=1000000 mvasc1=1000000",
        # 12.66 kV 非标准基值，须显式登记，否则 CalcVoltageBases 会吸附到
        # 标准 12.47 kV 档，p.u. 电压系统性偏大 1.5%（已实测）。
        "set voltagebases=[12.66]",
        "set maxiterations=100 tolerance=1e-8",
    ]
    for _, row in net.line[net.line.in_service].iterrows():
        frm, to = int(row.from_bus) + 1, int(row.to_bus) + 1
        normamps = (
            TRUNK_NORMAMPS
            if frm in TRUNK_BUSES and to in TRUNK_BUSES
            else LATERAL_NORMAMPS
        )
        name = f"line_{frm}_{to}"
        blocks.append(
            f"new line.{name} bus1=bus{frm}.1.2.3 bus2=bus{to}.1.2.3"
            f" R1={row.r_ohm_per_km:.6f} X1={row.x_ohm_per_km:.6f}"
            f" normamps={normamps:.1f}"
        )
        lines.append(
            {
                "name": name,
                "from_bus": frm,
                "to_bus": to,
                "r_ohm": float(row.r_ohm_per_km),
                "x_ohm": float(row.x_ohm_per_km),
                "normamps": normamps,
            }
        )
    for _, row in net.load.iterrows():
        bus = int(row.bus) + 1
        blocks.append(
            f"new load.load_b{bus} bus1=bus{bus}.1.2.3"
            f" kW={row.p_mw * 1000:.4f} kvar={row.q_mvar * 1000:.4f}"
            " model=1 status=fixed Vminpu=0.8 Vmaxpu=1.2"
        )
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return lines


def emit_der_overlay(path: Path) -> None:
    """DER 覆盖层：每 DER 母线一台 PV Generator，每 CBESS 母线一台 CBESS Generator。

    Generator model=1 为恒 PQ 注入，kW/kvar 由环境每步直写，
    语义严格对应论文式(2)(3)（PV 逆变器）与式(1e)（CBESS 圆约束，
    由环境侧动作映射保证，见 env/cbes_env.py）。
    """
    blocks = [
        f"new generator.pv_b{bus} bus1=bus{bus}.1.2.3 kW=0 kvar=0"
        " model=1 status=fixed Vminpu=0.8 Vmaxpu=1.2"
        for bus in DER_BUSES
    ]
    blocks += [
        f"new generator.cbes_b{bus} bus1=bus{bus}.1.2.3 kW=0 kvar=0"
        " model=1 status=fixed Vminpu=0.8 Vmaxpu=1.2"
        for bus in CBESS_BUSES
    ]
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    folder = Path(__file__).resolve().parent
    net = build_net()
    lines = emit_network_dss(net, folder / "network.dss")
    emit_der_overlay(folder / "der_overlay.dss")
    meta = {
        "case": "IEEE33Bus",
        "source": "pandapower case33bw（Baran-Wu 1989 经典编号，母线号 = 索引 + 1）",
        "base_kv_ll": BASE_KV,
        "num_buses": 33,
        "slack_bus": 1,
        "lines": lines,
        "der_buses": DER_BUSES,
        "cbess_buses": CBESS_BUSES,
        "der_deployment_ratio": round(len(DER_BUSES) / 33, 4),
        "line_rating_note": "默认主干 400 A / 支线 200 A，M2 标定后冻结",
    }
    (folder / "case_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"生成完成：{len(lines)} 条线路、{len(DER_BUSES)} 个 DER 母线、"
        f"{len(CBESS_BUSES)} 台 CBESS"
    )


if __name__ == "__main__":
    main()
