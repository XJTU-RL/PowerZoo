"""奖励、约束成本与动作映射（论文式(1)、(24)-(26) 的逐条实现）。

单位约定：功率 kW/kvar，能量 kWh，时间步 Δt=300 s；线路视在容量 kVA；
电压 p.u.。c_VD 单位 p.u.，c_VN 为次数，c_LO 用相对越限百分数
（表 III 口径 "%/(5min·line)"，式(24f) 的绝对量乘 100 换算）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COST_KEYS = ("VD", "VN", "LO")


@dataclass(frozen=True)
class CBESSParams:
    """单台 CBESS 参数（论文 Sec. IV-A：1000 kWh、±250 kW、±50 kVAr、η=0.95）。"""

    capacity_kwh: float = 1000.0
    soc_min: float = 0.1
    soc_max: float = 0.9
    p_limit_kw: float = 250.0
    q_limit_kvar: float = 50.0
    efficiency: float = 0.95
    s_rating_kva: float = (250.0**2 + 50.0**2) ** 0.5


def soc_aware_limits(energy_kwh: float, params: CBESSParams, dt_hours: float):
    """式(25) 语境下的 SOC 感知充/放上限：保证式(1b) 下一时刻 SOC 在界内。

    充电：η·p_ch·Δt ≤ SOC_max·E_cap − E；放电：p_dis/η·Δt ≤ E − SOC_min·E_cap。
    """
    headroom = max(params.soc_max * params.capacity_kwh - energy_kwh, 0.0)
    p_ch_max = min(params.p_limit_kw, headroom / (params.efficiency * dt_hours))
    room = max(energy_kwh - params.soc_min * params.capacity_kwh, 0.0)
    p_dis_max = min(params.p_limit_kw, room * params.efficiency / dt_hours)
    return p_ch_max, p_dis_max


def map_cbess_action(
    alpha_p: float,
    alpha_q: float,
    energy_kwh: float,
    params: CBESSParams,
    dt_hours: float,
):
    """式(25)(26)：α∈[-1,1] → (p_CB, q_CB)，p_CB>0 为放电（注入）。

    线性映射到 SOC 感知上下界保证充放互斥与速率界；q 按 S_CB 圆约束缩放
    并叠加论文文字口径的 ±50 kVAr 硬界（论文未明说二者关系，偏离文档记录）。
    """
    p_ch_max, p_dis_max = soc_aware_limits(energy_kwh, params, dt_hours)
    mid = (p_dis_max - p_ch_max) / 2.0
    half = (p_dis_max + p_ch_max) / 2.0
    p_cb = mid + half * float(np.clip(alpha_p, -1.0, 1.0))
    q_circle = np.sqrt(max(params.s_rating_kva**2 - p_cb**2, 0.0))
    q_cb = float(np.clip(alpha_q, -1.0, 1.0)) * float(q_circle)
    q_cb = float(np.clip(q_cb, -params.q_limit_kvar, params.q_limit_kvar))
    return p_cb, q_cb


def cbess_energy_update(
    energy_kwh: float, p_cb_kw: float, params: CBESSParams, dt_hours: float
) -> float:
    """式(1a)：E' = E + η·p_ch·Δt − p_dis/η·Δt。"""
    delta = (
        params.efficiency * max(-p_cb_kw, 0.0)
        - max(p_cb_kw, 0.0) / params.efficiency
    ) * dt_hours
    return float(
        np.clip(energy_kwh + delta, params.soc_min * params.capacity_kwh,
                params.soc_max * params.capacity_kwh)
    )


def map_pv_action(
    alpha_kappa: float, alpha_q: float, pv_available_kw: float, s_max_kva: float
):
    """式(2)(3)：κ=(α_κ+1)/2∈[0,1] 弃光比；p 优先、q 受剩余圆约束。

    修订（PR #65 复审 P2）：决策弃光后的有功自身也受视在额定约束，
    此前不设上限会使可用出力超出额定时 p 越出圆。现物理截断 p 于
    s_max 并把截断计入有效弃光比 κ_eff，保证 (p, q) 始终满足式(3)。
    """
    kappa = (float(np.clip(alpha_kappa, -1.0, 1.0)) + 1.0) / 2.0
    p_pv = min((1.0 - kappa) * pv_available_kw, float(s_max_kva))
    q_circle = np.sqrt(max(s_max_kva**2 - p_pv**2, 0.0))
    q_inv = float(np.clip(alpha_q, -1.0, 1.0)) * float(q_circle)
    if pv_available_kw > 0:
        kappa_eff = 1.0 - p_pv / pv_available_kw
    else:
        kappa_eff = 0.0
    return p_pv, q_inv, kappa_eff


def economic_reward(
    p_load_kw: np.ndarray,
    p_pv_kw: float,
    p_cbess_kw: np.ndarray,
    price_aud_per_mwh: float,
    degradation_aud_per_kwh: float,
    dt_hours: float,
) -> float:
    """式(24c)：能量成本（净购电）+ 退化成本，返回负成本奖励。"""
    net_import_kw = float(p_load_kw.sum() - p_pv_kw - p_cbess_kw.sum())
    energy_cost = net_import_kw * dt_hours * price_aud_per_mwh / 1000.0
    degradation = float(np.abs(p_cbess_kw).sum()) * dt_hours * degradation_aud_per_kwh
    return -(energy_cost + degradation)


def constraint_costs(
    v_pu: np.ndarray,
    branch_s_kva: np.ndarray,
    branch_limit_kva: np.ndarray,
    voltage_min: float,
    voltage_max: float,
) -> dict[str, float]:
    """式(24d)-(24f)：VD=越限幅度之和(p.u.)，VN=越限母线数，LO=过载率之和(%)。"""
    over = np.maximum(v_pu - voltage_max, 0.0)
    under = np.maximum(voltage_min - v_pu, 0.0)
    vd = float((over + under).sum())
    vn = float(np.count_nonzero((v_pu > voltage_max) | (v_pu < voltage_min)))
    overload = np.maximum(branch_s_kva / branch_limit_kva - 1.0, 0.0)
    lo = float(overload.sum() * 100.0)
    return {"VD": vd, "VN": vn, "LO": lo}
