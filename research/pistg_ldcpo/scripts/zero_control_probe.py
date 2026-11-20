"""零控制标定探针：字面口径下核对论文 Fig.9(a) 工作区间。

跑 N 个 train 日零控制（κ=0、CBESS 静默），报告：
- 日 Vmax/Vmin 均值与极值（论文 Fig.9(a) 基线工作区间：白天 Vmax 至
  1.05+、夜间 Vmin ~0.90-0.95）；
- 日累计 VD/VN/LO 与折扣口径（预算估计同式）；
- 日成本（price_level 可扫，锚定零控制 ≈1400 AUD 的 M2 准则）；
- 午间反送功率与可用 PV 能量（对照 Fig.5(b) 1390 kWh/母线/日与
  Table VI 反推 4.0 MWh/日全网的矛盾幅度）。

用法：python -m research.pistg_ldcpo.scripts.zero_control_probe [--days 5] [--price-levels 0.55]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..algo.ldcppo import COST_KEYS
from ..config import CASE_FOLDER, RunConfig
from ..data.profile_store import ProfileStore
from ..environment.cbes_env import CBESEnv


def probe(price_level: float, days: int, seed: int = 2026,
          data_dir=None) -> dict:
    cfg = RunConfig()
    cfg.scenario.price_level = price_level
    if data_dir is not None:
        cfg.data_dir = data_dir
    store = ProfileStore(cfg.data_dir)
    env = CBESEnv(CASE_FOLDER, store, cfg.scenario)
    rng = np.random.default_rng(seed + 1)
    action = np.zeros(env.action_dim)
    action[8:34] = -1.0  # κ=0
    rows = []
    for _ in range(days):
        env.reset(rng, "train")
        feeder_p = []
        pv_energy = 0.0
        vmax_day = -np.inf
        vmin_day = np.inf
        midday_reverse = -np.inf
        night_peak_load = -np.inf
        for t in range(env.steps_per_day):
            _, _, _, _, info = env.step(action)
            feeder_p.append(float(info["p_pv_kw"].sum()
                                  - env.profile.p_load_kw[t].sum()))
            pv_energy += float(info["pv_available_kw"].sum()) * 5.0 / 60.0
            vmax_day = max(vmax_day, float(info["v_pu"].max()))
            vmin_day = min(vmin_day, float(info["v_pu"].min()))
            net = float(info["p_pv_kw"].sum()
                       - env.profile.p_load_kw[t].sum())
            if 108 <= t < 180:  # 9:00-15:00 高 PV 时段
                midday_reverse = max(midday_reverse, net)
            elif t < 72 or t >= 264:  # 18:00-6:00 夜间
                night_peak_load = max(night_peak_load, -net)
        m = env.day_metrics
        rows.append({
            "vmax": vmax_day, "vmin": vmin_day,
            "VD": m["VD"], "VN": m["VN"], "LO": m["LO"],
            "cost_aud": -m["reward"],
            "midday_reverse_mw": midday_reverse / 1000.0,
            "night_peak_load_mw": night_peak_load / 1000.0,
            "pv_energy_mwh": pv_energy / 1000.0,
        })
    env.close()

    def agg(key):
        vals = np.array([r[key] for r in rows])
        return {"mean": float(vals.mean()), "min": float(vals.min()),
                "max": float(vals.max())}

    return {
        "price_level": price_level,
        "days": days,
        "vmax": agg("vmax"), "vmin": agg("vmin"),
        "VD": agg("VD"), "VN": agg("VN"), "LO": agg("LO"),
        "cost_aud": agg("cost_aud"),
        "midday_reverse_mw": agg("midday_reverse_mw"),
        "night_peak_load_mw": agg("night_peak_load_mw"),
        "pv_energy_mwh_per_day": agg("pv_energy_mwh"),
        "note": "论文口径参照：Fig.9(a) 基线白天 Vmax≈1.05+/夜 Vmin≈0.90-0.95；"
                "零控制日成本标定目标≈1400 AUD；Fig.5(b) PV 柱≈1390 kWh/母线/日"
                "（×26≈36 MWh/日）vs Table VI 反推≈4.0 MWh/日",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--price-levels", type=float, nargs="+", default=[0.55])
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    for pl in args.price_levels:
        report = probe(pl, args.days, data_dir=args.data_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
