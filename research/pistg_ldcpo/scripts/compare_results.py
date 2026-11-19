"""三方法评估结果对照（论文表 III/VI/V 口径）。

用法：python -m research.pistg_ldcpo.scripts.compare_results \
        --runs full_pistg full_stg full_flat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER_TABLE3 = {  # 论文 IEEE 33-bus 测试均值（表 III）
    "LDCPPO": {"cost": 1453.1, "VD": 0.77e-3, "VN": 0.79e-2, "LO": 1.32e-3},
    "STG-LDCPPO": {"cost": 1414.8, "VD": 0.53e-3, "VN": 0.63e-2, "LO": 1.15e-3},
    "PISTG-LDCPPO": {"cost": 1382.4, "VD": 0.25e-3, "VN": 0.37e-2, "LO": 1.13e-3},
}
METHOD_LABEL = {"full_pistg": "PISTG-LDCPPO", "full_stg": "STG-LDCPPO",
                "full_flat": "LDCPPO"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+",
                        default=["full_pistg", "full_stg", "full_flat"])
    parser.add_argument("--root", type=Path,
                        default=Path("results/pistg_ldcpo"))
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        path = args.root / run / "evaluation.json"
        if not path.exists():
            print(f"跳过（无评估结果）：{path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        u = data["paper_units"]
        rows.append({
            "run": run,
            "label": METHOD_LABEL.get(run, run),
            "cost": (data["cost_aud"]["mean"], data["cost_aud"]["std"]),
            "VD": u["VD_per_bus_step"],
            "VN": u["VN_per_bus_step"],
            "LO": u["LO_per_line_step"],
            "curt_kwh": data["curt_kwh"]["mean"],
            "curt_ratio": data["curt_ratio"]["mean"],
            "hp_ratio": data["curt_hp_ratio"]["mean"],
            "infer_ms": data["inference"]["mean_ms"],
            "params_m": data["inference"]["params_million"],
        })
    if not rows:
        return
    print("=" * 88)
    print("本复现（100 测试回合均值，论文单位）vs 论文表 III 均值")
    print("=" * 88)
    print(f"{'方法':<16}{'成本 AUD/日':>14}{'VD p.u./5min·bus':>18}"
          f"{'VN t/5min·bus':>15}{'LO %/5min·line':>16}")
    for row in rows:
        print(f"{row['label']:<16}{row['cost'][0]:>10.1f}±{row['cost'][1]:.0f}"
              f"{row['VD']:>18.4f}{row['VN']:>15.4f}{row['LO']:>16.4f}")
        paper = PAPER_TABLE3.get(row["label"])
        if paper:
            print(f"{'  (论文)':<16}{paper['cost']:>14.1f}{paper['VD']:>18.4f}"
                  f"{paper['VN']:>15.4f}{paper['LO']:>16.4f}")
    print("-" * 88)
    print("弃光（表 VI 口径）与推理开销（表 V 口径）")
    for row in rows:
        print(f"{row['label']:<16}弃光 {row['curt_kwh']:>8.1f} kWh/日 "
              f"({row['curt_ratio']:.2f}%, 高PV时段占比 {row['hp_ratio']:.1f}%) | "
              f"推理 {row['infer_ms']:.3f} ms, 参数 {row['params_m']:.3f} M")


if __name__ == "__main__":
    main()
