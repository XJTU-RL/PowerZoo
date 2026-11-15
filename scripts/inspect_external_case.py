"""外部算例结构清单：依赖闭包 + 引擎口径盘点，输出可复现 JSON 报告。

    python scripts/inspect_external_case.py --case uk_mvlv_hv_ug
    python scripts/inspect_external_case.py --case uk_mvlv_hv_ug --structure-only

报告包含：来源（仓库/提交/SHA-256/下载时间）、引擎版本、入口依赖闭包
（逐文件 SHA-256，缺失精确到 文件:行号）、频率/电压等级/相别/母线与电气
节点数、启停负荷、线路/变压器/调压器/电容/PV/储能明细及原生控制绑定、
求解器设置与（默认）电压/功率汇总。原始模型的越限电压按原样报告。
"""

from pathlib import Path
import argparse
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="uk_mvlv_hv_ug")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--structure-only", action="store_true", help="不求解，只盘点结构")
    parser.add_argument("--output", default=None, help="报告 JSON 路径（默认 results/validation/external/）")
    args = parser.parse_args()

    from dss import DSS
    from utils.external_cases import CASES, locate, read_manifest, verify
    from envs.powergrid.case_inventory import dependency_closure, inventory

    if args.case not in CASES:
        raise SystemExit(f"未登记的外部算例：{args.case}")
    report = verify(args.case, args.data_dir)
    if report["status"] != "ok":
        raise SystemExit(f"外部算例未就绪：{report['problems']}")

    model = locate(args.case, args.data_dir)
    case = CASES[args.case]
    manifest = read_manifest(args.case, args.data_dir)

    started = time.perf_counter()
    closure = dependency_closure(model, case.master)
    ctx = DSS.NewContext()
    ctx.Text.Command = f'compile "{model / case.master}"'
    report = inventory(ctx, args.case, solved=not args.structure_only)
    report["provenance"] = {
        **manifest["source"],
        "zip_sha256": manifest["zip"]["sha256"],
        "downloaded_at": manifest["downloaded_at"],
        "extraction_file_count": manifest["extraction"]["file_count"],
        "repo_commit_note": "登记于 utils/external_cases.py，与 manifest 双向核对",
    }
    report["dependencies"] = closure
    report["host"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    report["elapsed_s"] = round(time.perf_counter() - started, 3)

    output = (
        Path(args.output)
        if args.output
        else ROOT / "results/validation/external" / f"{args.case}_inventory.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: v for k, v in report.items() if k != "_node_voltages"
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(output),
        "elapsed_s": report["elapsed_s"],
        "files_in_closure": len(closure["files"]),
        "missing_dependencies": closure["missing"],
        "num_buses": report["circuit"]["num_buses"],
        "num_nodes": report["circuit"]["num_nodes"],
        "loads": {k: v for k, v in report["loads"].items() if k != "disabled_names"},
        "voltage_summary": report.get("voltage_summary"),
        "power": report.get("power"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if closure["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
