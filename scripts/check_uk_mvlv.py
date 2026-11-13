"""UK MV-LV 外部算例物理基线：引擎直载（A）与 PowerZoo 会话（B）对照。

A1：DSS 引擎按模型原文编译（主入口自带 solve，求解器为引擎默认设置）。
A2：同一引擎编译后，显式对齐 DSSSession 的求解设置
    （snapshot/static 控制、maxiterations=100、tolerance=1e-8）再求解。
B ：envs.powergrid.DSSSession 加载同一模型（临时副本 + SHA256 身份），
    solve(native_controls=True) 保留原生 RegControl——与 A2 同设置对照。

比较口径：逐节点 pu 电压、源端 P/Q、损耗（Circuit.Losses 与逐支路求和双
口径）、元件计数、线路开断、调压器档位、收敛与迭代次数。公差依据：同引擎
同模型两处编译的数值路径一致，电压应达浮点噪声量级（<=2e-6 pu），功率相对
差 <=1e-6；超出即报 mismatch 并保留差异明细，不静默放行。

性能段为一次性量测（含机器与口径说明），不代表 RL 训练吞吐。输出写入
results/validation/external/uk_mvlv_hv_ug_baseline.json，match 与否决定退出码。
"""

from pathlib import Path
import argparse
import json
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SESSION_MAXITER = 100
SESSION_TOLERANCE = "1e-8"


def _peak_rss_gb():
    # darwin 的 ru_maxrss 单位是字节，Linux 是 KiB；按平台归一到 GB。
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024**3 if sys.platform == "darwin" else raw / 1024**2 / 1024


def solve_engine_snap(ctx, label):
    """镜像 DSSSession.solve(native_controls=True) 的求解循环（供 A2 使用）。"""
    solution = ctx.ActiveCircuit.Solution
    solution.ControlMode = 0  # static：原生 RegControl 参与
    for _ in range(12):
        solution.SolveSnap()
        if not solution.Converged:
            raise RuntimeError(f"{label} 未收敛")
        if not solution.ControlActionsDone:
            raise RuntimeError(f"{label} 原生控制循环未完成")
        break
    return {"converged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-probes", action="store_true", help="跳过性能与扰动探针")
    args = parser.parse_args()

    from dss import DSS
    from utils.external_cases import CASES, locate
    from envs.powergrid.case_inventory import (
        compare,
        inventory,
        node_voltages,
    )
    from envs.powergrid.session import DSSSession

    case_id = "uk_mvlv_hv_ug"
    case = CASES[case_id]
    model = locate(case_id, args.data_dir)
    master = model / case.master

    report = {
        "case_id": case_id,
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "method": {
            "timer": "time.perf_counter（单进程顺序量测）",
            "peak_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss（进程峰值，单调）",
            "note": "热启动重复求解=同工况再解；扰动求解=Set loadmult 1.01 后再解；"
            "静态求解时间不能当作 RL 训练吞吐",
        },
    }

    # ---- A1：引擎按原文编译（含入口自带 solve，引擎默认求解设置） ----
    ctx_a1 = DSS.NewContext()
    t0 = time.perf_counter()
    ctx_a1.Text.Command = f'compile "{master}"'
    report["timing_a1_compile_with_embedded_solve_s"] = round(
        time.perf_counter() - t0, 3
    )
    a1 = inventory(ctx_a1, "A1 引擎直载（模型默认设置）")
    a1["_node_voltages"] = node_voltages(ctx_a1)

    # ---- A2：同引擎，显式对齐 DSSSession 求解设置 ----
    ctx_a2 = DSS.NewContext()
    ctx_a2.Text.Command = f'compile "{master}"'
    for command in (
        "Set mode=snapshot controlmode=static",
        "CalcVoltageBases",
        f"Set maxiterations={SESSION_MAXITER} tolerance={SESSION_TOLERANCE}",
    ):
        ctx_a2.Text.Command = command
    t0 = time.perf_counter()
    solve_engine_snap(ctx_a2, "A2")
    report["timing_a2_solve_session_settings_s"] = round(
        time.perf_counter() - t0, 3
    )
    a2 = inventory(ctx_a2, "A2 引擎直载（对齐会话设置）")
    a2["_node_voltages"] = node_voltages(ctx_a2)

    # ---- B：PowerZoo DSSSession（临时副本 + SHA256 身份 + 原生控制求解） ----
    t0 = time.perf_counter()
    session = DSSSession(model, case.master)
    report["timing_b_session_init_s"] = round(time.perf_counter() - t0, 3)
    report["b_model_sha256"] = session.model_sha256
    report["b_input_file_count"] = len(session.input_files)
    t0 = time.perf_counter()
    session.solve(native_controls=True)
    report["timing_b_native_solve_s"] = round(time.perf_counter() - t0, 3)
    b = inventory(session.dss, "B PowerZoo DSSSession（原生控制）")
    b["_node_voltages"] = node_voltages(session.dss)

    # compare 会消费 _node_voltages，两次对照各自传入带电压表的副本。
    report["comparison_a2_b"] = compare(
        {**a2, "_node_voltages": a2["_node_voltages"]},
        {**b, "_node_voltages": b["_node_voltages"]},
    )
    report["comparison_a1_a2"] = compare(
        {**a1, "_node_voltages": a1["_node_voltages"]},
        {**a2, "_node_voltages": a2["_node_voltages"]},
    )
    report["a1"] = {k: v for k, v in a1.items() if k != "_node_voltages"}
    report["a2"] = {k: v for k, v in a2.items() if k != "_node_voltages"}
    report["b"] = {k: v for k, v in b.items() if k != "_node_voltages"}
    report["peak_rss_gb_after_a"] = round(_peak_rss_gb(), 3)

    if not args.skip_probes:
        # ---- 性能探针（口径见 method.note；一次性，不代表训练吞吐） ----
        solution = ctx_a2.ActiveCircuit.Solution
        probes = {}
        t0 = time.perf_counter()
        solution.LoadMult = 1.01
        solve_engine_snap(ctx_a2, "probe")
        probes["perturbed_solve_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        solution.LoadMult = 1.0
        solve_engine_snap(ctx_a2, "probe")
        probes["restore_solve_s"] = round(time.perf_counter() - t0, 3)
        circuit = ctx_a2.ActiveCircuit
        t0 = time.perf_counter()
        values = list(circuit.AllBusVmagPu)
        probes["extract_allbusvmagpu_s"] = round(time.perf_counter() - t0, 3)
        probes["extract_allbusvmagpu_nodes"] = len(values)
        t0 = time.perf_counter()
        node_voltages(ctx_a2)
        probes["extract_node_voltage_map_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        session_state = session.read_state()
        probes["session_read_state_s"] = round(time.perf_counter() - t0, 3)
        probes["session_read_state_buses"] = len(session_state.voltages_pu)
        probes["session_read_state_branches"] = len(session_state.branches)
        # 满帧负荷写入探针：等效 apply_operating_row 的逐负荷 Edit 风暴
        # （倍率 1.0，仅量测命令开销，不代表训练吞吐）。
        t0 = time.perf_counter()
        session.apply(
            {
                f"Load.{name}": {"kw": dev.kw, "kvar": dev.kvar}
                for name, dev in session.loads.items()
            }
        )
        probes["session_apply_full_load_frame_s"] = round(
            time.perf_counter() - t0, 3
        )
        report["probes"] = probes
        report["peak_rss_gb_final"] = round(_peak_rss_gb(), 3)

    session.close()
    output = (
        Path(args.output)
        if args.output
        else ROOT / "results/validation/external/uk_mvlv_hv_ug_baseline.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8",
    )

    summary = {
        "output": str(output),
        "match_a2_b": report["comparison_a2_b"]["match"],
        "max_node_voltage_diff_pu": report["comparison_a2_b"]["node_voltages"][
            "max_abs_diff_pu"
        ],
        "source_pq_rel_diff": report["comparison_a2_b"]["source_pq"]["max_rel_diff"],
        "a1_solver": report["a1"]["solver"],
        "a2_solver": report["a2"]["solver"],
        "b_solver": report["b"]["solver"],
        "voltage_summary_a1": report["a1"]["voltage_summary"],
        "timing_s": {
            k: v for k, v in report.items() if k.startswith("timing_")
        },
        "probes": report.get("probes"),
        "peak_rss_gb": report.get("peak_rss_gb_final"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not report["comparison_a2_b"]["match"]:
        raise SystemExit("A2 与 B 对照未通过公差；详见报告 JSON")


if __name__ == "__main__":
    main()
