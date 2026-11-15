"""SMART-DS 小馈线 24 小时基线：原生控制 / fixed pf / RL Q-only / RL P-Q 四模式对照。

对同一馈线（默认 smart_ds_sfo_p4udt4 的 solar_high_none 入口）按原始 15 分钟
时序跑 96 步（2017-01-01 当地标准时）：
- native：solve(native_controls=True)，原生 InvControl Volt-Var 参与求解
  （独立基线；该模式下不写任何 PV 设定，唯一写入方是原生控制器）。
- fixed：控制关闭，PV 保持模型 pf 设定（不探测不写）。
- rl_q_only：接管全部 PV，有功=当步可用功率，无功命令 +0.5（示例固定策略）。
- rl_pq：接管全部 PV，有功分数 0.75、无功命令 -0.3（示例固定策略）。

RL 两模式的动作构造与训练环境共用 envs.powergrid.devices.pv_command_request，
最终限幅统一由会话 apply 的设备层执行。每步记录电压越限幅度/节点数、网损、
支路过载、逐 PV 可用/实际 P/Q、限发与求解失败；输出 JSON 与摘要。示例策略
不代表算法收敛或泛化，仅验证控制权与物理链路。
"""

from pathlib import Path
import argparse
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INTERVAL = 900
STEPS_PER_DAY = 96


def branch_summary(state):
    """支路过载统计：无额定值（None）不得经 `or 0` 被当作 0 载荷计入。

    已采集但无可评级支路时 max 为 None——不得据此宣称网络无过载；
    未评级数单独记录。state.branches 为 None（未采集）时报错而非静默
    记 0（本脚本固定完整读取，None 属契约破坏）。
    """
    if state.branches is None:
        raise ValueError("支路统计收到未采集状态（branches=None），本脚本要求完整读取")
    rated = [
        info["loading_pu"]
        for info in state.branches.values()
        if info.get("loading_pu") is not None
    ]
    return {
        "overloaded_branches": sum(x > 1.0 for x in rated),
        "max_branch_loading_pu": max(rated) if rated else None,
        "unrated_branches": len(state.branches) - len(rated),
    }


def _run_day(session, series, mode, *, offset=0, fraction=0.75, qcommand=-0.3):
    """从全年第 offset 步起跑 96 步（一天）；返回逐步记录与汇总。"""
    from envs.powergrid.devices import pv_command_request

    session.reset(seconds=offset * INTERVAL)  # 时钟锚定到当日 00:00
    steps, failures = [], 0
    totals = {"loss_kwh": 0.0, "pv_available_kwh": 0.0, "pv_actual_kwh": 0.0, "curtailed_kwh": 0.0}
    for step in range(STEPS_PER_DAY):
        index = offset + step
        started = time.perf_counter()
        record = {"index": index, "mode": mode}
        try:
            session.begin_interval(index * INTERVAL)
            series.apply(session, index)
            apply_s = time.perf_counter() - started

            probe_s = 0.0
            available = {}
            if mode in ("rl_q_only", "rl_pq"):
                t0 = time.perf_counter()
                available = session.available_pv_power(list(series.pv_shapes))
                probe_s = time.perf_counter() - t0
                requests = {}
                for name, pv in session.pvs.items():
                    if mode == "rl_q_only":
                        request = pv_command_request(pv, 1.0, 0.5, available[name])
                    else:
                        request = pv_command_request(pv, fraction, qcommand, available[name])
                    requests["PVSystem." + name] = request
                session.apply(requests)

            t0 = time.perf_counter()
            session.solve(native_controls=(mode == "native"))
            solve_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            state = session.read_state()
            read_s = time.perf_counter() - t0
            session.finish_interval()
        except RuntimeError as error:
            if mode != "native" or "未收敛" not in str(error):
                raise
            failures += 1
            record["failure"] = str(error)
            steps.append(record)
            session.reset(seconds=(index + 1) * INTERVAL)
            continue

        voltages = [v for phases in state.voltages_pu.values() for v in phases.values()]
        branches = branch_summary(state)
        pv_rows = {
            name.split(".", 1)[1]: {
                "available_kw": available.get(name.split(".", 1)[1]),
                # 注入口径"发出为正"：injections 已是 -powers，直接取值。
                "p_kw": injection[0],
                "q_kvar": injection[1],
            }
            for name, injection in state.injections.items()
            if name.lower().startswith("pvsystem.")
        }
        for row in pv_rows.values():
            if row["available_kw"] is not None:
                totals["pv_available_kwh"] += row["available_kw"] * INTERVAL / 3600
                totals["curtailed_kwh"] += max(
                    0.0, row["available_kw"] - row["p_kw"]
                ) * INTERVAL / 3600
            totals["pv_actual_kwh"] += max(0.0, row["p_kw"]) * INTERVAL / 3600
        totals["loss_kwh"] += state.losses[0] * INTERVAL / 3600
        record.update(
            {
                "v_min_pu": min(voltages),
                "v_max_pu": max(voltages),
                "violating_nodes": sum(
                    1 for v in voltages if v < 0.95 or v > 1.05
                ),
                "max_violation_pu": max(
                    0.0,
                    max(0.95 - v for v in voltages),
                    max((v - 1.05 for v in voltages), default=0.0),
                ),
                "loss_kw": state.losses[0],
                "overloaded_branches": branches["overloaded_branches"],
                "unrated_branches": branches["unrated_branches"],
                "max_branch_loading_pu": branches["max_branch_loading_pu"],
                "grid_import_kw": state.grid_import[0],
                "pv": pv_rows,
                "seconds": {
                    "apply": round(apply_s, 3),
                    "probe": round(probe_s, 3),
                    "solve": round(solve_s, 3),
                    "read": round(read_s, 3),
                },
            }
        )
        steps.append(record)
    summary = {
        "mode": mode,
        "steps": len(steps),
        "failures": failures,
        **{key: round(value, 3) for key, value in totals.items()},
        "violation_steps": sum(1 for s in steps if s.get("violating_nodes")),
        "max_violation_pu": round(
            max((s.get("max_violation_pu", 0) for s in steps), default=0.0), 5
        ),
        "v_min_pu": round(min(s["v_min_pu"] for s in steps if "v_min_pu" in s), 5)
        if any("v_min_pu" in s for s in steps)
        else None,
        "v_max_pu": round(max(s["v_max_pu"] for s in steps if "v_max_pu" in s), 5)
        if any("v_max_pu" in s for s in steps)
        else None,
        "max_branch_loading_pu": (
            round(
                max(
                    s["max_branch_loading_pu"]
                    for s in steps
                    if s.get("max_branch_loading_pu") is not None
                ),
                4,
            )
            if any(s.get("max_branch_loading_pu") is not None for s in steps)
            else None
        ),
        "step_seconds_avg": round(
            sum(sum(s.get("seconds", {}).values()) for s in steps) / max(len(steps), 1), 3
        ),
    }
    return steps, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="smart_ds_sfo_p4udt4")
    parser.add_argument("--entry", default="solar_high_none")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--day", type=int, default=0, help="全年第几天（0=1月1日）")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from envs.powergrid.session import DSSSession
    from envs.powergrid.smartds_timeseries import SmartDSTimeseries
    from utils.external_cases import locate_entry

    folder, master = locate_entry(args.case, args.entry, args.data_dir)
    layout = folder.parents[4]
    rel_master = str(folder.relative_to(layout)) + "/" + master
    offset = args.day * STEPS_PER_DAY

    report = {
        "case_id": args.case,
        "entry": args.entry,
        "day_index": args.day,
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "time_base": "当地标准时（无夏令时），15 分钟原始采样不插值",
        "modes": {},
    }
    for mode in ("native", "fixed", "rl_q_only", "rl_pq"):
        t0 = time.perf_counter()
        session = DSSSession(layout, rel_master, step_seconds=INTERVAL)
        series = SmartDSTimeseries(layout, rel_master).bind(session)
        construct_s = time.perf_counter() - t0
        steps, summary = _run_day(session, series, mode, offset=offset)
        summary["construct_s"] = round(construct_s, 2)
        report["modes"][mode] = {"summary": summary, "steps": steps}
        print(json.dumps(summary, ensure_ascii=False))
        session.close()

    output = (
        Path(args.output)
        if args.output
        else ROOT
        / "results/validation/external"
        / f"{args.case}_day{args.day}_baseline.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"输出：{output}")


if __name__ == "__main__":
    main()
