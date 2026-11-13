"""规模基准：分算例记录构造/编译/探测/潮流/读取/动作/观测/推理/日志耗时与内存。

对每个算例构造正式环境（与训练同一装配路径），跑固定动作 N 步，按会话
timings 埋点（init.copy/sha256/context、reset.compile/connections/discover/
topology、step.apply/probe/solve/branch_state/read_state）分解耗时；观测构造
与奖励/日志为步内其余部分；策略推理用固定 MLP 前向量级（2×64 隐层）作为
参考。峰值内存 ru_maxrss（进程单调）。结果写 JSON 并打印表。

    python scripts/benchmark_scale.py --cases 34Bus_PV 9500Node UKMVLV_HVUG SMARTDS_P4UDT4 --steps 12

口径：单进程顺序量测；固定动作不含学习；耗时供优化前后对照，不是训练吞吐。
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

CASE_PRESETS = {
    "34Bus_PV": {
        "family": "smartgrid",
        "system_ref": "34Bus_PV",
        "values": {"env": {"max_episode_steps": 200}},
    },
    "9500Node": {
        "family": "vvc",
        "system_ref": "9500Node",
        "values": {"env": {"observation_neighborhood": 2}},
    },
    "9500Node_light": {
        "family": "vvc",
        "system_ref": "9500Node",
        "values": {"env": {"observation_neighborhood": 2, "light_state_read": True}},
    },
    "UKMVLV_HVUG_light": {
        "family": "vvc",
        "system_ref": "UKMVLV_HVUG",
        "values": {"env": {"observation_neighborhood": 2, "light_state_read": True}},
    },
    "9500Node_full_obs": {
        "family": "vvc",
        "system_ref": "9500Node",
        "values": {"env": {}},
    },
    "UKMVLV_HVUG": {
        "family": "vvc",
        "system_ref": "UKMVLV_HVUG",
        "values": {"env": {"observation_neighborhood": 2}},
    },
    "SMARTDS_P4UDT4": {
        "family": "smartgrid",
        "system_ref": "SMARTDS_P4UDT4",
        "values": {
            "env": {
                "pv_control_mode": "rl",
                "pv_plan": "none",
                "data": {"step_seconds": 900},
                "observation_neighborhood": 2,
                "max_episode_steps": 96,
            }
        },
    },
    "SMARTDS_RDT1534": {
        "family": "smartgrid",
        "system_ref": "SMARTDS_RDT1534",
        "values": {
            "env": {
                "pv_control_mode": "rl",
                "pv_plan": "none",
                "data": {"step_seconds": 900},
                "observation_neighborhood": 2,
                "max_episode_steps": 96,
            }
        },
    },
}


def _peak_rss_gb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(raw / 1024**3 if sys.platform == "darwin" else raw / 1024 / 1024, 3)


def _reference_policy_latency(obs_dim, act_dim, agents, runs=20):
    """固定 MLP 前向的参考推理耗时（ms/步，全部智能体一次 batch）。"""
    import torch

    torch.set_num_threads(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, act_dim),
    ).eval()
    batch = torch.zeros(agents, obs_dim)
    with torch.no_grad():
        for _ in range(3):
            net(batch)
        started = time.perf_counter()
        for _ in range(runs):
            net(batch)
    return (time.perf_counter() - started) / runs * 1000


def _baseline_actions(env):
    import numpy as np

    actions = []
    for space in env.action_space:
        if space.__class__.__name__ == "Discrete":
            actions.append(int(space.n // 2))
        else:
            actions.append(np.zeros(space.shape, dtype=np.float32))
    return actions


def run_case(case, steps):
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env

    preset = CASE_PRESETS[case]
    started = time.perf_counter()
    cfg = ConfigLoader().load(
        "happo", preset["family"], system_ref=preset["system_ref"], values=preset["values"]
    )
    env = make_power_env(preset["family"], cfg.env_args, 0, 0)
    construct_s = time.perf_counter() - started

    core = env.env
    circuit = core.circuit
    construct_timings = dict(circuit.timings)
    nodes = circuit.native.NumNodes
    buses = circuit.native.NumBuses
    agents = len(core.action_space)
    obs, _, _ = env.reset()
    obs_dim = int(obs[0].shape[0])
    reset_timings = {
        k: v - construct_timings.get(k, 0.0) for k, v in circuit.timings.items()
    }

    actions = _baseline_actions(env)
    # 步进计时：每步记录 timings 增量与总耗时。
    keys = (
        "step.apply", "step.probe", "step.solve",
        "step.branch_state", "step.read_state",
    )
    step_totals, step_breakdown, failures = [], {k: [] for k in keys}, 0
    for index in range(steps):
        before = dict(circuit.timings)
        t0 = time.perf_counter()
        try:
            env.step(actions)
        except Exception as error:  # 固定动作不触发发散重罚的路径按失败计
            failures += 1
            str(error)
            break
        step_totals.append((time.perf_counter() - t0) * 1000)
        for key in keys:
            step_breakdown[key].append(circuit.timings.get(key, 0) - before.get(key, 0))
        if index == 0:
            # 首步含观测空间构建等一次性开销，单独标记不纳入均值。
            step_totals[-1] = None
            for key in keys:
                step_breakdown[key][-1] = None

    valid_totals = [t for t in step_totals if t is not None]
    session_ms = {
        key: round(
            sum(v for v in values if v is not None)
            / max(len([v for v in values if v is not None]), 1),
            3,
        )
        for key, values in step_breakdown.items()
    }
    mean_total = sum(valid_totals) / max(len(valid_totals), 1)
    other_ms = mean_total - sum(session_ms.values())
    act_dim = 2 if any(s.__class__.__name__ == "Box" for s in core.action_space) else 1
    policy_ms = _reference_policy_latency(obs_dim, act_dim, max(agents, 1))
    result = {
        "case": case,
        "nodes": nodes,
        "buses": buses,
        "agents": agents,
        "obs_dim": obs_dim,
        "steps_measured": len(valid_totals),
        "step_failures": failures,
        "construct_s": round(construct_s, 3),
        "construct_timings_ms": {k: round(v, 1) for k, v in construct_timings.items()},
        "reset_timings_ms": {k: round(v, 1) for k, v in reset_timings.items() if v > 0.05},
        "step_total_ms": round(mean_total, 3),
        "step_session_ms": session_ms,
        "step_observation_reward_logging_ms": round(max(other_ms, 0), 3),
        "reference_policy_forward_ms": round(policy_ms, 3),
        "peak_rss_gb": _peak_rss_gb(),
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", default=["34Bus_PV", "9500Node", "UKMVLV_HVUG", "SMARTDS_P4UDT4"]
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "method": {
            "timer": "time.per_counter 会话埋点 + 步外层计时；首步一次性开销剔除",
            "policy": "固定 2×64 MLP 前向参考（非实际策略网络）",
            "note": "固定动作顺序量测，非训练吞吐；内存为进程单调峰值",
        },
        "cases": {},
    }
    for case in args.cases:
        print(f"== {case} ==", flush=True)
        try:
            result = run_case(case, args.steps)
        except Exception as error:
            result = {"case": case, "error": f"{type(error).__name__}: {error}"}
        report["cases"][case] = result
        print(json.dumps(result, ensure_ascii=False))

    output = (
        Path(args.output)
        if args.output
        else ROOT / "results/validation/external/scale_benchmark.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"输出：{output}")


if __name__ == "__main__":
    main()
