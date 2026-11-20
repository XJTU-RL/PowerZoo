"""测试期评估（论文表 III/VI/V 口径）。

对每方法加载 checkpoint，在 test split 上确定性策略 rollout N 回合：
- Operating cost（AUD/day）与四指标（表 III：p.u./(5min·bus)、
  times/(5min·bus)、%/(5min·line)，即日累计 ÷ 288 步 ÷ 母线/线路数）；
- 弃光统计（表 VI：E_curt kWh/日、ρ_curt %、η_HP %，T_HP 取 9:00-15:00，
  论文未定义精确时段，偏离文档记录）；
- 推理时延与参数量（表 V）。

用法：python -m research.pistg_ldcpo.evaluate --run-dir results/pistg_ldcpo/<tag>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .algo.ldcppo import COST_KEYS, LDCPPO
from .config import CASE_FOLDER, RunConfig
from .data.profile_store import ProfileStore
from .environment.cbes_env import CBESEnv
from .models.pistg import PISTGEncoder
from .train import StateEncoder, find_piae


def load_agent(run_dir: Path, cfg: RunConfig, env: CBESEnv) -> tuple[LDCPPO, StateEncoder]:
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                      weights_only=False)
    method = ckpt["method"]
    # 优先用训练 checkpoint 里记录的 PIAE 路径，防 ep1000/ep500 字典序选错
    piae_path = Path(ckpt["piae"]) if ckpt.get("piae") else None
    if piae_path is None or not piae_path.exists():
        piae_path = find_piae(cfg.output_dir, method)
    encoder = StateEncoder(method, piae_path)
    agent = LDCPPO(encoder.dim, env.action_dim, cfg.algo)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.reward_critic.load_state_dict(ckpt["reward_critic"])
    for k, critic in agent.cost_critics.items():
        critic.load_state_dict(ckpt["cost_critics"][k])
    agent.actor.eval()
    return agent, encoder


def evaluate(run_dir: Path, episodes: int = 100) -> dict:
    cfg = RunConfig()
    store = ProfileStore(cfg.data_dir)
    env = CBESEnv(CASE_FOLDER, store, cfg.scenario)
    agent, encoder = load_agent(run_dir, cfg, env)

    rng = np.random.default_rng(cfg.seed + 99)
    metrics = {k: [] for k in
               ("cost_aud", "VD", "VN", "LO", "curt_kwh", "curt_ratio",
                "curt_hp_ratio")}
    infer_ms = []
    for e in range(episodes):
        obs = env.reset(rng, "test")
        encoder.reset(obs)
        curt = curt_hp = pv_energy = 0.0
        dt_h = 5.0 / 60.0
        for t in range(env.steps_per_day):
            state = encoder.encode()
            started = time.perf_counter()
            action, _ = agent.act(state, deterministic=True)
            infer_ms.append((time.perf_counter() - started) * 1000)
            obs, r, c, done, info = env.step(action)
            curt += info["curtailed_kwh_step"]
            pv_energy += float(info["pv_available_kw"].sum()) * dt_h
            if 108 <= t < 180:  # 9:00-15:00
                curt_hp += info["curtailed_kwh_step"]
            encoder.append(obs)
        m = env.day_metrics
        metrics["cost_aud"].append(-m["reward"])
        metrics["VD"].append(m["VD"])
        metrics["VN"].append(m["VN"])
        metrics["LO"].append(m["LO"])
        metrics["curt_kwh"].append(curt)
        metrics["curt_ratio"].append(curt / max(pv_energy, 1e-9) * 100)
        metrics["curt_hp_ratio"].append(curt_hp / max(curt, 1e-9) * 100
                                        if curt > 1e-9 else 0.0)
    n_bus, n_line = env.n_bus, len(env.branch_names)
    steps = env.steps_per_day
    summary = {}
    for key, values in metrics.items():
        values = np.array(values)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    summary["paper_units"] = {
        "VD_per_bus_step": summary["VD"]["mean"] / steps / n_bus,
        "VN_per_bus_step": summary["VN"]["mean"] / steps / n_bus,
        "LO_per_line_step": summary["LO"]["mean"] / steps / n_line,
    }
    n_params = (
        sum(p.numel() for p in agent.actor.parameters())
        + sum(p.numel() for p in agent.reward_critic.parameters())
        + sum(p.numel() for c in agent.cost_critics.values()
              for p in c.parameters())
        + (sum(p.numel() for p in encoder.encoder.parameters())
           if encoder.encoder is not None else 0)
    )
    summary["inference"] = {
        "mean_ms": float(np.mean(infer_ms)),
        "p95_ms": float(np.percentile(infer_ms, 95)),
        "params_million": n_params / 1e6,
    }
    env.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    summary = evaluate(args.run_dir, args.episodes)
    out = args.run_dir / "evaluation.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary["paper_units"], indent=2))
    print("成本 %.1f±%.1f AUD/日 | VD %.2f | VN %.0f | LO %.1f | 弃光 %.1f kWh"
          % (summary["cost_aud"]["mean"], summary["cost_aud"]["std"],
             summary["VD"]["mean"], summary["VN"]["mean"], summary["LO"]["mean"],
             summary["curt_kwh"]["mean"]))
    print("推理 %.3f ms（论文 PISTG-LDCPPO 1.22 ms @33bus），参数 %.3f M"
          % (summary["inference"]["mean_ms"],
             summary["inference"]["params_million"]))


if __name__ == "__main__":
    main()
