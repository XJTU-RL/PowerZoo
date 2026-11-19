"""PISTG-LDCPPO / STG-LDCPPO / LDCPPO 训练入口（论文 Stage 2）。

用法：
    python -m research.pistg_ldcpo.train --method pistg --episodes 20
    python -m research.pistg_ldcpo.train --method flat --episodes 20

--method：pistg（PIAM 编码器，需先 pretrain）/ stg（二值邻接编码器）/
flat（平铺窗口，论文 LDCPPO 基线）。约束预算 d_k 默认取零控制基线
折扣成本的 30%（论文未给值，标定记录于偏离文档）。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .algo.ldcppo import COST_KEYS, LDCPPO
from .config import CASE_FOLDER, RunConfig
from .data.profile_store import ProfileStore
from .environment.cbes_env import CBESEnv
from .models.pistg import PISTGEncoder

WINDOW = 12


class StateEncoder:
    """W=12 滚动窗口 → 状态向量（PISTG/STG 嵌入或平铺）。"""

    def __init__(self, method: str, piae_path: Path | None, num_buses: int = 33):
        self.method = method
        self.window = deque(maxlen=WINDOW)
        self.encoder = None
        self.norm_adj = None
        self.stats = None
        if method in ("pistg", "stg"):
            if piae_path is None or not piae_path.exists():
                raise FileNotFoundError(
                    f"{method} 需要预训练编码器：{piae_path}（先运行 pretrain.py）"
                )
            ckpt = torch.load(piae_path, map_location="cpu", weights_only=False)
            self.stats = ckpt["stats"]
            c = ckpt["config"]
            self.encoder = PISTGEncoder(
                num_buses=num_buses,
                gcn_hidden=c["gcn_hidden"],
                d_spatial=c["d_spatial"],
                d_dn_tcn_out=c["d_dn_tcn_out"],
                d_price_tcn_out=c["d_price_tcn_out"],
            )
            self.encoder.load_state_dict(ckpt["encoder"])
            self.encoder.eval()  # Stage 2 冻结（论文两阶段训练）
            import json as _j

            from .models.pistg import physics_informed_adjacency

            meta = _j.loads(
                (CASE_FOLDER / "case_meta.json").read_text(encoding="utf-8")
            )
            if method == "pistg":
                adjacency = physics_informed_adjacency(
                    meta["lines"], num_buses, float(meta["base_kv_ll"])
                )
            else:
                adjacency = np.zeros((num_buses, num_buses), dtype=np.float32)
                for ln in meta["lines"]:
                    i, j = ln["from_bus"] - 1, ln["to_bus"] - 1
                    adjacency[i, j] = adjacency[j, i] = 1.0
                adjacency = torch.from_numpy(adjacency)
            degree = adjacency.sum(-1).clamp(min=1e-6)
            self.norm_adj = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
            self.dim = self.encoder.d_stg
        elif method == "flat":
            if piae_path is not None and piae_path.exists():
                ckpt = torch.load(piae_path, map_location="cpu", weights_only=False)
                self.stats = ckpt["stats"]
            self.dim = WINDOW * (num_buses * 5 + 1)
        else:
            raise ValueError(f"未知 method：{method}")

    def reset(self, obs: dict):
        self.window.clear()
        self.window.append(self._normalize(obs))

    def _normalize(self, obs: dict) -> tuple[np.ndarray, float]:
        features = obs["bus_features"].astype(np.float64)  # (33,5)
        price = float(obs["price"])
        if self.stats is not None:
            features = (features - self.stats["feature_mean"]) / self.stats[
                "feature_std"
            ]
            price = (price - self.stats["price_mean"]) / self.stats["price_std"]
        return features.astype(np.float32), np.float32(price)

    def append(self, obs: dict):
        self.window.append(self._normalize(obs))

    def encode(self) -> np.ndarray:
        # 窗口未满时重复首元素填充（论文未说明冷启动，偏离文档记录）。
        entries = list(self.window)
        while len(entries) < WINDOW:
            entries.insert(0, entries[0])
        features = np.stack([e[0] for e in entries])  # (W,33,5)
        prices = np.array([e[1] for e in entries])  # (W,)
        if self.method == "flat":
            return np.concatenate([features.ravel(), prices]).astype(np.float32)
        with torch.no_grad():
            h = self.encoder(
                torch.tensor(features).unsqueeze(0),
                torch.tensor(prices).unsqueeze(0),
                self.norm_adj,
            )
        return h.squeeze(0).numpy()


def estimate_budgets(env: CBESEnv, cfg: RunConfig, episodes: int = 3):
    """零控制基线的期望折扣成本 J_C_k（式28 口径）→ d_k = fraction × J_C_k。"""
    rng = np.random.default_rng(cfg.seed + 1)
    discount = cfg.algo.gamma ** np.arange(env.steps_per_day)
    episode_returns = {k: [] for k in COST_KEYS}
    for _ in range(episodes):
        env.reset(rng, "train")
        action = np.zeros(env.action_dim)
        action[8:34] = -1.0  # κ=0
        step_costs = {k: np.zeros(env.steps_per_day) for k in COST_KEYS}
        for t in range(env.steps_per_day):
            _, _, costs, _, _ = env.step(action)
            for k in COST_KEYS:
                step_costs[k][t] = costs[k]
        for k in COST_KEYS:
            episode_returns[k].append(float((step_costs[k] * discount).sum()))
    return {
        k: cfg.algo.cost_budget_fraction.get(k, 0.3)
        * float(np.mean(episode_returns[k]))
        for k in COST_KEYS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("pistg", "stg", "flat"),
                        default="pistg")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--piae", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    cfg = RunConfig()
    cfg.algo.total_episodes = args.episodes
    store = ProfileStore(cfg.data_dir)
    env = CBESEnv(CASE_FOLDER, store, cfg.scenario)
    out_dir = cfg.output_dir / (args.tag or f"{args.method}_ep{args.episodes}")
    out_dir.mkdir(parents=True, exist_ok=True)

    piae_path = args.piae or (
        cfg.output_dir / f"piae_{args.method}_ep50.pt"
        if args.method in ("pistg", "stg") else None
    )
    encoder = StateEncoder(args.method, piae_path)
    agent = LDCPPO(encoder.dim, env.action_dim, cfg.algo,
                   critics="dedicated", device=args.device)

    print("估计零控制预算 d_k …")
    budgets = estimate_budgets(env, cfg)
    agent.set_budgets(budgets)
    print("预算:", {k: round(v, 3) for k, v in budgets.items()})

    rng = np.random.default_rng(args.seed)
    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(
        ["episode", "reward", "cost_aud", "VD", "VN", "LO",
         "curtailed_kwh", *[f"lambda_{k}" for k in COST_KEYS], "kl", "seconds"]
    )
    started = time.time()
    for episode in range(1, args.episodes + 1):
        obs = env.reset(rng, "train")
        encoder.reset(obs)
        T = env.steps_per_day
        states = np.zeros((T, encoder.dim), np.float32)
        actions = np.zeros((T, env.action_dim), np.float32)
        logps = np.zeros(T)
        rewards = np.zeros(T)
        costs = {k: np.zeros(T) for k in COST_KEYS}
        for t in range(T):
            states[t] = encoder.encode()
            a, logp = agent.act(states[t])
            actions[t] = a
            logps[t] = logp
            obs, r, c, done, info = env.step(a)
            rewards[t] = r
            for k in COST_KEYS:
                costs[k][t] = c[k]
            encoder.append(obs)
        terminal_state = encoder.encode()
        last_values = (
            agent.values(terminal_state) if cfg.algo.bootstrap_at_episode_end
            else {"R": 0.0, **{k: 0.0 for k in agent.cost_critics}}
        )
        stats = agent.update(states, actions, logps, rewards, costs, last_values)
        m = env.day_metrics
        logger.writerow(
            [episode, round(m["reward"], 2), round(-m["reward"], 2),
             round(m["VD"], 3), round(m["VN"], 1), round(m["LO"], 2),
             round(m["curtailed_kwh"], 1),
             *[round(agent.multipliers[k], 4) for k in COST_KEYS],
             round(stats["kl"], 4), round(time.time() - started, 1)]
        )
        if episode % 5 == 0 or episode == 1:
            print(
                f"ep{episode}: cost={-m['reward']:.0f} VD={m['VD']:.2f} "
                f"VN={m['VN']:.0f} LO={m['LO']:.1f} "
                f"λ={{{', '.join(f'{k}:{agent.multipliers[k]:.3f}' for k in COST_KEYS)}}} "
                f"kl={stats['kl']:.4f}"
            )
        if episode % 100 == 0 or episode == args.episodes:
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "reward_critic": agent.reward_critic.state_dict(),
                    "cost_critics": {k: c.state_dict()
                                     for k, c in agent.cost_critics.items()},
                    "multipliers": agent.multipliers,
                    "budgets": agent.budgets,
                    "method": args.method,
                    "episode": episode,
                    "encoder_stats": encoder.stats,
                },
                out_dir / "checkpoint.pt",
            )
    log_file.close()
    env.close()
    print(f"完成：日志 {log_path}，checkpoint {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
