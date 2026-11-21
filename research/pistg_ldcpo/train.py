"""PISTG-LDCPPO / STG-LDCPPO / LDCPPO 训练入口（论文 Stage 2）。

用法：
    python -m research.pistg_ldcpo.train --method pistg --episodes 20
    python -m research.pistg_ldcpo.train --method flat --episodes 20

--method：pistg（PIAM 编码器，需先 pretrain）/ stg（二值邻接编码器）/
flat（平铺窗口，论文 LDCPPO 基线）。约束预算 d_k = --budget-fraction
× 零控制基线折扣成本（论文未给值；v1 全量 0.30 预算过松致排序失去
区分度，v2 收紧至 0.10，依据见偏离文档 C 节）。--resume 支持断点续训。
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

from algorithms.centralized.ldcppo import COST_KEYS, LDCPPO
from .config import CASE_FOLDER, RunConfig
from envs.cbess.profile_store import ProfileStore
from envs.cbess.cbes_env import CBESEnv
from .models.pistg import PISTGEncoder

WINDOW = 12  # 仅为向后兼容保留的默认；实际取 RunConfig().encoder.window


def find_piae(output_dir: Path, method: str) -> Path | None:
    """找该方法最新的 PIAE checkpoint（按回合数数值序，防字典序 ep1000<ep500）。"""
    import re

    cands = []
    for p in output_dir.glob(f"piae_{method}_ep*.pt"):
        m = re.fullmatch(rf"piae_{method}_ep(\d+)\.pt", p.name)
        if m:
            cands.append((int(m.group(1)), p))
    return max(cands)[1] if cands else None


class StateEncoder:
    """W=12 滚动窗口 → 状态向量（PISTG/STG 嵌入或平铺）。"""

    def __init__(self, method: str, piae_path: Path | None, num_buses: int = 33,
                 window: int = WINDOW):
        self.method = method
        self.window_len = window
        self.window = deque(maxlen=window)
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
            self.dim = self.window_len * (num_buses * 5 + 1)
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
        while len(entries) < self.window_len:
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
    parser.add_argument(
        "--budget-fraction", type=float, default=None,
        help="覆盖 d_k = fraction × 零控制折扣成本（默认 0.30；收紧如 0.10）"
    )
    parser.add_argument(
        "--lambda-min", type=float, default=None,
        help="对偶乘子下限（默认 0 = 式(36d) 逐字；正值抑制末期约束回弹）"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从 out_dir/checkpoint.pt 续训（权重+优化器+乘子+RNG）"
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    torch.manual_seed(args.seed)  # 网络初始化/采样/randperm 可复现（复审 P2）
    cfg = RunConfig()
    cfg.algo.total_episodes = args.episodes
    if args.budget_fraction is not None:
        cfg.algo.cost_budget_fraction = {
            k: args.budget_fraction for k in COST_KEYS
        }
    if args.lambda_min is not None:
        cfg.algo.lambda_min = args.lambda_min
    store = ProfileStore(cfg.data_dir)
    env = CBESEnv(CASE_FOLDER, store, cfg.scenario)
    out_dir = cfg.output_dir / (args.tag or f"{args.method}_ep{args.episodes}")
    out_dir.mkdir(parents=True, exist_ok=True)

    piae_path = args.piae or (
        find_piae(cfg.output_dir, args.method)
        if args.method in ("pistg", "stg") else None
    )
    encoder = StateEncoder(args.method, piae_path,
                           window=cfg.encoder.window)
    agent = LDCPPO(encoder.dim, env.action_dim, cfg.algo,
                   critics="dedicated", device=args.device)

    start_episode = 1
    rng = np.random.default_rng(args.seed)
    if args.resume and (out_dir / "checkpoint.pt").exists():
        ckpt = torch.load(out_dir / "checkpoint.pt", map_location=args.device,
                          weights_only=False)
        agent.load_state_dict(ckpt)
        start_episode = int(ckpt.get("episode", 0)) + 1
        if "np_rng_state" in ckpt:
            rng.bit_generator.state = ckpt["np_rng_state"]
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        print(f"续训：自 ep{start_episode} 恢复（λ={agent.multipliers}）")
    if not agent.budgets:
        print("估计零控制预算 d_k …")
        budgets = estimate_budgets(env, cfg)
        agent.set_budgets(budgets)
        print("预算:", {k: round(v, 3) for k, v in budgets.items()})

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "a", newline="")
    logger = csv.writer(log_file)
    if start_episode == 1:
        logger.writerow(
            ["episode", "reward", "cost_aud", "VD", "VN", "LO",
             "curtailed_kwh", *[f"lambda_{k}" for k in COST_KEYS], "kl",
             "seconds"]
        )
    started = time.time()
    for episode in range(start_episode, args.episodes + 1):
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
                    **agent.state_dict(),
                    "method": args.method,
                    "episode": episode,
                    "encoder_stats": encoder.stats,
                    "piae": str(piae_path) if piae_path else None,
                    "seed": args.seed,
                    "np_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                },
                out_dir / "checkpoint.pt",
            )
    log_file.close()
    env.close()
    print(f"完成：日志 {log_path}，checkpoint {out_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
