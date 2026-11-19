"""PISTG/STG 编码器 PIAE 预训练（论文 Stage 1，式(18)-(22)）。

流程：随机策略（α~U(-1,1)）采集回合 → 组装 W=12 窗口 → 最小化
β1·L_KCL + β2·L_VDrop + β3·L_recon（Adam，lr 1e-3，论文表 II）。

用法：
    python -m research.pistg_ldcpo.pretrain --episodes 50 --variant pistg
    python -m research.pistg_ldcpo.pretrain --episodes 2000 --variant pistg

variant：pistg（PIAM + 全部物理损失）| stg（二值邻接 + 仅 recon 损失，
论文 STG-LDCPPO 基线的编码器）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import CASE_FOLDER, EncoderConfig, RunConfig
from .data.profile_store import ProfileStore
from .environment.cbes_env import CBESEnv
from .models.piae import PIAE, ReconstructionHead, S_BASE_MVA
from .models.pistg import PISTGEncoder, physics_informed_adjacency

WINDOW = 12
DT_KW = S_BASE_MVA * 1000.0


def collect_episodes(env: CBESEnv, episodes: int, rng: np.random.Generator,
                     split: str = "train", out_dir: Path | None = None):
    """随机策略采集：返回预训练所需数组（E 为回合数）。"""
    n_bus, n_branch = env.n_bus, len(env.branch_names)
    feats = np.zeros((episodes, env.steps_per_day, n_bus, 5), np.float32)
    prices = np.zeros((episodes, env.steps_per_day), np.float32)
    v_true = np.zeros((episodes, env.steps_per_day, n_bus), np.float32)
    flows = np.zeros((episodes, env.steps_per_day, n_branch, 2), np.float32)
    net_inj = np.zeros((episodes, env.steps_per_day, n_bus), np.float32)
    der_pos = np.array(env.der_buses) - 1
    cbess_pos = np.array(env.cbess_buses) - 1
    for e in range(episodes):
        env.reset(rng, split)
        for t in range(env.steps_per_day):
            action = rng.uniform(-1, 1, env.action_dim)
            obs, _, _, _, info = env.step(action)
            feats[e, t] = obs["bus_features"]
            prices[e, t] = obs["price"]
            v_true[e, t] = info["v_pu"]
            flows[e, t, :, 0] = info["branch_p_kw"]
            flows[e, t, :, 1] = info["branch_q_kvar"]
            inj = np.zeros(n_bus)
            inj[der_pos] += info["p_pv_kw"]
            inj[cbess_pos] += info["p_cbess_kw"]  # 放电为正
            inj -= env.profile.p_load_kw[t]
            net_inj[e, t] = inj
        if (e + 1) % 10 == 0:
            print(f"  采集 {e + 1}/{episodes} 回合")
    return feats, prices, v_true, flows, net_inj


def build_datasets(feats, prices, v_true, flows, net_inj):
    """窗口化（end = W-1..T-1，论文 277 窗/回合口径）与标准化统计。"""
    n_ep, n_steps, n_bus, _ = feats.shape
    n_branch = flows.shape[2]
    feature_mean = feats.reshape(-1, 5).mean(axis=0)
    feature_std = feats.reshape(-1, 5).std(axis=0) + 1e-6
    v_mean, v_std = v_true.mean(), v_true.std() + 1e-6
    p_mean, p_std = flows[..., 0].mean(), flows[..., 0].std() + 1e-6
    q_mean, q_std = flows[..., 1].mean(), flows[..., 1].std() + 1e-6
    price_mean, price_std = prices.mean(), prices.std() + 1e-6
    stats = {
        "feature_mean": feature_mean, "feature_std": feature_std,
        "v_mean": float(v_mean), "v_std": float(v_std),
        "p_mean": float(p_mean), "p_std": float(p_std),
        "q_mean": float(q_mean), "q_std": float(q_std),
        "price_mean": float(price_mean), "price_std": float(price_std),
    }
    # 标准化缓存（float32 控制内存：E×T×(33×5+…)×4B）。
    feats_n = (feats - feature_mean) / feature_std
    prices_n = (prices - price_mean) / price_std
    v_n = ((v_true - v_mean) / v_std).astype(np.float32)
    p_n = ((flows[..., 0] - p_mean) / p_std).astype(np.float32)
    q_n = ((flows[..., 1] - q_mean) / q_std).astype(np.float32)
    inj_pu = (net_inj / DT_KW).astype(np.float32)
    return (feats_n, prices_n, v_n, p_n, q_n, inj_pu), stats


def make_piae(meta: dict, variant: str, cfg: EncoderConfig, stats, lines):
    n_bus = int(meta["num_buses"])
    n_branch = len(lines)
    if variant == "pistg":
        adjacency = physics_informed_adjacency(
            lines, n_bus, float(meta["base_kv_ll"]),
            epsilon=cfg.piam_epsilon, nu=cfg.piam_nu,
        )
    elif variant == "stg":
        adjacency = np.zeros((n_bus, n_bus), dtype=np.float32)
        for ln in lines:
            i, j = ln["from_bus"] - 1, ln["to_bus"] - 1
            adjacency[i, j] = adjacency[j, i] = 1.0
        adjacency = torch.from_numpy(adjacency)
    else:
        raise ValueError(f"未知 variant：{variant}")
    degree = adjacency.sum(-1).clamp(min=1e-6)
    norm_adj = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
    encoder = PISTGEncoder(
        num_buses=n_bus,
        gcn_hidden=cfg.gcn_hidden,
        d_spatial=cfg.d_spatial,
        gcn_layers=cfg.gcn_layers,
        tcn_layers=cfg.tcn_layers,
        tcn_kernel=cfg.tcn_kernel,
        tcn_dilations=cfg.tcn_dilations,
        d_dn_tcn_out=cfg.d_dn_tcn_out,
        d_price_tcn_out=cfg.d_price_tcn_out,
    )
    head = ReconstructionHead(encoder.d_stg, n_bus, n_branch, cfg.decoder_hidden)
    betas = {
        "pistg": (cfg.beta_kcl, cfg.beta_vdrop, cfg.beta_recon),
        # STG 基线：去除物理正则，仅保留重建损失（论文表 VIII 消融口径）。
        "stg": (0.0, 0.0, cfg.beta_recon),
    }[variant]
    piae = PIAE(
        encoder, head, lines, n_bus, float(meta["base_kv_ll"]),
        feature_stats={"mean": stats["feature_mean"], "std": stats["feature_std"]},
        target_stats={k: np.float64(stats[k]) for k in
                      ("v_mean", "v_std", "p_mean", "p_std", "q_mean", "q_std")},
        beta_kcl=betas[0], beta_vdrop=betas[1], beta_recon=betas[2],
    )
    return piae, norm_adj


def train(piae: PIAE, norm_adj, arrays, cfg: EncoderConfig, episodes: int,
          device, out_path: Path, variant: str, stats, meta, lines,
          epochs_per_pass: int = 3):
    feats, prices, v_n, p_n, q_n, inj_pu = arrays
    n_ep, n_steps = feats.shape[0], feats.shape[1]
    ends = np.arange(WINDOW - 1, n_steps)  # 窗口末步 11..287（277 个）
    for module in (piae.encoder, piae.head):
        module.to(device)
    adj = norm_adj.to(device)
    optimizer = torch.optim.Adam(
        list(piae.encoder.parameters()) + list(piae.head.parameters()),
        lr=cfg.piae_lr,
    )
    rng = np.random.default_rng(0)
    total_windows = n_ep * len(ends)
    batch = min(cfg.pretrain_batch_windows, total_windows)
    steps_per_epoch = max(1, total_windows // batch)
    losses = []
    for epoch in range(epochs_per_pass):
        for _ in range(steps_per_epoch):
            e_idx = rng.integers(0, n_ep, batch)
            w_start = rng.integers(0, len(ends), batch)  # 窗口起始 0..T-W
            win = np.stack([
                feats[e, s : s + WINDOW] for e, s in zip(e_idx, w_start)
            ])
            bus_window = torch.tensor(win, device=device)
            price_window = torch.tensor(
                np.stack([prices[e, s : s + WINDOW] for e, s in zip(e_idx, w_start)]),
                device=device,
            )
            target = w_start + WINDOW - 1  # 窗口末步 = 重建目标
            v_t = torch.tensor(v_n[e_idx, target], device=device)
            p_t = torch.tensor(p_n[e_idx, target], device=device)
            q_t = torch.tensor(q_n[e_idx, target], device=device)
            inj = torch.tensor(inj_pu[e_idx, target], device=device)
            loss, parts = piae.loss(bus_window, price_window, adj, v_t, p_t, q_t, inj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), parts))
        mean = np.mean([x[0] for x in losses[-100:]])
        parts = {
            k: round(float(np.mean([p[k] for _, p in losses[-100:]])), 6)
            for k in ("kcl", "vdrop", "recon")
        }
        print(f"epoch {epoch + 1}: loss={mean:.5f} {parts}")
    torch.save(
        {
            "variant": variant,
            "encoder": piae.encoder.state_dict(),
            "head": piae.head.state_dict(),
            "stats": stats,
            "config": {
                "window": WINDOW,
                "gcn_hidden": cfg.gcn_hidden,
                "d_spatial": cfg.d_spatial,
                "d_dn_tcn_out": cfg.d_dn_tcn_out,
                "d_price_tcn_out": cfg.d_price_tcn_out,
            },
            "episodes": episodes,
        },
        out_path,
    )
    print(f"已保存 {out_path}")
    # 表 VII 式重建质量评估（最后两回合作为 held-out）。
    with torch.no_grad():
        e_hold = np.arange(max(0, n_ep - 2), n_ep)
        parts = {"volt_mae": [], "flow_nrmse": [], "kcl": [], "vdrop": []}
        for e in e_hold:
            for t in np.arange(WINDOW - 1, n_steps, 8):
                bw = torch.tensor(feats[e, t - WINDOW + 1 : t + 1][None])
                pw = torch.tensor(prices[e, t - WINDOW + 1 : t + 1][None])
                h = piae.encoder(bw, pw, adj)
                v_hat, p_hat, q_hat = piae.head(h)
                v_p, p_p, q_p = piae.denorm(v_hat, p_hat, q_hat)
                v_true = torch.tensor(v_n[e, t][None]) * float(stats["v_std"]) + float(stats["v_mean"])
                p_true = torch.tensor(p_n[e, t][None]) * float(stats["p_std"]) + float(stats["p_mean"])
                q_true = torch.tensor(q_n[e, t][None]) * float(stats["q_std"]) + float(stats["q_mean"])
                parts["volt_mae"].append(float((v_p - v_true).abs().mean()) * 1e3)
                rmse = float((p_p - p_true).pow(2).mean() + (q_p - q_true).pow(2).mean()) ** 0.5
                scale = float(p_true.abs().mean() + q_true.abs().mean())
                parts["flow_nrmse"].append(rmse / max(scale, 1e-6) * 100)
                p_pu = p_p / DT_KW
                q_pu = q_p / DT_KW
                inflow = (p_pu[:, piae.in_idx] * piae.in_mask).sum(-1)
                outflow = (p_pu[:, piae.out_idx] * piae.out_mask).sum(-1)
                inj = torch.tensor(inj_pu[e, t][None])
                parts["kcl"].append(float((inflow - outflow - inj)[:, 1:].abs().mean()))
                v_from = v_p[:, piae.branch_from]
                ell = (p_pu.pow(2) + q_pu.pow(2)) / v_from.pow(2).clamp(min=1e-6)
                drop = (v_p[:, piae.branch_to].pow(2) - v_from.pow(2)
                        + 2 * (piae.r_pu * p_pu + piae.x_pu * q_pu)
                        - (piae.r_pu.pow(2) + piae.x_pu.pow(2)) * ell)
                parts["vdrop"].append(float(drop.abs().mean()))
        summary = {k: float(np.mean(v)) for k, v in parts.items()}
        print("表VII式评估（held-out 回合）：VoltMAE×10⁻³p.u.=%.2f "
              "FlowNRMSE%%=%.2f KCL残差=%.2e VDrop残差=%.2e"
              % (summary["volt_mae"], summary["flow_nrmse"],
                 summary["kcl"], summary["vdrop"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="采集回合数（论文 2000）")
    parser.add_argument("--variant", choices=("pistg", "stg"), default="pistg")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run = RunConfig()
    meta = json.loads((CASE_FOLDER / "case_meta.json").read_text(encoding="utf-8"))
    store = ProfileStore(run.data_dir)
    env = CBESEnv(CASE_FOLDER, store, run.scenario)
    out_dir = run.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[1/3] 随机策略采集 {args.episodes} 回合…")
    started = time.time()
    arrays_raw = collect_episodes(env, args.episodes, rng)
    print(f"    采集用时 {time.time() - started:.0f}s")
    print("[2/3] 窗口化与标准化…")
    arrays, stats = build_datasets(*arrays_raw)
    print("[3/3] PIAE 训练…")
    piae, norm_adj = make_piae(meta, args.variant, run.encoder, stats, env.lines)
    out_path = out_dir / f"piae_{args.variant}_ep{args.episodes}.pt"
    train(piae, norm_adj, arrays, run.encoder, args.episodes, args.device,
          out_path, args.variant, stats, meta, env.lines,
          epochs_per_pass=args.epochs)
    env.close()


if __name__ == "__main__":
    main()
