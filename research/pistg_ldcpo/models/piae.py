"""PIAE：物理信息自编码器（论文式(18)-(22)）与预训练数据契约。

重建头从 h_STG 解码 (V_hat, p_hat, q_hat)（标准化空间），反标准化后
计算三项物理损失：
- L_KCL 式(19)：节点有功平衡残差（注入用当步真实 p_load/p_PV/p_CB 拆分
  的 p_ch、p_dis）；
- L_VDrop 式(20)：DistFlow 电压降一致性（p.u. 系统，S_base=10 MVA）；
- L_recon 式(21)：V 与支路 p/q 的 MSE（标准化空间，论文未提归一化，
  线性变换下等价于物理空间加权 MSE，偏离文档记录）。

数据契约（collect 产物，float32 memmap）：
- features: (E, T, 33, 5)  式(8b) 母线特征（V 列为伪量测=上步解）
- price:    (E, T)         AUD/MWh
- v_true:   (E, T, 33)     p.u.
- flow_true:(E, T, 32, 2)  kW/kvar（parent 端）
- net_inj:  (E, T, 33)     kW（p_PV+p_dis−p_load−p_ch，即"注入−消耗"）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

S_BASE_MVA = 10.0


class ReconstructionHead(nn.Module):
    """轻量解码器 ω_recon（论文表 II：hidden [128,128]）。"""

    def __init__(self, d_in: int, n_bus: int, n_branch: int,
                 hidden=(128, 128)):
        super().__init__()
        self.n_bus, self.n_branch = n_bus, n_branch
        dims = [d_in, *hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_bus + 2 * n_branch))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor):
        out = self.net(h)
        v = out[:, : self.n_bus]
        p = out[:, self.n_bus : self.n_bus + self.n_branch]
        q = out[:, self.n_bus + self.n_branch :]
        return v, p, q


class PIAE:
    """编码器 + 重建头 + 物理损失（式(19)-(22)）。"""

    def __init__(
        self,
        encoder: nn.Module,
        head: ReconstructionHead,
        lines: list[dict],
        num_buses: int,
        base_kv_ll: float,
        feature_stats: dict[str, np.ndarray],  # bus 特征 mean/std (5,)
        target_stats: dict[str, np.ndarray],  # v/p/q mean/std
        beta_kcl: float = 1.0,
        beta_vdrop: float = 1.0,
        beta_recon: float = 0.5,
    ):
        self.encoder, self.head = encoder, head
        self.beta_kcl, self.beta_vdrop, self.beta_recon = (
            beta_kcl, beta_vdrop, beta_recon,
        )
        self.n_bus, self.n_branch = num_buses, len(lines)
        # 拓扑与 p.u. 阻抗（式(20)）：z_pu = Z_ohm × S_base / V_LL²。
        z_base = base_kv_ll**2 / S_BASE_MVA
        self.register(lines, z_base)
        self.feature_mean = torch.tensor(feature_stats["mean"])
        self.feature_std = torch.tensor(feature_stats["std"])
        self.v_mean = torch.tensor(target_stats["v_mean"])
        self.v_std = torch.tensor(target_stats["v_std"])
        self.p_mean = torch.tensor(target_stats["p_mean"])
        self.p_std = torch.tensor(target_stats["p_std"])
        self.q_mean = torch.tensor(target_stats["q_mean"])
        self.q_std = torch.tensor(target_stats["q_std"])

    def register(self, lines, z_base):
        """预计算支路 p.u. 阻抗与 KCL 的父/子支路索引（式(19)）。"""
        r = np.array([ln["r_ohm"] / z_base for ln in lines])
        x = np.array([ln["x_ohm"] / z_base for ln in lines])
        self.r_pu = torch.tensor(r, dtype=torch.float32)
        self.x_pu = torch.tensor(x, dtype=torch.float32)
        self.branch_from = torch.tensor(
            [ln["from_bus"] - 1 for ln in lines], dtype=torch.long
        )
        self.branch_to = torch.tensor(
            [ln["to_bus"] - 1 for ln in lines], dtype=torch.long
        )
        in_idx = [[] for _ in range(self.n_bus)]
        out_idx = [[] for _ in range(self.n_bus)]
        for k, ln in enumerate(lines):
            out_idx[ln["from_bus"] - 1].append(k)
            in_idx[ln["to_bus"] - 1].append(k)
        # 汇聚为定长张量索引 + 掩码（torch 无变长列表索引）。
        max_deg = max(1, max(max(len(a) for a in in_idx),
                             max(len(a) for a in out_idx)))
        self.in_idx, self.in_mask = self._pack(in_idx, max_deg)
        self.out_idx, self.out_mask = self._pack(out_idx, max_deg)

    @staticmethod
    def _pack(index_lists, max_deg):
        idx = np.zeros((len(index_lists), max_deg), dtype=np.int64)
        mask = np.zeros((len(index_lists), max_deg), dtype=np.float32)
        for i, lst in enumerate(index_lists):
            for j, branch in enumerate(lst):
                idx[i, j] = branch
                mask[i, j] = 1.0
        return (torch.tensor(idx), torch.tensor(mask))

    # ------------------------------------------------------------------
    def to(self, device):
        """迁移物理层张量到目标设备（PIAE 非 nn.Module，CUDA 入口需手动迁移）。"""
        for name in ("in_idx", "out_idx", "in_mask", "out_mask",
                     "r_pu", "x_pu", "branch_from", "branch_to"):
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(device))
        return self

    def denorm(self, v, p, q):
        """标准化输出 → 物理单位（V p.u.、p/q kW；统计量本身就是 kW）。"""
        v_phys = v * self.v_std + self.v_mean
        p_phys = p * self.p_std + self.p_mean
        q_phys = q * self.q_std + self.q_mean
        return v_phys, p_phys, q_phys

    def loss(
        self,
        bus_window, price_window, norm_adj,  # 标准化输入
        v_target, p_target, q_target,  # 标准化目标
        net_inj_pu,  # (B,33) 当步净注入 p.u.（KCL 已知项）
    ):
        h = self.encoder(bus_window, price_window, norm_adj)
        v_hat, p_hat, q_hat = self.head(h)

        # 式(21)：重建 MSE（标准化空间）。
        l_recon = (
            (v_hat - v_target).pow(2).mean()
            + (p_hat - p_target).pow(2).mean()
            + (q_hat - q_target).pow(2).mean()
        )

        # 物理单位与 p.u.：kW → p.u.。
        v_phys, p_phys, q_phys = self.denorm(v_hat, p_hat, q_hat)
        p_pu = p_phys / (S_BASE_MVA * 1000.0)
        q_pu = q_phys / (S_BASE_MVA * 1000.0)

        # 式(19)：L_KCL。节点平衡：入流和 + 净注入 − 出流和 = 0
        #（净注入 = PV + CBESS 放电 − 负荷，注入为正）。
        # 修订（PR #65 复审 P1）：此前误写为「入流 − 出流 − 净注入」，
        # 符号反了会使真实潮流也受罚（末端负荷母线残差 ≈ 2×负荷）。
        # 松弛母线（bus1）的源注入不在式(19)已知项内，排除之（偏离文档记录）。
        inflow = (p_pu[:, self.in_idx] * self.in_mask).sum(-1)
        outflow = (p_pu[:, self.out_idx] * self.out_mask).sum(-1)
        residual = (inflow - outflow + net_inj_pu)[:, 1:]
        l_kcl = residual.pow(2).sum(-1).mean()

        # 式(20)：L_VDrop。ℓ̂ = (p̂²+q̂²)/V̂_i²。
        v_from = v_phys[:, self.branch_from]
        ell = (p_pu.pow(2) + q_pu.pow(2)) / v_from.pow(2).clamp(min=1e-6)
        drop = (
            v_phys[:, self.branch_to].pow(2)
            - v_from.pow(2)
            + 2.0 * (self.r_pu * p_pu + self.x_pu * q_pu)
            - (self.r_pu.pow(2) + self.x_pu.pow(2)) * ell
        )
        l_vdrop = drop.pow(2).mean()

        total = self.beta_kcl * l_kcl + self.beta_vdrop * l_vdrop + (
            self.beta_recon * l_recon
        )
        return total, {
            "kcl": float(l_kcl.detach()),
            "vdrop": float(l_vdrop.detach()),
            "recon": float(l_recon.detach()),
        }
