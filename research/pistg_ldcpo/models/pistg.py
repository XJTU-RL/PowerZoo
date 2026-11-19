"""PISTG 编码器（论文式(10)-(17)）：PIAM + GCN + 注意力池化 + 双 TCN。

式(10) 的物理信息邻接矩阵（PIAM）：
    Ã_ij = σ(ψ_ij)，ψ_ij = −log(ε + √(r²+x²)) − ν/S_ij
单位取 r/x 为 Ω、S 为 MVA（两项同为 O(1) 量级，论文未说明单位，偏离
文档记录）。式(10) 的 σ 为 sigmoid；式(11) 的 σ 为激活函数（取 ReLU，
GCN 惯例，论文未指明，偏离文档记录）。

归一化：母线特征/电价/重建目标均按预训练统计 z-score 标准化（论文未
提及归一化，深度学习常规做法；物理损失在反标准化后的物理单位上计算）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def physics_informed_adjacency(
    lines: list[dict],
    num_buses: int,
    base_kv_ll: float,
    epsilon: float = 1e-6,
    nu: float = 1.0,
) -> torch.Tensor:
    """式(10)：由线路阻抗（Ω）与容量（MVA）构造 |N|×|N| PI 邻接矩阵。

    lines 为 case_meta.json 的线路表（from_bus/to_bus 经典编号、r_ohm、
    x_ohm、normamps）。无自环；辐射网双向传播取对称权重（式(4) A 对称）。
    """
    adjacency = np.zeros((num_buses, num_buses), dtype=np.float64)
    for line in lines:
        impedance = float(np.hypot(line["r_ohm"], line["x_ohm"]))
        s_mva = line["normamps"] * base_kv_ll * np.sqrt(3.0) / 1000.0
        psi = -np.log(epsilon + impedance) - nu / s_mva
        weight = 1.0 / (1.0 + np.exp(-psi))  # sigmoid
        i, j = line["from_bus"] - 1, line["to_bus"] - 1
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return torch.from_numpy(adjacency.astype(np.float32))


class GCNLayer(nn.Module):
    """式(11)：H' = ReLU(D̃^{-1/2} Ã D̃^{-1/2} H ω)。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        return F.relu(norm_adj @ h @ self.weight)


class AttentionPooling(nn.Module):
    """式(12)：α = softmax(ω_att·h_i)，h^S = Σ α_i h_i。"""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Linear(dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, d) → (B, d)
        alpha = torch.softmax(self.attention(h).squeeze(-1), dim=-1)
        return (h * alpha.unsqueeze(-1)).sum(dim=1)


class CausalConv1d(nn.Module):
    """因果膨胀卷积（chomp 截断未来侧填充）。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCN(nn.Module):
    """式(14)(15)：2 层因果膨胀卷积，kernel 3、dilation {1, 2}。"""

    def __init__(self, in_ch: int, out_ch: int, layers: int, kernel: int,
                 dilations: list[int]):
        super().__init__()
        blocks = []
        channels = in_ch
        for dilation in dilations[:layers]:
            blocks.append(CausalConv1d(channels, out_ch, kernel, dilation))
            channels = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → 输出取最后时间步 (B, out_ch)
        h = x.transpose(1, 2)
        for block in self.blocks:
            h = F.relu(block(h))
        return h[:, :, -1]


class PISTGEncoder(nn.Module):
    """式(13)-(17)：窗口母线特征 + 电价 → h_STG ∈ R^{d_STG}。"""

    def __init__(
        self,
        num_buses: int,
        num_features: int = 5,
        gcn_hidden: int = 64,
        d_spatial: int = 64,
        gcn_layers: int = 2,
        tcn_layers: int = 2,
        tcn_kernel: int = 3,
        tcn_dilations: list[int] | None = None,
        d_dn_tcn_out: int = 64,
        d_price_tcn_out: int = 16,
    ):
        super().__init__()
        tcn_dilations = tcn_dilations or [1, 2]
        dims = [num_features] + [gcn_hidden] * (gcn_layers - 1) + [d_spatial]
        self.gcn = nn.ModuleList(GCNLayer(a, b) for a, b in zip(dims, dims[1:]))
        self.pooling = AttentionPooling(d_spatial)
        self.dn_tcn = TCN(d_spatial, d_dn_tcn_out, tcn_layers, tcn_kernel,
                          tcn_dilations)
        self.price_tcn = TCN(1, d_price_tcn_out, tcn_layers, tcn_kernel,
                             tcn_dilations)
        self.d_stg = d_dn_tcn_out + d_price_tcn_out

    def forward(
        self, bus_window: torch.Tensor, price_window: torch.Tensor,
        norm_adj: torch.Tensor,
    ) -> torch.Tensor:
        """bus_window (B,W,N,df)、price_window (B,W)、norm_adj (N,N) → (B,d_STG)。"""
        batch, window, n_bus, _ = bus_window.shape
        h = bus_window.reshape(batch * window, n_bus, -1)
        for layer in self.gcn:
            h = layer(h, norm_adj)
        graph = self.pooling(h)  # (B*W, d_S)
        graph = graph.reshape(batch, window, -1)
        h_dn = self.dn_tcn(graph)  # (B, d_DN)
        h_price = self.price_tcn(price_window.unsqueeze(-1))  # (B, d_price)
        return torch.cat([h_dn, h_price], dim=-1)  # 式(16)
