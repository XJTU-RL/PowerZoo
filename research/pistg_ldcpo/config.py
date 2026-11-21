"""PISTG-LDCPPO 复现模块的全局配置（论文表 II + 场景参数）。

所有超参数集中在此；与论文逐字一致的值标注论文出处，未明说值标注
"论文未明说"并记录于 docs/pistg-repro-deviations.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
# 算例已按框架 case 库规则入库（configs/systems/33Bus_CBESS.yaml 同步登记）
CASE_FOLDER = REPO_ROOT / "node_systems" / "33Bus_CBESS"


# 场景配置唯一事实源已入框架 envs/cbess/config.py（CBESSConfig）；
# research 侧保留 ScenarioConfig 别名以维持文档语言与既有引用。
from envs.cbess.config import CBESSConfig as ScenarioConfig  # noqa: E402,F401


@dataclass
class EncoderConfig:
    """PISTG 编码器与 PIAE 预训练（论文表 II）。"""

    window: int = 12  # W=12（论文 p.10 数据规模说明）
    gcn_layers: int = 2
    gcn_hidden: int = 64
    d_spatial: int = 64
    tcn_layers: int = 2
    tcn_kernel: int = 3
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2])
    d_stg: int = 80  # h_DN ⊕ h_price 总维（论文未明说切分，默认 64+16）
    d_dn_tcn_out: int = 64
    d_price_tcn_out: int = 16
    decoder_hidden: List[int] = field(default_factory=lambda: [128, 128])
    # PIAM（式(10)）：
    piam_epsilon: float = 1e-6
    piam_nu: float = 1.0
    # PIAE 损失权重（式(22)、表 II）：
    beta_kcl: float = 1.0
    beta_vdrop: float = 1.0
    beta_recon: float = 0.5
    piae_lr: float = 1e-3
    pretrain_episodes: int = 2000  # 随机策略采样回合数（论文数据规模口径）
    pretrain_batch_windows: int = 256


# LDCPPO 超参唯一事实源已入框架 algorithms/centralized/ldcppo.py
from algorithms.centralized.ldcppo import LdcpoConfig  # noqa: E402,F401

@dataclass
class RunConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    algo: LdcpoConfig = field(default_factory=LdcpoConfig)
    seed: int = 2026
    data_dir: Path = REPO_ROOT / "data" / "external" / "ausgrid-solar-home"
    output_dir: Path = REPO_ROOT / "results" / "pistg_ldcpo"
