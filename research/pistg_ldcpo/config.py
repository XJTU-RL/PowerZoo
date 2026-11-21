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


@dataclass
class LdcpoConfig:
    """LDCPPO（论文表 II 与 p.10 训练细节）。"""

    actor_hidden: List[int] = field(default_factory=lambda: [128, 128])
    critic_hidden: List[int] = field(default_factory=lambda: [128, 128])
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.015
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    dual_lr: float = 5e-4
    init_multipliers: float = 0.0
    value_coef: float = 0.5
    entropy_coef: float = 0.0  # 论文明示零熵奖励
    max_grad_norm: float = 0.5
    log_std_init: float = -0.5
    ppo_epochs: int = 10  # 论文未明说，PPO 惯例值
    # minibatch 上限（式(33)-(36) 的 "episode or mini-batch"）：单回合
    # T=288 < 4096 时退化为整批更新（与全量运行口径一致）；调小即启用
    # minibatch 切分（每 epoch 随机重排）。
    minibatch_size: int = 4096
    # 对偶乘子下限：0.0 = 式(36d) 逐字（λ 可降至 0，训练末期约束回弹的
    # 已知根源）；>0 时按下限截断（复现实验的稳定性开关，偏离文档记录）。
    lambda_min: float = 0.0
    total_episodes: int = 2000
    # 约束预算 d_k：论文未给数值；默认取零控制基线折扣成本的 30%，
    # 由标定实验冻结（偏离文档记录）。
    cost_budget_fraction: Dict[str, float] = field(
        default_factory=lambda: {"VD": 0.3, "VN": 0.3, "LO": 0.3}
    )
    bootstrap_at_episode_end: bool = True  # 日循环任务用 V(s_T) 自举（论文未明说）


@dataclass
class RunConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    algo: LdcpoConfig = field(default_factory=LdcpoConfig)
    seed: int = 2026
    data_dir: Path = REPO_ROOT / "data" / "external" / "ausgrid-solar-home"
    output_dir: Path = REPO_ROOT / "results" / "pistg_ldcpo"
