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


@dataclass
class ScenarioConfig:
    """场景参数（论文 Sec. IV-A）。"""

    # 明示值：288 步 × 300 s = 一天。
    steps_per_day: int = 288
    step_seconds: int = 300
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    # 明示值：4 台 CBESS @ 6/11/16/31，各 1000 kWh、SOC 初 50%、
    # ±250 kW、q ±50 kVAr、η=0.95、退化 0.03 AUD/kWh。
    cbess_buses: List[int] = field(default_factory=lambda: [6, 11, 16, 31])
    cbess_capacity_kwh: float = 1000.0
    cbess_soc_init: float = 0.5
    cbess_soc_min: float = 0.1
    cbess_soc_max: float = 0.9
    cbess_p_limit_kw: float = 250.0
    cbess_q_limit_kvar: float = 50.0
    cbess_efficiency: float = 0.95  # η_ch = η_dis
    cbess_degradation_aud_per_kwh: float = 0.03
    # S_CB 取 √(250²+50²) kVA，使满功率放电时仍可输出 ±50 kVAr（式(1e)）。
    cbess_s_rating_kva: float = (250.0**2 + 50.0**2) ** 0.5
    # PV 逆变器容量 = 1.1 × PV 峰值（论文未明说，常见超配）。
    pv_inverter_oversize: float = 1.1
    pv_penetration: float = 0.9  # PV 峰值 = 0.9 × 母线峰荷（表 I 脚注口径）
    # 论文未明说的标定参数（M2 冻结）：
    der_bus_daily_energy_kwh: float = 850.0  # Fig.5(b) 基荷柱 ≈850 kWh/母线/日
    ev_daily_energy_kwh: float = 350.0  # Fig.5(b) EV 柱 ≈350 kWh/母线/日
    load_power_factor: float = 0.85  # 标准 33-bus 系统功率因数 3715/√(3715²+2300²)
    non_der_scale: float = 1.0  # 非 DER 母线 = 标准铭牌 × 日形状
    # 电价整体缩放（M2 标定冻结）：0.55 使零控制日成本 ≈1400 AUD，
    # 与论文表 III 有控制成本 1382 AUD/日 同量级。
    price_level: float = 0.55


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
    minibatch_size: int = 4096  # 论文未明说；288 步/回合 × 若干回合
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
