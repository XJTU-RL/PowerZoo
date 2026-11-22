"""CBESS 场景配置（注册表口径）：论文式(7) 场景参数 + 框架接入字段。

字段与论文 Sec. IV-A 明示值逐字对齐（出处见 research/pistg_ldcpo/
README.md 映射表）；标定参数依据见 docs/pistg-repro-deviations.md。
research 侧以 `ScenarioConfig` 别名复用本类（单一事实源）。
"""

from dataclasses import dataclass, field


@dataclass
class CBESSConfig:
    """PISTG-LDCPPO 论文场景（IEEE 33-bus + 4 CBESS + 26 DER 母线）。"""

    # 框架接入：node_systems 算例名与 Ausgrid 剖面目录（相对仓库根或绝对）。
    system: str = "33Bus_CBESS"
    data_dir: str = "data/external/ausgrid-solar-home"
    split: str = "train"
    seed: int = 1
    worker_idx: int = 0

    # 明示值：288 步 × 300 s = 一天。
    steps_per_day: int = 288
    step_seconds: int = 300
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    # 明示值：4 台 CBESS @ 6/11/16/31，各 1000 kWh、SOC 初 50%、
    # ±250 kW、q ±50 kVAr、η=0.95、退化 0.03 AUD/kWh。
    cbess_buses: list[int] = field(default_factory=lambda: [6, 11, 16, 31])
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
    load_power_factor: float = 0.85  # 标准 33-bus 系统功率因数
    # 电价整体缩放（v2 标定冻结）：零控制 ≈1378 AUD 对齐论文表 III 量级。
    price_level: float = 0.70
