"""区域协同调度配置：原生设备绑定、物理时序和有单位的费用。"""

from dataclasses import dataclass, field, asdict
import math
from envs.powergrid.config import DataConfig


@dataclass
class RegionConfig:
    name: str = ""
    buses: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)


@dataclass
class MarketConfig:
    base_price_yuan_kwh: float = 0.5
    tou_multipliers: list[float] = field(
        default_factory=lambda: [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.8,
            1.0,
            1.5,
            1.5,
            1.2,
            1.0,
            1.0,
            1.0,
            1.2,
            1.5,
            1.5,
            1.5,
            1.2,
            1.0,
            0.8,
            0.5,
            0.5,
            0.5,
        ]
    )
    sell_ratio: float = 0.8
    carbon_intensity_kg_kwh: float = 0.6
    carbon_price_yuan_kg: float = 0.05

    def __post_init__(self):
        if len(self.tou_multipliers) != 24 or any(
            not math.isfinite(v) or v < 0 for v in self.tou_multipliers
        ):
            raise ValueError("market.tou_multipliers必须是24个非负有限数")
        for key in (
            "base_price_yuan_kwh",
            "sell_ratio",
            "carbon_intensity_kg_kwh",
            "carbon_price_yuan_kg",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"market.{key}必须非负有限")
        if self.sell_ratio > 1:
            raise ValueError("sell_ratio必须在[0,1]")

    def price_at(self, seconds):
        buy = self.base_price_yuan_kwh * self.tou_multipliers[int(seconds // 3600) % 24]
        return buy, buy * self.sell_ratio


@dataclass
class DispatchCosts:
    scale_yuan: float = 100.0
    battery_wear_yuan_kwh: float = 0.01
    curtailment_yuan_kwh: float = 0.05
    voltage_yuan_pu2_h: float = 100000.0
    thermal_yuan_pu2_h: float = 10000.0
    terminal_energy_yuan_kwh: float = 2.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"costs.{key}必须非负有限")
        if self.scale_yuan <= 0:
            raise ValueError("costs.scale_yuan必须为正，且不依赖本轮电价")


@dataclass
class DistrictDispatchConfig:
    system_name: str = "District_34Bus_3Zone"
    dss_folder: str | None = None
    dss_file: str = "network.dss"
    overlays: list[str] = field(default_factory=list)
    seed: int = 123456
    max_episode_steps: int = 96
    data: DataConfig = field(default_factory=lambda: DataConfig(step_seconds=900))
    regions: list[RegionConfig] = field(default_factory=list)
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    use_neighbor_obs: bool = True
    state_type: str = "EP"
    market: MarketConfig = field(default_factory=MarketConfig)
    costs: DispatchCosts = field(default_factory=DispatchCosts)

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.market, dict):
            self.market = MarketConfig(**self.market)
        if isinstance(self.costs, dict):
            self.costs = DispatchCosts(**self.costs)
        self.regions = [
            RegionConfig(**r) if isinstance(r, dict) else r for r in self.regions
        ]
        if self.state_type != "EP":
            raise ValueError("区域协同调度采用EP共享团队价值，不能丢弃部分主体奖励")
        if not 0 < self.voltage_min < self.voltage_max < 2:
            raise ValueError("电压上下界无效")
        if self.max_episode_steps < 1 or (
            self.data.source == "synthetic"
            and self.max_episode_steps * self.data.step_seconds > 86400
        ):
            raise ValueError("合成调度窗口必须为正且不超过一天")
        names = [r.name for r in self.regions]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("区域名称必须非空且唯一")
        for region in self.regions:
            if not region.buses or not region.controls:
                raise ValueError("每个调度区域必须有母线及可控DER")
            if len(region.buses) != len(set(region.buses)) or len(
                region.controls
            ) != len(set(region.controls)):
                raise ValueError(f"{region.name}存在重复绑定")

    @classmethod
    def from_env_args(cls, values):
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, values)
