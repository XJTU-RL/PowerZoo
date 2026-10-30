"""顺序定价任务的配置：功率/容量来自DSS，效用与价格单位为元。"""

from dataclasses import dataclass, field
import math
from envs.powergrid.config import DataConfig


@dataclass
class ConsumerConfig:
    name: str = ""
    loads: list[str] = field(default_factory=list)
    discomfort_yuan_per_kwh: float = 1.0


@dataclass
class StackelbergConfig:
    system_name: str = "PricingSmall"
    dss_folder: str | None = None
    dss_file: str = "network.dss"
    overlays: list[str] = field(default_factory=list)
    max_episode_steps: int = 24
    seed: int = 12345
    data: DataConfig = field(default_factory=lambda: DataConfig(step_seconds=3600))
    consumers: list[ConsumerConfig] = field(default_factory=list)
    retail_prices_yuan_per_kwh: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    load_fractions: list[float] = field(default_factory=lambda: [0.6, 0.8, 1.0])
    wholesale_prices_yuan_per_kwh: list[float] = field(
        default_factory=lambda: [0.3] * 24
    )
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    network_cost_yuan_per_hour: float = 100.0

    @classmethod
    def from_env_args(cls, values):
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, values)

    def validate(self):
        if self.data.irradiance_scale != 1.0:
            raise ValueError(
                "定价首版没有PV，data.irradiance_scale没有消费者；请使用默认值"
            )
        if not self.consumers or len({c.name for c in self.consumers}) != len(
            self.consumers
        ):
            raise ValueError("consumers必须非空且名称唯一")
        for c in self.consumers:
            if (
                not c.name
                or not c.loads
                or not math.isfinite(c.discomfort_yuan_per_kwh)
                or c.discomfort_yuan_per_kwh <= 0
            ):
                raise ValueError(
                    "消费者需要名称、真实Load清单和正的discomfort_yuan_per_kwh"
                )
        for name in ("retail_prices_yuan_per_kwh", "load_fractions"):
            values = getattr(self, name)
            if (
                len(values) < 2
                or values != sorted(set(values))
                or not all(math.isfinite(v) for v in values)
            ):
                raise ValueError(f"{name}必须为递增、无重复的有限列表")
        if (
            min(self.retail_prices_yuan_per_kwh) < 0
            or not 0 < min(self.load_fractions) <= max(self.load_fractions) == 1
        ):
            raise ValueError("零售价格不能为负，负荷比例必须在(0,1]且包含1")
        if len(self.wholesale_prices_yuan_per_kwh) != 24 or any(
            not math.isfinite(v) or v < 0 for v in self.wholesale_prices_yuan_per_kwh
        ):
            raise ValueError("wholesale_prices_yuan_per_kwh须为24个非负小时价格")
        if (
            not math.isfinite(self.network_cost_yuan_per_hour)
            or self.network_cost_yuan_per_hour < 0
        ):
            raise ValueError("network_cost_yuan_per_hour须非负有限")
        if (
            self.max_episode_steps < 1
            or self.max_episode_steps * self.data.step_seconds > 86400
        ):
            raise ValueError("回合必须在一天以内")
        if not 0 < self.voltage_min < self.voltage_max < 2:
            raise ValueError("无效电压范围")
        return True
