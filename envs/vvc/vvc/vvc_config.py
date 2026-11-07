"""经典VVC任务配置；容量/相别由DSS提供，时序与物理执行复用公共层。"""

from dataclasses import dataclass, field
import math
from envs.powergrid.config import DataConfig, validate_operation


@dataclass
class VVCConfig:
    system_name: str = "13Bus"
    dss_folder: str | None = None
    dss_file: str = "network.dss"
    overlays: list[str] = field(default_factory=list)
    max_episode_steps: int = 24
    seed: int = 123456
    data: DataConfig = field(default_factory=lambda: DataConfig(step_seconds=3600))
    reg_act_num: int = 33
    bat_act_num: int | float = 33
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    state_type: str = "EP"
    power_w: float = 10.0
    cap_w: float = 0.0303
    reg_w: float = 0.0303
    soc_w: float = 0.0
    dis_w: float = 0.1818

    def __post_init__(self):
        validate_operation(self)
        for key in ("power_w", "cap_w", "reg_w", "soc_w", "dis_w"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key}必须为非负有限值")

    @classmethod
    def from_env_args(cls, values):
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, values)
