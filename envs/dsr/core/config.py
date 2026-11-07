"""源连接静态恢复配置：真实控制归属、故障输入和服务评价。"""

from dataclasses import dataclass, field
import math


@dataclass
class FaultDataConfig:
    path: str = "data/restoration/small/manifest.json"
    manifest_sha256: str | None = None
    split: str = "train"
    step_seconds: int = 300

    def __post_init__(self):
        if not self.path or self.split not in ("train", "validation", "test"):
            raise ValueError("必须指定故障清单及合法split")
        if self.step_seconds <= 0:
            raise ValueError("恢复操作区间必须为正秒数")


@dataclass
class ControlGroup:
    name: str = ""
    controls: list[str] = field(default_factory=list)


@dataclass
class DSRConfig:
    system_name: str = "RestorationSmall"
    dss_folder: str | None = None
    dss_file: str = "network.dss"
    overlays: list[str] = field(default_factory=list)
    max_episode_steps: int = 12
    seed: int = 123456
    state_type: str = "EP"
    data: FaultDataConfig = field(default_factory=FaultDataConfig)
    control_groups: list[ControlGroup] = field(default_factory=list)
    priorities: dict[str, float] = field(default_factory=dict)
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    operation_cost: float = 0.01
    invalid_action_cost: float = 0.1
    voltage_cost: float = 1.0
    thermal_cost: float = 1.0

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = FaultDataConfig(**self.data)
        self.control_groups = [
            ControlGroup(**g) if isinstance(g, dict) else g for g in self.control_groups
        ]
        if self.state_type != "EP" or self.max_episode_steps < 1:
            raise ValueError("恢复任务为EP共享团队收益，回合长度必须为正")
        if not 0 < self.voltage_min < self.voltage_max < 2:
            raise ValueError("恢复服务电压边界无效")
        for value in [
            *self.priorities.values(),
            self.operation_cost,
            self.invalid_action_cost,
            self.voltage_cost,
            self.thermal_cost,
        ]:
            if not math.isfinite(value) or value < 0:
                raise ValueError("优先级及代价必须非负有限")
        names = [g.name for g in self.control_groups]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("控制组名称必须非空且唯一")
        if any(not group.controls for group in self.control_groups):
            raise ValueError("每个控制组需要真实控制设备")

    @classmethod
    def from_env_args(cls, values):
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, values)
