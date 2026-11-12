"""运行控制的共享时序参数；不读取文件或初始化物理引擎。"""

from dataclasses import asdict, dataclass, field
import math

GROUPS = ("residential", "commercial", "industrial")


@dataclass
class DataConfig:
    source: str = "synthetic"
    path: str | None = None  # manifest.json；相对路径以仓库根目录解析
    manifest_sha256: str | None = None
    seed: int = 2026  # 数据种子与策略/worker 种子相互独立
    step_seconds: int = 60
    episodes_per_split: int = 12
    split: str = "train"
    load_scale: float = 1.0
    irradiance_scale: float = 1.0
    noise: float = 0.025
    load_groups: dict[str, str] = field(default_factory=dict)
    scenario_names: list[str] | None = None  # 读取期按 manifest 场景名过滤；不改变生成

    def __post_init__(self):
        if self.source not in ("synthetic", "files"):
            raise ValueError("data.source 必须为 synthetic 或 files")
        if self.source == "files" and not self.path:
            raise ValueError("files 数据源必须指定 data.path")
        if self.source == "files":
            for name in (
                "seed",
                "episodes_per_split",
                "load_scale",
                "irradiance_scale",
                "noise",
            ):
                if getattr(self, name) != self.__dataclass_fields__[name].default:
                    raise ValueError(
                        f"data.{name}只控制合成数据，files模式请删除该生成设置"
                    )
        if self.split not in ("train", "validation", "test"):
            raise ValueError("data.split 必须为 train/validation/test")
        if not 0 <= self.seed < 2**32 or not 1 <= self.step_seconds <= 3600:
            raise ValueError("无效数据种子或 step_seconds（1..3600）")
        if self.episodes_per_split < 6:
            raise ValueError("episodes_per_split 至少为6；默认12可轮转覆盖全部八类场景")
        if not (0 < self.load_scale <= 3 and 0 < self.irradiance_scale <= 1.5):
            raise ValueError("数据倍率超出合成器支持范围")
        if not 0 <= self.noise <= 0.2:
            raise ValueError("data.noise 必须在 [0, 0.2]")
        if any(g not in GROUPS for g in self.load_groups.values()):
            raise ValueError(f"load_groups 的值必须属于 {GROUPS}")
        if self.scenario_names is not None and (
            not isinstance(self.scenario_names, list)
            or not self.scenario_names
            or any(not isinstance(s, str) or not s.strip() for s in self.scenario_names)
            or len(set(self.scenario_names)) != len(self.scenario_names)
        ):
            raise ValueError("scenario_names 必须是非空、不重复的场景名列表")


def validate_observation_projection(buses, neighborhood):
    """关键母线观测投影的配置校验；两任务（VVC/SmartGrid）共用。"""
    if buses is not None and neighborhood is not None:
        raise ValueError(
            "observation_buses 与 observation_neighborhood 互斥，只能选择一种投影方式"
        )
    if neighborhood is not None and (
        not isinstance(neighborhood, int)
        or isinstance(neighborhood, bool)
        or not 1 <= neighborhood <= 4
    ):
        raise ValueError("observation_neighborhood 必须是 1..4 的整数跳数")
    if buses is not None:
        if not isinstance(buses, list) or not buses:
            raise ValueError("observation_buses 必须是非空母线名列表")
        if any(
            not isinstance(b, str) or not b.strip() for b in buses
        ) or len(set(b.lower() for b in buses)) != len(buses):
            raise ValueError("observation_buses 含空名或重复项（大小写不敏感）")


def validate_operation(config):
    if isinstance(config.data, dict):
        config.data = DataConfig(**config.data)
    if config.state_type != "EP":
        raise ValueError("同步运行控制基准只支持EP共享全状态观测")
    if getattr(config, "solve_algorithm", "snapshot") not in ("snapshot", "direct"):
        raise ValueError("solve_algorithm 必须是 snapshot（Newton）或 direct（前推回代）")
    n = config.reg_act_num
    if not math.isfinite(n) or n < 2 or int(n) != n:
        raise ValueError("reg_act_num必须为>=2的整数")
    n = config.bat_act_num
    if n != math.inf and (not math.isfinite(n) or n < 3 or int(n) != n or n % 2 != 1):
        raise ValueError("bat_act_num必须为>=3的奇数或.inf，保留零功率动作")
    if config.max_episode_steps < 1 or (
        config.data.source == "synthetic"
        and config.max_episode_steps * config.data.step_seconds > 86400
    ):
        raise ValueError("回合长度必须为正，合成场景不能超过一天")
    if not 0 < config.voltage_min < config.voltage_max < 2:
        raise ValueError("电压约束必须满足0<min<max<2")
    validate_observation_projection(
        getattr(config, "observation_buses", None),
        getattr(config, "observation_neighborhood", None),
    )
