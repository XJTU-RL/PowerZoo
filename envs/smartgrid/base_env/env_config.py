"""SmartGrid 配置定义集中在此；文件读取、生成数据与仿真均不在这里执行。"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union
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
            raise ValueError("episodes_per_split 至少为6，覆盖六类场景")
        if not (0 < self.load_scale <= 3 and 0 < self.irradiance_scale <= 1.5):
            raise ValueError("数据倍率超出合成器支持范围")
        if not 0 <= self.noise <= 0.2:
            raise ValueError("data.noise 必须在 [0, 0.2]")
        if any(g not in GROUPS for g in self.load_groups.values()):
            raise ValueError(f"load_groups 的值必须属于 {GROUPS}")


@dataclass
class RewardWeights:
    """奖励权重配置"""

    power_loss: float = 1.0  # 功率损耗权重
    capacitor: float = 0.0303  # 电容器切换权重
    regulator: float = 0.0303  # 调压器调节权重
    battery_soc: float = 0.0  # 电池SOC权重
    battery_discharge: float = 0.303  # 电池放电权重
    pv_control: float = 0.0606  # 光伏控制权重

    control: float = 0.5
    pv: float = 0.8
    action_smoothing: float = 0.3

    def to_dict(self) -> Dict[str, float]:
        """转换为旧格式字典（兼容性）"""
        return {
            "power_w": self.power_loss,
            "cap_w": self.capacitor,
            "reg_w": self.regulator,
            "soc_w": self.battery_soc,
            "dis_w": self.battery_discharge,
            "pv_w": self.pv_control,
        }


@dataclass
class SmartGridConfig:
    """仅声明已实现且有实际消费者的参数；正式入口严格拒绝未知字段。"""

    env_name: str = "34Bus_pv"
    system_name: str = "34Bus_PV"
    dss_folder: Optional[str] = None
    dss_file: str = "ieee34Mod1_duty.dss"
    max_episode_steps: int = 360
    seed: int = 123456
    data: DataConfig = field(default_factory=DataConfig)

    # 设备动作：电容器固定为物理二状态；调压器只支持离散档位。
    reg_act_num: int = 33
    bat_act_num: Union[int, float] = 33
    pv_act_num: Union[int, float] = float("inf")
    pv_control_enabled: bool = True
    pv_plan: Optional[str] = None
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    state_type: str = "EP"

    use_cmdp: bool = True
    use_lagrangian_reward: bool = False
    voltage_threshold: float = 0.02
    power_base: float = 0.02
    use_action_smoothing: bool = True
    lambda_init: float = 1.0
    lambda_lr: float = 0.001
    target_cost: float = 0.01
    lambda_max: float = 100.0
    lambda_min: float = 0.0
    lambda_update_strategy: str = "standard"
    lambda_momentum: float = 0.0
    lambda_adaptive_lr: bool = False

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if self.state_type != "EP":
            raise ValueError("SmartGrid只支持EP全状态观测")
        if (
            not math.isfinite(self.reg_act_num)
            or self.reg_act_num < 2
            or int(self.reg_act_num) != self.reg_act_num
        ):
            raise ValueError("reg_act_num必须是>=2的离散档位数")
        for name in ("bat_act_num", "pv_act_num"):
            n = getattr(self, name)
            if n != math.inf and (not math.isfinite(n) or n < 2 or int(n) != n):
                raise ValueError(f"{name}必须是>=2的整数或.inf")
        if self.bat_act_num != math.inf and self.bat_act_num % 2 != 1:
            raise ValueError("电池离散动作数必须为奇数，包含零功率动作")
        if self.max_episode_steps < 1 or (
            self.data.source == "synthetic"
            and self.max_episode_steps * self.data.step_seconds > 86400
        ):
            raise ValueError("回合长度必须为正，合成场景不能超过一天")
        if not 0 < self.voltage_min < self.voltage_max < 2:
            raise ValueError("电压约束必须满足0<min<max<2")
        for name, value in asdict(self.reward_weights).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reward_weights.{name}必须为非负有限值")
        for name in (
            "power_base",
            "voltage_threshold",
            "lambda_min",
            "lambda_init",
            "lambda_max",
            "lambda_lr",
            "lambda_momentum",
            "target_cost",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name}必须为有限值")
        if (
            self.power_base <= 0
            or not 0 <= self.voltage_threshold < 1
            or self.target_cost < 0
        ):
            raise ValueError("网损基准、电压死区或target_cost无效")
        if not 0 <= self.lambda_min <= self.lambda_init <= self.lambda_max:
            raise ValueError("lambda_min/init/max的顺序无效")
        if self.lambda_lr < 0 or not 0 <= self.lambda_momentum < 1:
            raise ValueError("lambda_lr或momentum无效")
        if self.lambda_update_strategy not in ("standard", "augmented", "adaptive"):
            raise ValueError("未知lambda更新策略")
        if self.use_lagrangian_reward and not self.use_cmdp:
            raise ValueError("拉格朗日奖励要求启用use_cmdp，不能静默失去电压惩罚")

    @property
    def RBP_act_num(self):
        return self.reg_act_num, self.bat_act_num, self.pv_act_num

    def to_info_dict(self):
        # 只兼容旧奖励调用；正式运行直接消费本配置对象。
        return {
            **asdict(self),
            **self.reward_weights.to_dict(),
            "voltage_range": (self.voltage_min, self.voltage_max),
            "pv_control": self.pv_control_enabled,
        }

    @classmethod
    def from_dict(cls, values):
        return cls.from_env_args(values)

    @classmethod
    def from_env_args(cls, values):
        from utils.config_validation import build_dataclass, normalize_env

        return build_dataclass(cls, normalize_env(values, "smartgrid"))
