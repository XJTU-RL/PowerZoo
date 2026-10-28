# -*- coding: utf-8 -*-
"""
SmartGrid 环境统一配置系统

设计原则：
1. 单一数据源 - 所有配置通过 SmartGridConfig 传递
2. 类型安全 - 使用 dataclass 和类型注解
3. 验证完整 - 在 __post_init__ 中验证所有参数
4. 清晰层次 - DeviceConfig → SmartGridConfig → 环境/电路
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import math
from envs.smartgrid.data_process.dataset import DataConfig


@dataclass
class DeviceConfig:
    """设备控制配置

    统一描述电容器、调压器、电池、光伏的控制参数
    """

    action_num: Union[int, float] = 33  # 动作数量，float('inf') 表示连续控制
    control_enabled: bool = True  # 是否启用控制

    @property
    def is_continuous(self) -> bool:
        """是否为连续控制模式"""
        return self.action_num == float("inf") or self.action_num == math.inf

    def validate(self, device_type: str) -> None:
        """验证配置有效性"""
        if self.control_enabled:
            if not self.is_continuous and self.action_num < 2:
                raise ValueError(
                    f"{device_type} action_num 必须 >= 2 或为 inf，当前值: {self.action_num}"
                )


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
    """SmartGrid 环境统一配置

    这是整个参数传递链的唯一数据源。
    所有组件（Env, Circuits, 节点）都从这个配置获取参数。

    使用示例:
            config = SmartGridConfig.from_env_args({'system_name': '34Bus_PV'})
            env = Env(config)
    """

    # === 基础环境配置 ===
    env_name: str = "34Bus_pv"
    system_name: str = "34Bus_PV"
    dss_folder: Optional[str] = None
    dss_file: str = "ieee34Mod1_duty.dss"
    max_episode_steps: int = 360
    seed: int = 123456
    data: DataConfig = field(default_factory=DataConfig)

    # === 设备控制配置 ===
    # 使用 default_factory 避免可变默认值问题
    capacitor: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=2))
    regulator: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=33))
    battery: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=33))
    pv: DeviceConfig = field(
        default_factory=lambda: DeviceConfig(
            action_num=float("inf"), control_enabled=True
        )
    )

    # === 奖励配置 ===
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # === 约束配置 ===
    voltage_min: float = 0.95
    voltage_max: float = 1.05

    # === 运行时配置 ===
    for_LLM: bool = False
    use_cmdp: bool = True
    voltage_threshold: float = 0.02
    power_base: float = 0.02
    use_action_smoothing: bool = True
    use_lagrangian_reward: bool = False
    dss_act: bool = False  # 是否使用 OpenDSS 自动控制
    pv_data_source: Optional[str] = None
    temperature_data_source: Optional[str] = None
    lambda_init: float = 1.0
    lambda_lr: float = 0.001
    target_cost: float = 0.01
    lambda_max: float = 100.0
    lambda_min: float = 0.0
    lambda_update_strategy: str = "standard"
    lambda_momentum: float = 0.0
    lambda_adaptive_lr: bool = False

    # === 显示配置 ===
    useS: bool = False
    big2small: bool = False
    state_type: str = "EP"
    use_render: bool = False
    record_node: bool = False
    load_noise: bool = False

    source_bus: str = "sourcebus"
    node_size: int = 300
    shift: int = 50
    show_node_labels: bool = False
    scale: float = 1.0

    # === Worker 配置 ===
    worker_idx: Optional[int] = None

    # === PV 方案 ===
    pv_plan: Optional[str] = None  # PV方案: none/conservative/optimized/aggressive

    def __post_init__(self):
        """验证并处理配置"""
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if self.useS or self.big2small or self.state_type != "EP" or self.for_LLM:
            raise ValueError(
                "SmartGrid 当前支持 EP 全状态适配；useS/big2small/for_LLM 尚未接入"
            )
        if (
            not self.capacitor.control_enabled
            or not self.regulator.control_enabled
            or not self.battery.control_enabled
            or self.capacitor.action_num != 2
            or self.regulator.is_continuous
            or self.dss_act
        ):
            raise ValueError("当前支持启用 CRB、二状态电容器、离散调压器和策略控制")
        for device in (self.regulator, self.battery, self.pv):
            if not device.is_continuous and int(device.action_num) != device.action_num:
                raise ValueError("离散 action_num 必须是整数")
        if (
            self.battery.action_num != float("inf")
            and int(self.battery.action_num) % 2 != 1
        ):
            raise ValueError("电池离散动作数必须为奇数，包含零功率动作")
        if (
            self.data.source == "synthetic"
            and self.max_episode_steps * self.data.step_seconds > 86400
        ):
            raise ValueError("合成场景长度不能超过一天")
        if (
            self.pv_data_source
            or self.temperature_data_source
            or self.load_noise
            or self.scale != 1.0
        ):
            raise ValueError(
                "旧时序路径/scale/load_noise 已移除，请使用 data.path/load_scale/noise"
            )
        # 验证设备配置
        self.capacitor.validate("Capacitor")
        self.regulator.validate("Regulator")
        self.battery.validate("Battery")
        if self.pv.control_enabled:
            self.pv.validate("PV")
        from dataclasses import asdict

        for name, value in asdict(self.reward_weights).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reward_weights.{name} 必须是非负有限值")
        if not math.isfinite(self.power_base) or self.power_base <= 0:
            raise ValueError("power_base 必须为正有限值")
        for name in (
            "voltage_threshold",
            "target_cost",
            "lambda_min",
            "lambda_init",
            "lambda_max",
            "lambda_lr",
            "lambda_momentum",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须为有限值")
        if not 0 <= self.voltage_threshold < 1 or self.target_cost < 0:
            raise ValueError("电压死区或 target_cost 无效")
        if not 0 <= self.lambda_min <= self.lambda_init <= self.lambda_max:
            raise ValueError("lambda_min/init/max 的顺序无效")
        if self.lambda_lr < 0 or not 0 <= self.lambda_momentum < 1:
            raise ValueError("lambda_lr 或 momentum 无效")

        # 验证时间步
        if self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps 必须 >= 1，当前值: {self.max_episode_steps}"
            )

        # 验证电压约束
        if not 0 < self.voltage_min < self.voltage_max < 2:
            raise ValueError(
                f"voltage_min({self.voltage_min}) 必须小于 voltage_max({self.voltage_max})"
            )

    # === 便捷属性（兼容旧代码）===
    @property
    def reg_act_num(self) -> int:
        """调压器动作数量"""
        return (
            int(self.regulator.action_num) if not self.regulator.is_continuous else 33
        )

    @property
    def bat_act_num(self) -> Union[int, float]:
        """电池动作数量"""
        return self.battery.action_num

    @property
    def pv_act_num(self) -> Union[int, float]:
        """光伏动作数量"""
        return self.pv.action_num

    @property
    def pv_control_enabled(self) -> bool:
        """是否启用光伏控制"""
        return self.pv.control_enabled

    @property
    def RBP_act_num(self) -> tuple:
        """调压器、电池、光伏动作数量元组（用于 Circuits）"""
        return (self.reg_act_num, self.bat_act_num, self.pv_act_num)

    def to_info_dict(self) -> Dict[str, Any]:
        """转换为旧格式 info 字典（向后兼容）

        这个方法用于兼容旧代码，新代码应直接使用 SmartGridConfig
        """
        from dataclasses import asdict

        info = {
            # 基础配置
            "env_name": self.env_name,
            "system_name": self.system_name,
            "dss_file": self.dss_file,
            "max_episode_steps": self.max_episode_steps,
            # 设备动作配置
            "reg_act_num": self.reg_act_num,
            "bat_act_num": self.bat_act_num,
            "pv_act_num": self.pv_act_num,
            "pv_control": self.pv_control_enabled,
            # 奖励权重
            **self.reward_weights.to_dict(),
            "reward_weights": asdict(self.reward_weights),
            "voltage_range": (self.voltage_min, self.voltage_max),
            # 约束
            "voltage_min": self.voltage_min,
            "voltage_max": self.voltage_max,
            # 运行时
            "for_LLM": self.for_LLM,
            "dss_act": self.dss_act,
            "load_noise": self.load_noise,
            "use_render": self.use_render,
            "record_node": self.record_node,
            "use_cmdp": self.use_cmdp,
            # 显示
            "source_bus": self.source_bus,
            "node_size": self.node_size,
            "shift": self.shift,
            "show_node_labels": self.show_node_labels,
            "scale": self.scale,
            # Worker
            "worker_idx": self.worker_idx,
            # PV 方案
            "pv_plan": self.pv_plan,
        }
        for key in (
            "voltage_threshold",
            "power_base",
            "use_action_smoothing",
            "use_lagrangian_reward",
            "dss_folder",
            "pv_data_source",
            "temperature_data_source",
            "lambda_init",
            "lambda_lr",
            "target_cost",
            "lambda_max",
            "lambda_min",
            "lambda_update_strategy",
            "lambda_momentum",
            "lambda_adaptive_lr",
        ):
            info[key] = getattr(self, key)
        return info

    @classmethod
    def from_dict(cls, config_dict):
        return cls.from_env_args(config_dict)

    @classmethod
    def from_env_args(cls, env_args):
        """旧入口统一转换；正式训练在 ConfigLoader 处严格检查。"""
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, env_args)

    def with_worker_idx(self, worker_idx: int) -> "SmartGridConfig":
        """创建带有 worker_idx 的新配置副本"""
        import copy

        new_config = copy.deepcopy(self)
        new_config.worker_idx = worker_idx
        return new_config

    def __repr__(self) -> str:
        return (
            f"SmartGridConfig(\n"
            f"  env_name='{self.env_name}',\n"
            f"  system_name='{self.system_name}',\n"
            f"  max_episode_steps={self.max_episode_steps},\n"
            f"  devices=(\n"
            f"    regulator: action_num={self.reg_act_num},\n"
            f"    battery: action_num={self.bat_act_num},\n"
            f"    pv: action_num={self.pv_act_num}, enabled={self.pv_control_enabled}\n"
            f"  )\n"
            f")"
        )
