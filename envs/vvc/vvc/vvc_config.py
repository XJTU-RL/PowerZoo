# -*- coding: utf-8 -*-
"""VVC实际配置字段；系统与实验默认值来自公共YAML解析器。"""
import dataclasses
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class VVCConfig:
    """VVC 环境配置 — 唯一数据源

    字段由VVC核心和包装器实际消费。
    默认值取自 '13Bus' 配置。
    """

    # 系统配置
    system_name: str = "13Bus"
    dss_folder: Optional[str] = None
    dss_file: str = "IEEE13Nodeckt_daily.dss"
    source_bus: str = "sourcebus"
    max_episode_steps: int = 24
    voltage_min: float = 0.95
    voltage_max: float = 1.05
    seed: int = 123456

    # 设备动作维度
    reg_act_num: int = 33
    bat_act_num: Union[int, float] = 33

    # 奖励权重（旧核心直接读取）
    power_w: float = 10.0
    cap_w: float = 0.0303  # 1.0/33
    reg_w: float = 0.0303  # 1.0/33
    soc_w: float = 0.0
    dis_w: float = 0.1818  # 6.0/33

    # 显示与负荷曲线配置
    node_size: int = 500
    shift: int = 10
    show_node_labels: bool = True
    load_noise: bool = True

    # 运行时
    scale: float = 1.0
    use_render: bool = False
    useS: bool = False
    state_type: str = "EP"
    record_node: bool = False
    dss_act: bool = False
    env_name: str = "13Bus"  # for backward compat (close() 需要)

    @classmethod
    def from_env_args(cls, env_args):
        """旧入口统一转换；正式训练在 ConfigLoader 处严格检查。"""
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, env_args)

    def to_env_info(self) -> dict:
        """转为 env_register.make_base_env 期望的 dict 格式 (legacy compat)

        只提供 Env.__init__() 实际读取的控制与奖励字段。
        """
        info = {
            "system_name": self.system_name,
            "dss_file": self.dss_file,
            "max_episode_steps": self.max_episode_steps,
            "voltage_min": self.voltage_min,
            "voltage_max": self.voltage_max,
            "reg_act_num": self.reg_act_num,
            "bat_act_num": self.bat_act_num,
            "power_w": self.power_w,
            "cap_w": self.cap_w,
            "reg_w": self.reg_w,
            "soc_w": self.soc_w,
            "dis_w": self.dis_w,
            "scale": self.scale,
        }
        return info

    def to_sys_info(self) -> dict:
        """旧核心需要的系统视图（不读取另一份默认配置）"""
        return {
            "source_bus": self.source_bus,
            "node_size": self.node_size,
            "shift": self.shift,
            "show_node_labels": self.show_node_labels,
            "load_noise": self.load_noise,
        }
