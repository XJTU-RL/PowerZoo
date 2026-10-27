# -*- coding: utf-8 -*-
"""VVC 环境配置 — 替代 env_register._ENV_INFO 硬编码字典

VVCConfig 是 VVC 环境的唯一配置数据源。
通过 from_env_args() 从 env_args dict 构建，修复了历史键名不匹配问题
(episode_length vs max_episode_steps, pv_control vs pv_control_enabled)，
消除了 _ENV_INFO 中的硬编码默认值。

_ENV_INFO 和 _SYS_INFO 保留在 env_register.py 作为过渡期兜底。
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class VVCConfig:
    """VVC 环境配置 — 唯一数据源

    字段与 _ENV_INFO 条目 + _SYS_INFO 条目一一对应。
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
    pv_act_num: Union[int, float] = 33
    pv_control_enabled: bool = False
    irrad_dss: Optional[str] = None

    # 奖励权重 (与 _ENV_INFO 键一一对应)
    power_w: float = 10.0
    cap_w: float = 0.0303  # 1.0/33
    reg_w: float = 0.0303  # 1.0/33
    soc_w: float = 0.0
    dis_w: float = 0.1818  # 6.0/33

    # 显示配置 (来自 _SYS_INFO)
    node_size: int = 500
    shift: int = 10
    show_node_labels: bool = True
    load_noise: bool = True

    # 运行时
    scale: float = 1.0
    use_render: bool = False
    useS: bool = False
    big2small: bool = False
    state_type: str = "EP"
    record_node: bool = False
    dss_act: bool = False
    for_LLM: bool = False
    env_name: str = "13Bus"  # for backward compat (close() 需要)

    @classmethod
    def from_env_args(cls, env_args):
        """旧入口统一转换；正式训练在 ConfigLoader 处严格检查。"""
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, env_args)

    def to_env_info(self) -> dict:
        """转为 env_register.make_base_env 期望的 dict 格式 (legacy compat)

        合并 _ENV_INFO + _SYS_INFO 的内容，供 Env.__init__() 消费。
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
        if self.pv_control_enabled:
            info["pv"] = True
            info["pv_act_num"] = self.pv_act_num
        if self.irrad_dss:
            info["irrad_dss"] = self.irrad_dss
        if self.for_LLM:
            info["for_LLM"] = True
        return info

    def to_sys_info(self) -> dict:
        """转为 _SYS_INFO 提供的 dict 格式 (legacy compat)"""
        return {
            "source_bus": self.source_bus,
            "node_size": self.node_size,
            "shift": self.shift,
            "show_node_labels": self.show_node_labels,
            "load_noise": self.load_noise,
        }
