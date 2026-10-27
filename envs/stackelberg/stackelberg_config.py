# -*- coding: utf-8 -*-
"""Stackelberg 博弈环境配置 — 替代 _parse_config 手工拼装

StackelbergConfig 是 Stackelberg 环境的唯一配置数据源。
通过 from_env_args() 从 env_args dict 构建，修复了历史键名不匹配问题
(num_steps vs max_episode_steps)，消除了 _parse_config 中的硬编码
default_dss_files 映射和重复 YAML 读取。
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StackelbergConfig:
    """Stackelberg VVC 环境配置 — 唯一数据源

    所有字段都有合理默认值，可通过 from_env_args() 从训练脚本的
    env_args dict 覆盖。子配置 (tou_config 等) 保留为 dict 类型，
    因为 StackelbergBaseEnv 尚未迁移到 typed config。
    """

    # 系统配置
    system_name: str = "13Bus"
    dss_file: str = "IEEE13Nodeckt_daily.dss"
    dss_folder: Optional[str] = None
    max_episode_steps: int = 24
    seed: int = 123456

    # 智能体配置
    n_consumer_agents: int = 8
    state_type: str = "EP"
    ess_config: dict = field(default_factory=dict)
    stackelberg_config: dict = field(default_factory=dict)
    circuit_adapter_config: dict = field(default_factory=dict)
    history_length: int = 5
    der_profile: Optional[list] = None
    dss_act: bool = False

    # 子配置 (dict 类型，结构灵活)
    tou_config: Optional[dict] = None
    tier_config: Optional[dict] = None
    reward_weights: Optional[dict] = None
    load_aggregation: Optional[dict] = None
    async_config: Optional[dict] = None
    monitoring_config: Optional[dict] = None
    n1_security: Optional[dict] = None

    # 运行时
    worker_idx: Optional[int] = None
    use_render: bool = False
    use_load_noise: bool = False
    scale: float = 1.0
    exp_name: Optional[str] = None

    @classmethod
    def from_env_args(cls, env_args):
        """旧入口统一转换；正式训练在 ConfigLoader 处严格检查。"""
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, env_args)
