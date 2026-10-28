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
    env_args dict 覆盖。ESS、聚合、异步观测、监控子配置仍以字典交给旧核心，
    允许的键在本类校验，不能用任意字典隐藏未实现参数。
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
    history_length: int = 5
    der_profile: Optional[list] = None
    dss_act: bool = False

    # 子配置 (dict 类型，结构灵活)
    load_aggregation: Optional[dict] = None
    async_config: Optional[dict] = None
    monitoring_config: Optional[dict] = None

    # 运行时
    worker_idx: Optional[int] = None
    use_load_noise: bool = False
    scale: float = 1.0
    exp_name: Optional[str] = None

    @classmethod
    def from_env_args(cls, env_args):
        """旧入口统一转换；正式训练在 ConfigLoader 处严格检查。"""
        from utils.config_validation import legacy_env_config

        return legacy_env_config(cls, env_args)

    def validate(self):
        # 这些子配置尚由旧核心消费字典；在配置定义所在文件限制真实支持的键。
        supported = {
            "ess_config": {
                "capacity",
                "max_power",
                "eta_c",
                "eta_o",
                "retention_factor",
                "initial_soc",
            },
            "stackelberg_config": {"price_bounds", "dr_incentive_bounds"},
            "async_config": {"partial_observability", "observation_noise"},
            "monitoring_config": {
                "enable",
                "log_dir",
                "save_interval",
                "plot_interval",
                "track_convergence",
                "track_nash_gap",
            },
            "load_aggregation": {
                "method",
                "use_electrical_distance",
                "priority_weights",
                "spectral_clusters",
                "consider_power_flow",
            },
        }
        for name, fields in supported.items():
            values = getattr(self, name) or {}
            unknown = set(values) - fields
            if unknown:
                raise ValueError(f"{name}: 未实现或未知参数 {sorted(unknown)}")
        aggregation = self.load_aggregation or {}
        if aggregation.get("method", "zone") not in ("zone", "priority", "graph"):
            raise ValueError(
                "正式控制链只接入zone/priority/graph聚合，动态重聚合尚未接入"
            )
        if (self.async_config or {}).get("observation_noise", 0) < 0:
            raise ValueError("observation_noise不能为负")
        if self.n_consumer_agents < 1 or self.history_length < 1:
            raise ValueError("Stackelberg消费者数量和历史长度必须为正整数")
        return True
