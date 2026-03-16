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
    system_name: str = '13Bus'
    dss_file: str = 'IEEE13Nodeckt_daily.dss'
    dss_folder: Optional[str] = None
    max_episode_steps: int = 24
    seed: int = 123456

    # 智能体配置
    n_consumer_agents: int = 8

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
    def from_env_args(cls, env_args: Dict[str, Any]) -> 'StackelbergConfig':
        """从 env_args 构建配置

        修复历史键名不匹配问题 (num_steps vs max_episode_steps)。
        未知键会被安全忽略。

        Args:
            env_args: 训练脚本传入的环境参数 dict

        Returns:
            填充完毕的 StackelbergConfig 实例
        """
        config = cls()

        # 系统名：优先 system_name，回退到 env_name 去前缀
        config.system_name = env_args.get(
            'system_name',
            env_args.get('env_name', 'stackelberg_13Bus').replace('stackelberg_', '')
        )

        # 统一键名 (修复 num_steps vs max_episode_steps)
        config.max_episode_steps = env_args.get(
            'max_episode_steps',
            env_args.get('num_steps', config.max_episode_steps)
        )

        # DSS 文件
        config.dss_file = env_args.get('dss_file', config.dss_file)
        config.dss_folder = env_args.get('dss_folder', config.dss_folder)

        # dict 子配置直传
        dict_fields = [
            'tou_config', 'tier_config', 'reward_weights',
            'load_aggregation', 'async_config', 'monitoring_config', 'n1_security',
        ]
        for f in dict_fields:
            if f in env_args and env_args[f] is not None:
                setattr(config, f, env_args[f])

        # 标量参数
        scalar_fields = [
            'n_consumer_agents', 'seed', 'use_render', 'use_load_noise',
            'scale', 'worker_idx', 'exp_name',
        ]
        for f in scalar_fields:
            if f in env_args and env_args[f] is not None:
                setattr(config, f, env_args[f])

        return config
