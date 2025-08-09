# -*- coding: utf-8 -*-
"""
PowerZoo环境配置类和多智能体包装器
"""
from typing import Optional, Any, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class PowerZooConfig:
    """PowerZoo环境配置类"""
    
    # 基础环境配置
    env_name: str = 'default'
    num_agents: int = 3
    num_env: int = 1
    seed: int = 0
    
    # 动作空间配置
    action_space_mode: str = 'discrete'  # 'discrete' or 'continuous'
    
    # 观测配置
    useS: bool = False  # 是否使用敏感性矩阵
    
    # 其他配置
    max_episode_steps: int = 1000
    reward_type: str = 'default'
    
    def __post_init__(self):
        """后处理初始化"""
        # 验证配置参数
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        
        if self.action_space_mode not in ['discrete', 'continuous']:
            raise ValueError("action_space_mode must be 'discrete' or 'continuous'")
        
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'env_name': self.env_name,
            'num_agents': self.num_agents,
            'num_env': self.num_env,
            'seed': self.seed,
            'action_space_mode': self.action_space_mode,
            'useS': self.useS,
            'max_episode_steps': self.max_episode_steps,
            'reward_type': self.reward_type
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PowerZooConfig':
        """从字典创建配置"""
        return cls(**config_dict)
