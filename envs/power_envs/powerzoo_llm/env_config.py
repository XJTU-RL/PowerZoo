# -*- coding: utf-8 -*-
"""
PowerZoo环境配置类和多智能体包装器
"""
from typing import Optional, Any, Dict
from dataclasses import dataclass
import numpy as np

try:
    from envs.power_envs.powerzoo_llm.env_wrapper import PowerZooEnvWrapper
except ImportError:
    from .env_wrapper import PowerZooEnvWrapper


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


class PowerZooMultiAgentWrapper:
    """PowerZoo多智能体环境包装器"""
    
    def __init__(self, config: PowerZooConfig):
        """
        初始化多智能体环境包装器
        
        Args:
            config: PowerZoo配置对象
        """
        self.config = config
        
        # 创建基础环境
        self.env = PowerZooEnvWrapper.create_base_env(
            config=config,
            env_name=config.env_name,
            optimization_level="standard"
        )
        
        # 设置基本属性
        self.num_agents = config.num_agents
        self.n_agents = config.num_agents
        
        # 从底层环境获取空间信息
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.share_observation_space = self._create_share_obs_space()
        
        # 环境状态
        self._episode_step = 0
        self._max_episode_steps = config.max_episode_steps
    
    def _create_share_obs_space(self):
        """创建共享观测空间"""
        try:
            import gymnasium as gym
        except ImportError:
            import gym
        
        # 处理观测空间可能是列表的情况
        if isinstance(self.env.observation_space, list):
            obs_dim = self.env.observation_space[0].shape[0]
        else:
            obs_dim = self.env.observation_space.shape[0]
        
        # 返回单一的共享观测空间，包含所有智能体的观测
        return gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(obs_dim * self.num_agents,),
            dtype='float32'
        )
        
    def reset(self):
        """重置环境"""
        self._episode_step = 0
        # PowerZooEnv返回3个值：obs, state, available_actions
        obs, state, available_actions = self.env.reset()
        # 训练器期望2个值：obs, share_obs
        return obs, state
    
    def step(self, actions):
        """执行一步"""
        self._episode_step += 1
        
        # 执行动作 - PowerZooEnv返回6个值
        local_obs, global_state, rewards, dones, infos, available_actions = self.env.step(actions)
        
        # 检查是否达到最大步数
        if self._episode_step >= self._max_episode_steps:
            dones = [True] * self.num_agents
        
        # 返回训练器期望的5个值：next_obs, next_share_obs, rewards, dones, infos
        return local_obs, global_state, rewards, dones, infos
    
    def render(self, mode='human'):
        """渲染环境"""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
    
    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def seed(self, seed=None):
        """设置随机种子"""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
    
    def get_avail_actions(self):
        """获取所有智能体可用动作"""
        if hasattr(self.env, 'get_avail_actions'):
            return self.env.get_avail_actions()
        else:
            # 默认所有动作都可用
            if hasattr(self.action_space, '__len__'):
                return [np.ones(space.n) for space in self.action_space]
            else:
                return [np.ones(self.action_space.n) for _ in range(self.num_agents)]