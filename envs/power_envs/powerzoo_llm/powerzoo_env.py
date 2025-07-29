# -*- coding: utf-8 -*-
"""
@File      : powerzoo_env.py
@Time      : 2025-04-08 17:51
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此 Python 文件旨在实现一个电力系统仿真环境 `PowerZooEnv`，用于多智能体强化学习实验。
- 关键库：使用 `gym` 进行环境管理，`numpy` 进行数值计算，`imageio` 与 `matplotlib.pyplot` 用于可能的图像操作。
- 关键函数：
  - `seeding`：设置随机种子，保证结果可复现。
- 关键类：
  - `PowerZooEnv`：
    - 初始化时创建环境，确定智能体数量和名称，设置动作和观测空间。
    - `step`：执行动作，返回局部观测、全局状态、奖励、终止信息等。
    - `reset`：重置环境，返回初始观测和状态。
    - `get_avail_actions`：获取所有智能体可用动作。
    - `get_avail_agent_actions`：获取单个智能体可用动作。
    - `render`：预留渲染功能。
    - `close`：关闭环境，移除并行 DSS。
    - `seed`：设置环境随机种子。
    - `unwrap`：处理观测数据。
    - `get_env_action_space`：拆分动作空间给各智能体。
    - `repeat`：复制观测空间。
"""
import copy
import gym
from gym.spaces import Discrete, Box, MultiDiscrete
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
import torch
try:
    from envs.power_envs.powerzoo_llm.env_register import make_env, remove_parallel_dss
except ImportError:
    # 相对导入用于测试
    from .env_register import make_env, remove_parallel_dss
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache
import logging

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

try:
    from .utils import get_logger
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

logger = get_logger(__name__)


def seeding(seed: int) -> None:
    """优化的随机种子设置"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class PowerZooEnv:
    """PowerZoo环境类（集成优化功能）"""
    
    def __init__(self, env, config, rank: Optional[int] = None):
        """
        初始化优化环境
        
        Args:
            env: 底层环境实例
            config: 环境配置
            rank: 进程编号，用于多进程训练
        """
        self.env = env
        self.config = config
        self.rank = rank
        
        # 核心配置缓存
        self._setup_core_config()
        
        # 智能体配置
        self._setup_agents()
        
        # 动作和观测空间
        self._setup_spaces()
        
        # 状态缓存（优化功能）
        self._state_cache = {}
        self._last_obs = None
        self._last_actions = None
        
        logger.info(f"PowerZoo环境初始化完成 - 智能体数: {self.n_agents}, 环境数: {self.num_env}")
    
    def _setup_core_config(self) -> None:
        """设置核心配置"""
        self.num_env = getattr(self.config, 'num_env', 1)
        self.env_name = getattr(self.config, 'env_name', 'default')
        self.use_s = getattr(self.config, 'useS', False)
        
        # 设置种子
        seed = getattr(self.config, 'seed', 0)
        self.env.seed(seed)
        seeding(seed)
        
        # 设置环境使用敏感性矩阵
        self.env.useS = self.use_s
    
    def _setup_agents(self) -> None:
        """设置智能体配置"""
        # 缓存设备信息以减少重复访问
        self.cap_num = self.env.cap_num
        self.reg_num = self.env.reg_num
        self.bat_num = self.env.bat_num
        
        # 智能体总数
        self.n_agents = self.cap_num + self.reg_num + self.bat_num
        self.agents = list(range(self.n_agents))
        
        # 设备名称缓存
        self.cap_names = self.env.cap_names.copy()
        self.reg_names = self.env.reg_names.copy()
        self.bat_names = self.env.bat_names.copy()
        
        # 智能体名称映射
        self._agent_names = self.cap_names + self.reg_names + self.bat_names
        
        # 智能体-设备映射（如果使用敏感性矩阵）
        if self.use_s:
            self._setup_sensitivity_mapping()
    
    def _setup_sensitivity_mapping(self) -> None:
        """设置敏感性矩阵相关映射"""
        update_orders = list(range(self.n_agents))
        self.ordered_agents_pairs = dict(zip(self._agent_names, update_orders))
        self.agents_bus = self.env.agents_bus
    
    def _setup_spaces(self) -> None:
        """设置动作和观测空间"""
        # 观测空间
        self.share_observation_space = self._repeat_space(self.env.observation_space)
        self.observation_space = self._unwrap_space(self.env.observation_space)
        
        # 动作空间 - 优化动作空间分解
        self.action_space = self._get_env_action_space(self.env.action_space)
        
        # 可用动作缓存
        self._avail_actions_cache = None
        
        # 确定是否为离散动作空间
        self.discrete = not isinstance(self.env.action_space, Box)
        
    
    @lru_cache(maxsize=1)
    def get_avail_actions(self) -> List[List[int]]:
        """获取可用动作（带缓存）"""
        if self._avail_actions_cache is None:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self._get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            self._avail_actions_cache = avail_actions
        
        return self._avail_actions_cache
    
    def _get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """获取单个智能体的可用动作"""
        return [1] * self.action_space[agent_id].n
        
    def step(self, actions: np.ndarray) -> Tuple[List, List, List[List[float]], List[bool], List[Dict], List]:
        """
        优化的环境步进
        
        Args:
            actions: 动作数组，形状为 (env_num, n_agents, act_dim) 或 (n_agents, act_dim)
            
        Returns:
            tuple: (local_obs, global_state, rewards, dones, infos, available_actions)
        """
        # 动作预处理
        processed_actions = self._preprocess_actions(actions)
        
        # 执行步进
        try:
            if self.discrete:
                obs, rew, done, info = self.env.step(processed_actions.flatten())
            else:
                obs, rew, done, info = self.env.step(processed_actions[0])
        except Exception as e:
            logger.error(f"环境步进失败: {e}")
            # 返回安全的默认值
            return self._get_safe_step_result()
        
        # 处理终止信息
        if done and "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
            info["bad_transition"] = True
        
        # 处理观测
        wrapped_obs = self._unwrap_space_data(obs)
        
        # 更新缓存
        self._last_obs = wrapped_obs
        self._last_actions = actions
        
        # 为兼容性处理dones
        dones = [done] * self.n_agents if self.discrete else [[done]] * self.n_agents
        
        return (
            wrapped_obs,  # local_obs  
            wrapped_obs,  # global_state 
            [[rew]],      # rewards
            dones,        # dones
            [info],       # infos
            self.get_avail_actions()  # available_actions
        )

    def reset(self) -> Tuple[List, List, List]:
        """
        优化的环境重置
        
        Returns:
            tuple: (obs, state, available_actions)
        """
        try:
            # 清理缓存
            self._clear_caches()
            
            # 确定加载配置文件索引
            load_profile_idx = self.rank if self.rank is not None else 0
            
            # 重置环境 - 移除不必要的参数
            obs = self.env.reset(load_profile_idx=load_profile_idx)
            
            # 处理观测
            wrapped_obs = self._unwrap_space_data(obs)
            state_obs = copy.deepcopy(wrapped_obs)  # 只在必要时进行深拷贝
            
            # 更新缓存
            self._last_obs = wrapped_obs
            
            return wrapped_obs, state_obs, self.get_avail_actions()
            
        except Exception as e:
            logger.error(f"环境重置失败: {e}")
            # 返回安全的默认值
            return self._get_safe_reset_result()



    # 重复的方法已在上面定义
        

    def render(self, mode: str = 'human') -> None:
        """渲染环境（预留接口）"""
        # 可以在这里添加可视化逻辑
        pass

    def close(self) -> None:
        """关闭环境"""
        try:
            remove_parallel_dss(self.env_name, self.rank)
            logger.info("环境已关闭")
        except Exception as e:
            logger.error(f"关闭环境时出错: {e}")

    def seed(self, seed: int) -> None:
        """设置环境种子"""
        self.env.seed(seed)
        seeding(seed)
        
    def _unwrap_space_data(self, data: Any) -> List:
        """展开空间数据为智能体列表"""
        return [data for _ in range(self.n_agents)]
    
    def unwrap(self, data: Any) -> List:
        """兼容原unwrap方法"""
        return self._unwrap_space_data(data)
    
    def _preprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        """预处理动作数组"""
        if isinstance(actions, (list, tuple)):
            actions = np.array(actions)
        
        # 确保动作形状正确
        if actions.ndim == 3:  # (env_num, n_agents, act_dim)
            actions = actions[0]  # 取第一个环境的动作
        elif actions.ndim == 1 and len(actions) == self.n_agents:
            # 已经是正确形状
            pass
        else:
            logger.warning(f"动作形状异常: {actions.shape}, 尝试重塑")
            actions = actions.reshape(-1)[:self.n_agents]
        
        return actions
    
    def _clear_caches(self) -> None:
        """清理缓存"""
        self._state_cache.clear()
        self._avail_actions_cache = None
        if hasattr(self, 'get_avail_actions'):
            self.get_avail_actions.cache_clear()
    
    def _get_safe_step_result(self) -> Tuple:
        """获取安全的步进结果（错误时使用）"""
        safe_obs = [np.zeros(100) for _ in range(self.n_agents)]  # 假设观测维度
        return (
            safe_obs,
            safe_obs,  
            [[0.0]],
            [True],
            [{"error": True}],
            self.get_avail_actions()
        )
    
    def _get_safe_reset_result(self) -> Tuple:
        """获取安全的重置结果（错误时使用）"""
        safe_obs = [np.zeros(100) for _ in range(self.n_agents)]
        return safe_obs, safe_obs, self.get_avail_actions()
    
    def _get_env_action_space(self, env_action_space) -> List[Discrete]:
        """分解环境动作空间给各智能体"""
        if hasattr(env_action_space, 'nvec'):
            return [Discrete(n) for n in env_action_space.nvec]
        else:
            # 简化处理：假设所有智能体动作空间相同
            default_action_dim = 2  # 默认二元动作
            return [Discrete(default_action_dim) for _ in range(self.n_agents)]
    
    def _repeat_space(self, space) -> List:
        """复制空间给每个智能体"""
        return [space for _ in range(self.n_agents)]
    
    def _unwrap_space(self, space) -> List:
        """展开空间为智能体列表"""
        return [space for _ in range(self.n_agents)]
    
    def get_env_action_space(self, env_action_space) -> List[Discrete]:
        """兼容原版动作空间方法"""
        return self._get_env_action_space(env_action_space)
    
    def repeat(self, data: Any) -> List:
        """兼容原版repeat方法"""
        return [data for _ in range(self.n_agents)]
    
    # 属性访问优化
    @property
    def agents_bus(self) -> Optional[Dict]:
        """智能体总线映射"""
        return getattr(self, '_agents_bus', None)
    
    @property
    def ordered_agents_pairs(self) -> Optional[Dict]:
        """有序智能体对映射"""
        return getattr(self, '_ordered_agents_pairs', None)


# === 向后兼容性别名 ===
# 保持与powerzoo_env_optimized.py的兼容性
OptimizedPowerZooEnv = PowerZooEnv




