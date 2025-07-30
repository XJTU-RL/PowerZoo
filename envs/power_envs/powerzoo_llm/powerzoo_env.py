# -*- coding: utf-8 -*-
"""
@File      : powerzoo_env.py
@Time      : 2025-04-08 17:51
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
"""
import copy
import gym
from gym.spaces import Discrete, Box, MultiDiscrete
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
import torch
import time
try:
    from envs.power_envs.powerzoo_llm.env_register import make_base_env, remove_parallel_dss
except ImportError:
    # 相对导入用于测试
    from .env_register import make_base_env, remove_parallel_dss
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache
import logging

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

try:
    from .utils import get_logger, log_training_step, setup_training_logger
    from .system_logger import get_system_logger, SystemLogger
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    
    def log_training_step(*args, **kwargs):
        pass
    
    def setup_training_logger(name):
        return get_logger(name)
    
    # 系统记录器备用实现
    class SystemLogger:
        def __init__(self, **kwargs):
            pass
        def log_system_state(self, *args, **kwargs):
            pass
        def close(self):
            pass
    
    def get_system_logger(**kwargs):
        return SystemLogger()

logger = get_logger(__name__)

# 添加训练日志记录器（如果需要详细的训练日志）
_training_logger = None

def get_training_logger():
    """获取训练专用日志记录器"""
    global _training_logger
    if _training_logger is None:
        _training_logger = setup_training_logger("powerzoo_training")
    return _training_logger


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
        
        # 训练统计信息
        self._episode_count = 0
        self._total_steps = 0
        self._training_logger = get_training_logger()
        
        # 初始化系统参数记录器
        self._enable_system_logging = getattr(config, 'enable_system_logging', True)
        if self._enable_system_logging:
            log_dir = getattr(config, 'system_log_dir', f"./logs/system_params/rank_{rank if rank is not None else 0}")
            self._system_logger = get_system_logger(
                log_dir=log_dir,
                buffer_size=getattr(config, 'log_buffer_size', 5000),
                save_interval=getattr(config, 'log_save_interval', 50),
                enable_realtime_log=getattr(config, 'enable_realtime_log', True)
            )
        else:
            self._system_logger = None
        
        logger.info(f"PowerZoo环境初始化完成 - 智能体数: {self.n_agents}, 环境数: {self.num_env}")
        self._training_logger.train_info(
            f"环境初始化 | 智能体数: {self.n_agents} | 环境数: {self.num_env} | "
            f"电容器: {self.cap_num} | 调压器: {self.reg_num} | 电池: {self.bat_num} | "
            f"系统记录: {'启用' if self._enable_system_logging else '禁用'}"
        )
    
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
        
        # 执行步进（记录计算时间）
        step_start_time = time.time()
        try:
            if self.discrete:
                obs, rew, done, info = self.env.step(processed_actions.flatten())
            else:
                obs, rew, done, info = self.env.step(processed_actions[0])
            step_computation_time = time.time() - step_start_time
        except Exception as e:
            step_computation_time = time.time() - step_start_time
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
        
        # 更新训练统计并记录步骤信息
        self._total_steps += 1
        current_step = getattr(self.env, 't', 0)
        
        # 记录详细的训练步骤信息
        action_summary = self._format_actions_summary(processed_actions)
        simplified_info = {
            'reward': rew,
            'power_loss': info.get('power_loss_ratio', 0),
            'voltage_reward': info.get('vol_reward', 0),
            'ctrl_reward': info.get('ctrl_reward', 0)
        }
        
        self._training_logger.train_info(
            f"Episode {self._episode_count:4d} | Step {current_step:3d} | "
            f"Actions: {action_summary} | Reward: {rew:8.4f} | Done: {done}"
        )
        
        # 记录奖励分解信息（如果可用）
        if 'vol_reward' in info and 'ctrl_reward' in info:
            reward_components = {
                'total': rew,
                'voltage': info.get('vol_reward', 0),
                'control': info.get('ctrl_reward', 0),
                'power_loss': info.get('power_loss_ratio', 0)
            }
            reward_str = " | ".join([f"{k}: {v:6.3f}" for k, v in reward_components.items()])
            self._training_logger.reward_debug(f"Reward Components: {reward_str}")
        
        # 系统参数记录
        if self._enable_system_logging and self._system_logger:
            try:
                # 增强info字典，添加系统状态信息
                enhanced_info = info.copy()
                enhanced_info.update({
                    'dss_convergence': getattr(self.env, 'converged', True),
                    'active_powers': getattr(self.env, 'active_powers', {}),
                    'reactive_powers': getattr(self.env, 'reactive_powers', {}),
                    'load_distribution': getattr(self.env, 'load_distribution', {}),
                    'voltage_violations': self._calculate_voltage_violations(),
                })
                
                # 记录系统状态
                self._system_logger.log_system_state(
                    env=self.env,
                    actions=processed_actions,
                    reward=rew,
                    info=enhanced_info,
                    computation_time=step_computation_time
                )
            except Exception as e:
                logger.debug(f"系统参数记录失败: {e}")
        
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
            
            # 更新回合统计
            self._episode_count += 1
            
            # 记录环境重置信息
            self._training_logger.train_info(
                f"环境重置 | Episode {self._episode_count:4d} | "
                f"LoadProfile: {load_profile_idx} | 总步数: {self._total_steps}"
            )
            
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
            # 关闭系统记录器
            if self._enable_system_logging and self._system_logger:
                self._system_logger.close()
                logger.info("系统记录器已关闭")
            
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
            logger.debug(f"动作形状异常: {actions.shape}, 尝试重塑")
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
    
    def _format_actions_summary(self, actions: np.ndarray) -> str:
        """格式化动作摘要用于日志记录"""
        if len(actions) <= 6:
            return str(actions.tolist())
        else:
            # 对于长动作向量，只显示前几个和后几个
            start = actions[:3].tolist()
            end = actions[-3:].tolist()
            return f"[{start}...{end}]({len(actions)})"
    
    def _calculate_voltage_violations(self) -> List[str]:
        """计算电压违规"""
        violations = []
        voltage_limits = {'min': 0.95, 'max': 1.05}
        
        try:
            if hasattr(self.env, 'obs') and 'bus_voltages' in self.env.obs:
                for bus_name, voltages in self.env.obs['bus_voltages'].items():
                    if isinstance(voltages, (list, np.ndarray)):
                        for i, v in enumerate(voltages):
                            if v < voltage_limits['min'] or v > voltage_limits['max']:
                                violations.append(f"{bus_name}_{i}")
                    else:
                        if voltages < voltage_limits['min'] or voltages > voltage_limits['max']:
                            violations.append(bus_name)
        except Exception as e:
            logger.debug(f"计算电压违规时出错: {e}")
        
        return violations
    
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




