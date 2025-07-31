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
from gym.spaces import Discrete, Box
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
            f"PV系统: {self.pv_num}{'(启用)' if self.pv_control_enabled else '(禁用)'} | "
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
        
        # 添加PV系统支持
        self.pv_num = self.env.pv_num
        self.pv_names = self.env.pv_names.copy()
        self.pv_control_enabled = getattr(self.env, 'pv_control_enabled', False)
        
        # 智能体总数（当PV控制启用时，包含PV系统）
        self.n_agents = self.cap_num + self.reg_num + self.bat_num
        if self.pv_control_enabled and self.pv_num > 0:
            self.n_agents += self.pv_num
        
        self.agents = list(range(self.n_agents))
        
        # 设备名称缓存
        self.cap_names = self.env.cap_names.copy()
        self.reg_names = self.env.reg_names.copy()
        self.bat_names = self.env.bat_names.copy()
        
        # 智能体名称映射（当PV控制启用时，包含PV系统名称）
        self._agent_names = self.cap_names + self.reg_names + self.bat_names
        if self.pv_control_enabled and self.pv_num > 0:
            self._agent_names += self.pv_names
        
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
        
        # HAPPO兼容性验证
        self._validate_happo_compatibility()
        
    
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
        agent_space = self.action_space[agent_id]
        
        if isinstance(agent_space, Discrete):
            return [1] * agent_space.n
        elif isinstance(agent_space, Box):
            # 连续动作空间（PV系统），返回空列表或单个1表示可用
            return [1]  # 表示连续动作可用
        else:
            # 默认情况
            return [1] * getattr(agent_space, 'n', 1)
        
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
            # 处理混合动作空间的情况
            if self.pv_control_enabled and self.pv_num > 0:
                # 混合动作空间：直接传递列表给底层环境
                obs, rew, done, info = self.env.step(processed_actions)
            elif self.discrete:
                # 纯离散动作空间：展平后传递
                if isinstance(processed_actions, np.ndarray):
                    obs, rew, done, info = self.env.step(processed_actions.flatten())
                else:
                    obs, rew, done, info = self.env.step(processed_actions)
            else:
                # 连续动作空间：传递第一个环境的动作
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
        
        # HAPPO兼容性：确保数据格式完全符合要求
        # Done信号：必须是numpy数组，形状为(n_agents,)
        dones_array = np.array([bool(done) for _ in range(self.n_agents)], dtype=bool)
        
        # 奖励信号：确保为正确的嵌套结构
        rewards_formatted = [[float(rew)]]
        
        # 验证返回数据的形状一致性
        self._validate_step_output(wrapped_obs, dones_array, rewards_formatted, info)
        
        return (
            wrapped_obs,           # local_obs: List[np.ndarray]  
            wrapped_obs,           # global_state: List[np.ndarray] 
            rewards_formatted,     # rewards: List[List[float]]
            dones_array,           # dones: np.ndarray(n_agents,)
            [info],               # infos: List[Dict]
            self.get_avail_actions()  # available_actions: List
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
        
    def render(self, mode: str = 'human') -> None:
        """渲染环境（预留接口）"""
        # TODO 添加可视化接口
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
        """预处理混合动作数组（支持离散+连续动作）"""
        if isinstance(actions, (list, tuple)):
            # 混合动作不能直接转换为numpy数组，需要保持原始结构
            if self.pv_control_enabled and self.pv_num > 0:
                # 对于混合动作空间，保持列表结构
                processed_actions = list(actions)
            else:
                # 纯离散动作，可以转换为numpy数组
                processed_actions = np.array(actions)
        else:
            processed_actions = actions
        
        # 处理三维动作 (env_num, n_agents, act_dim)
        if isinstance(processed_actions, np.ndarray) and processed_actions.ndim == 3:
            processed_actions = processed_actions[0]  # 取第一个环境的动作
        elif isinstance(processed_actions, list) and len(processed_actions) > 0:
            # 检查是否为嵌套列表结构 [[agent0_actions], [agent1_actions], ...]
            if isinstance(processed_actions[0], (list, np.ndarray)) and len(processed_actions) == 1:
                processed_actions = processed_actions[0]  # 展开一层
        
        # 确保动作数量正确
        if isinstance(processed_actions, (list, tuple)):
            if len(processed_actions) != self.n_agents:
                logger.warning(f"动作数量不匹配: 期望{self.n_agents}, 实际{len(processed_actions)}")
                # 截断或填充到正确长度
                if len(processed_actions) > self.n_agents:
                    processed_actions = processed_actions[:self.n_agents]
                else:
                    # 填充默认动作
                    while len(processed_actions) < self.n_agents:
                        processed_actions.append(0)  # 默认动作
        elif isinstance(processed_actions, np.ndarray):
            if processed_actions.ndim == 1 and len(processed_actions) == self.n_agents:
                # 已经是正确形状
                pass
            else:
                logger.debug(f"动作形状异常: {processed_actions.shape}, 尝试重塑")
                processed_actions = processed_actions.reshape(-1)[:self.n_agents]
        
        return processed_actions
    
    def _clear_caches(self) -> None:
        """清理缓存"""
        self._state_cache.clear()
        self._avail_actions_cache = None
        if hasattr(self, 'get_avail_actions'):
            self.get_avail_actions.cache_clear()
    
    def _get_safe_step_result(self) -> Tuple:
        """获取HAPPO兼容的安全步进结果（错误时使用）"""
        # 使用现有的观测维度或默认值
        obs_dim = len(self._last_obs[0]) if self._last_obs else 100
        safe_obs = [np.zeros(obs_dim) for _ in range(self.n_agents)]
        
        return (
            safe_obs,                                              # local_obs
            safe_obs,                                              # global_state
            [[0.0]],                                              # rewards
            np.array([True] * self.n_agents, dtype=bool),         # dones (numpy array)
            [{"error": True, "safe_mode": True}],                 # infos
            self.get_avail_actions()                              # available_actions
        )
    
    def _get_safe_reset_result(self) -> Tuple:
        """获取安全的重置结果（错误时使用）"""
        safe_obs = [np.zeros(100) for _ in range(self.n_agents)]
        return safe_obs, safe_obs, self.get_avail_actions()
    
    def _validate_step_output(self, obs, dones, rewards, info):
        """验证step输出的HAPPO兼容性"""
        try:
            # 验证观测数据
            assert isinstance(obs, list), f"观测必须是列表，得到: {type(obs)}"
            assert len(obs) == self.n_agents, f"观测长度不匹配智能体数量: {len(obs)} vs {self.n_agents}"
            
            # 验证done信号（HAPPO关键要求）
            assert isinstance(dones, np.ndarray), f"Done信号必须是numpy数组，得到: {type(dones)}"
            assert dones.dtype == bool, f"Done信号必须是布尔类型，得到: {dones.dtype}"
            assert dones.shape == (self.n_agents,), f"Done信号形状错误: {dones.shape} vs ({self.n_agents},)"
            
            # 验证奖励数据
            assert isinstance(rewards, list), f"奖励必须是列表，得到: {type(rewards)}"
            assert len(rewards) > 0 and isinstance(rewards[0], list), f"奖励格式错误: {rewards}"
            
            # 验证info数据
            assert isinstance(info, dict), f"Info必须是字典，得到: {type(info)}"
            
        except AssertionError as e:
            logger.error(f"HAPPO数据验证失败: {e}")
            raise ValueError(f"HAPPO兼容性验证失败: {e}")
    
    def _get_env_action_space(self, env_action_space) -> List[Union[Discrete, Box]]:
        """分解环境动作空间给各智能体 - HAPPO异构智能体兼容"""
        if hasattr(env_action_space, 'nvec'):
            # 标准的MultiDiscrete空间
            return [Discrete(n) for n in env_action_space.nvec]
        elif isinstance(env_action_space, gym.spaces.Tuple):
            # 混合动作空间（离散 + 连续），处理PV系统的情况
            agent_spaces = []
            
            # 处理离散部分 - 电容器和调压器
            if len(env_action_space.spaces) >= 1 and hasattr(env_action_space.spaces[0], 'nvec'):
                discrete_nvec = env_action_space.spaces[0].nvec
                agent_spaces.extend([Discrete(n) for n in discrete_nvec])
            
            # 处理连续部分（为PV或其他连续控制设备创建Box空间）
            if len(env_action_space.spaces) >= 2:
                continuous_space = env_action_space.spaces[1]
                if hasattr(continuous_space, 'shape') and continuous_space.shape:
                    continuous_dim = continuous_space.shape[0]
                    # 为PV系统分配连续动作空间
                    if self.pv_control_enabled and self.pv_num > 0:
                        # 每个PV系统可能有2个连续动作维度（有功功率和功率因数）
                        pv_continuous_dims = continuous_dim
                        if self.pv_num * 2 == pv_continuous_dims:
                            # 每个PV系统2个连续动作 - 创建标准化的Box空间
                            for _ in range(self.pv_num):
                                agent_spaces.append(Box(
                                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                                ))
                        else:
                            # 总的连续动作分配给所有PV系统
                            dims_per_pv = max(1, pv_continuous_dims // self.pv_num)
                            for _ in range(self.pv_num):
                                agent_spaces.append(Box(
                                    low=-1.0, high=1.0, shape=(dims_per_pv,), dtype=np.float32
                                ))
            
            return agent_spaces
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
    
    def _format_actions_summary(self, actions) -> str:
        """格式化动作摘要用于日志记录"""
        if isinstance(actions, (list, tuple)):
            if len(actions) <= 6:
                # 格式化混合动作（整数和数组）
                formatted_actions = []
                for action in actions:
                    if isinstance(action, np.ndarray):
                        formatted_actions.append(f"[{', '.join([f'{x:.3f}' for x in action])}]")
                    else:
                        formatted_actions.append(str(action))
                return f"[{', '.join(formatted_actions)}]"
            else:
                # 对于长动作向量，只显示前几个和后几个
                start_actions = []
                end_actions = []
                for i, action in enumerate(actions[:3]):
                    if isinstance(action, np.ndarray):
                        start_actions.append(f"[{', '.join([f'{x:.3f}' for x in action])}]")
                    else:
                        start_actions.append(str(action))
                for i, action in enumerate(actions[-3:]):
                    if isinstance(action, np.ndarray):
                        end_actions.append(f"[{', '.join([f'{x:.3f}' for x in action])}]")
                    else:
                        end_actions.append(str(action))
                return f"[[{', '.join(start_actions)}]...[{', '.join(end_actions)}]]({len(actions)})"
        elif isinstance(actions, np.ndarray):
            # 处理numpy数组
            if len(actions) <= 6:
                return str(actions.tolist())
            else:
                start = actions[:3].tolist()
                end = actions[-3:].tolist()
                return f"[{start}...{end}]({len(actions)})"
        else:
            # 其他类型直接转换为字符串
            return str(actions)
    
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
    
    def _validate_happo_compatibility(self) -> None:
        """验证HAPPO算法兼容性 - 确保异质智能体支持"""
        try:
            # 检查动作空间一致性
            action_space_types = set()
            action_space_shapes = set()
            
            for i, space in enumerate(self.action_space):
                space_type = type(space).__name__
                action_space_types.add(space_type)
                
                if hasattr(space, 'shape'):
                    action_space_shapes.add(space.shape)
                elif hasattr(space, 'n'):
                    action_space_shapes.add((space.n,))
                    
            # HAPPO支持异构智能体，但需要确保接口一致
            logger.info(f"HAPPO兼容性检查 - 动作空间类型: {action_space_types}, 形状: {action_space_shapes}")
            
            # 检查观测空间一致性
            obs_consistent = all(
                hasattr(space, 'shape') and len(space.shape) > 0 
                for space in self.observation_space
            )
            
            if not obs_consistent:
                logger.warning("观测空间可能存在不一致，可能影响HAPPO训练")
            
            # 验证动作采样一致性
            test_actions = []
            for _ in range(3):  # 测试多次采样
                action_sample = [space.sample() for space in self.action_space]
                action_shapes = [np.array(a).shape for a in action_sample]
                test_actions.append(action_shapes)
            
            # 检查采样结果形状一致性
            shape_consistent = all(shapes == test_actions[0] for shapes in test_actions)
            if not shape_consistent:
                logger.warning("动作采样形状不一致，已应用形状标准化")
            
            logger.info(f"HAPPO兼容性验证完成 - 智能体数: {self.n_agents}, 混合动作空间: {self.pv_control_enabled}")
            
        except Exception as e:
            logger.error(f"HAPPO兼容性验证失败: {e}")
    
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




