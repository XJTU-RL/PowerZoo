# -*- coding: utf-8 -*-
"""
@File      : vvc_env.py
@Time      : 2025-04-08 17:51
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
"""
import copy
try:
    import gymnasium as gym
    from gymnasium.spaces import Discrete, Box, MultiDiscrete
except ImportError:
    import gym
    from gym.spaces import Discrete, Box, MultiDiscrete
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
try:
    from envs.smartgrid.base_env.env_register import make_base_env, remove_parallel_dss
except ImportError:
    # 相对导入用于测试
    from .env_register import make_base_env, remove_parallel_dss
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache
import logging

import argparse
import random
import itertools
import sys
import os

# 使用统一日志系统
from envs.smartgrid.logging import (
	get_logger,
	log_training_step,
	setup_training_logger
)
# SystemLogger已被移除，功能已整合到smartgrid_logger

logger = get_logger(__name__)

# 添加训练日志记录器（如果需要详细的训练日志）
_training_logger = None

def get_training_logger() -> logging.Logger:
    """获取训练专用日志记录器"""
    global _training_logger
    if _training_logger is None:
        _training_logger = setup_training_logger("vvc_training")
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


class VVCEnv:
    """VVC环境类（集成优化功能）

    支持两种 config 参数类型:
    1. SmartGridConfig 对象（推荐，新代码路径）
    2. dict（legacy env_args，向后兼容）
    """

    def __init__(self, env, config_or_args, rank: Optional[int] = None):
        """
        初始化优化环境

        Args:
            env: 底层环境实例
            config_or_args: SmartGridConfig 对象或 legacy env_args 字典
            rank: 进程编号，用于多进程训练
        """
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        # 统一 config 处理：dict -> SmartGridConfig adapter
        if isinstance(config_or_args, SmartGridConfig):
            self.config = config_or_args
        elif isinstance(config_or_args, dict):
            # Legacy dict 路径：包装为 SmartGridConfig
            self.config = SmartGridConfig.from_env_args(config_or_args)
        else:
            # 兜底：假设是对象，直接使用
            self.config = config_or_args

        self.env = env
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
        # NOTE: get_system_logger 已被移除，系统日志功能由 smartgrid_logger 统一管理
        self._enable_system_logging = getattr(self.config, 'enable_system_logging', False)
        self._system_logger = None  # 系统日志功能已整合到统一日志系统
        
        logger.info(f"VVC环境初始化完成 - 智能体数: {self.n_agents}, 环境数: {self.num_env}")
        self._training_logger.info(
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

        # 缓存动作数量信息（用于默认动作生成等）
        self.bat_act_num = self.env.bat_act_num
        self.reg_act_num = self.env.reg_act_num

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
        # 修复属性名称不匹配问题：设置带下划线的私有属性
        self._ordered_agents_pairs = dict(zip(self._agent_names, update_orders))
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
    
    def _get_avail_agent_actions(self, agent_id: int) -> Optional[List[int]]:
        """获取单个智能体的可用动作

        Returns:
            离散动作空间: 返回可用动作掩码列表 [1, 1, ..., 1]
            连续动作空间: 返回 None（连续动作不需要掩码）
        """
        agent_space = self.action_space[agent_id]

        if isinstance(agent_space, Discrete):
            return [1] * agent_space.n
        elif isinstance(agent_space, Box):
            # 连续动作空间（PV系统）不需要可用动作掩码
            return None
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
            # 底层env.step()始终期望一个扁平的numpy数组
            obs, rew, done, info = self.env.step(processed_actions)
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
        
        self._training_logger.info(
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
            # 使用debug方法而不是reward_debug
            if hasattr(self._training_logger, 'reward_debug'):
                self._training_logger.reward_debug(f"Reward Components: {reward_str}")
            elif hasattr(self._training_logger, 'debug'):
                self._training_logger.debug(f"Reward Components: {reward_str}")
            else:
                # 如果都没有，就打印出来
                print(f"[DEBUG] Reward Components: {reward_str}")
        
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
        
        # 奖励信号：HAPPO要求形状为(n_agents, 1)的numpy数组
        # 所有智能体共享相同的团队奖励（标准MARL协作设置）
        rewards_formatted = np.array([[float(rew)] for _ in range(self.n_agents)], dtype=np.float32)
        
        # 验证返回数据的形状一致性
        self._validate_step_output(wrapped_obs, dones_array, rewards_formatted, info)
        
        return (
            wrapped_obs,           # local_obs: List[np.ndarray]  
            wrapped_obs,           # global_state: List[np.ndarray] 
            rewards_formatted,     # rewards: np.ndarray(n_agents, 1)
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
            self._training_logger.info(
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
    
    def _preprocess_actions(self, actions) -> np.ndarray:
        """
        预处理动作，确保输出为底层环境期望的扁平numpy数组格式
        
        处理多种动作输入格式：
        1. 3D numpy数组: (env_num, n_agents, act_dim) 
        2. 2D numpy数组: (n_agents, act_dim) - HAPPO标准格式
        3. 按智能体分解的列表: [agent0_action, agent1_action, ...]
        4. 混合动作格式: [discrete_actions, continuous_actions]
        
        Args:
            actions: 输入动作，可能的格式多样
            
        Returns:
            扁平的numpy数组，供底层env.step()使用
        """
        
        # 步骤1: 输入格式标准化
        if isinstance(actions, np.ndarray):
            if actions.ndim == 3:
                # (env_num, n_agents, act_dim) -> 取第一个环境
                actions = actions[0]
            # 如果已经是1D数组且长度正确，直接返回
            if actions.ndim == 1:
                expected_len = (self.cap_num + self.reg_num + self.bat_num + 
                              (self.pv_num * 2 if self.pv_control_enabled else 0))
                if len(actions) == expected_len:
                    return actions
        elif isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], (list, np.ndarray)):
            # [[agent_actions]] -> [agent_actions] 
            actions = actions[0]
        
        # 步骤2: 转换为扁平数组
        if self.pv_control_enabled and self.pv_num > 0:
            return self._process_mixed_actions_to_flat(actions)
        else:
            # 纯离散动作空间
            return self._process_discrete_actions_to_flat(actions)
    
    def _process_mixed_actions_to_flat(self, actions) -> np.ndarray:
        """
        将混合动作转换为扁平numpy数组
        
        底层env.step()期望的格式：
        - 离散设备动作在前（电容器、调压器、电池）
        - 连续PV动作在后（如果启用）
        """
        flat_actions = []
        
        # 情况1: 动作已按智能体分解
        if isinstance(actions, (list, np.ndarray)) and len(actions) == self.n_agents:
            if isinstance(actions, np.ndarray) and actions.ndim == 2:
                # 2D数组 (n_agents, act_dim)
                discrete_count = self.cap_num + self.reg_num + self.bat_num
                
                # 提取离散动作
                for i in range(discrete_count):
                    flat_actions.append(int(actions[i].flatten()[0]))
                
                # 提取连续PV动作
                for i in range(self.pv_num):
                    agent_idx = discrete_count + i
                    if agent_idx < len(actions):
                        pv_action = actions[agent_idx].flatten()
                        # 每个PV有2个动作：有功功率和功率因数
                        if len(pv_action) >= 2:
                            flat_actions.extend(pv_action[:2])
                        else:
                            flat_actions.extend([pv_action[0], 1.0])  # 默认功率因数1.0
            else:
                # 列表格式或1D数组
                discrete_count = self.cap_num + self.reg_num + self.bat_num
                
                # 提取离散动作
                for i in range(discrete_count):
                    if i < len(actions):
                        flat_actions.append(int(actions[i]))
                    else:
                        flat_actions.append(0)
                
                # 提取连续PV动作
                for i in range(discrete_count, len(actions)):
                    pv_action = actions[i]
                    if isinstance(pv_action, (list, np.ndarray)):
                        flat_actions.extend(pv_action[:2])
                    else:
                        flat_actions.extend([float(pv_action), 1.0])
        
        # 情况2: 分组格式 [离散动作, 连续动作]
        elif isinstance(actions, (list, tuple)) and len(actions) == 2:
            discrete_actions, continuous_actions = actions
            
            # 添加离散动作
            discrete_actions = np.asarray(discrete_actions).flatten()
            expected_discrete = self.cap_num + self.reg_num + self.bat_num
            for i in range(expected_discrete):
                if i < len(discrete_actions):
                    flat_actions.append(int(discrete_actions[i]))
                else:
                    flat_actions.append(0)
            
            # 添加连续动作
            continuous_actions = np.asarray(continuous_actions).flatten()
            flat_actions.extend(continuous_actions)
        
        else:
            # 未知格式，返回默认动作
            logger.warning(f"未知动作格式: {type(actions)}")
            flat_actions = self._get_default_flat_actions()
        
        return np.array(flat_actions)
    
    def _process_discrete_actions_to_flat(self, actions) -> np.ndarray:
        """
        将纯离散动作转换为扁平numpy数组
        """
        if isinstance(actions, np.ndarray):
            # 如果是2D数组，扁平化
            if actions.ndim == 2:
                flat_actions = []
                for i in range(min(len(actions), self.n_agents)):
                    flat_actions.append(int(actions[i].flatten()[0]))
                # 填充剩余的
                while len(flat_actions) < self.n_agents:
                    flat_actions.append(0)
                return np.array(flat_actions)
            else:
                # 1D数组，直接返回
                return actions.flatten()[:self.n_agents].astype(int)
        elif isinstance(actions, (list, tuple)):
            # 列表格式
            flat_actions = []
            for i in range(self.n_agents):
                if i < len(actions):
                    flat_actions.append(int(actions[i]))
                else:
                    flat_actions.append(0)
            return np.array(flat_actions)
        else:
            logger.warning(f"未知离散动作格式: {type(actions)}")
            return np.zeros(self.n_agents, dtype=int)
    
    def _get_default_flat_actions(self) -> List[float]:
        """获取默认扁平动作列表"""
        flat_actions = []
        
        # 电容器默认动作
        flat_actions.extend([0] * self.cap_num)
        
        # 调压器默认动作
        flat_actions.extend([0] * self.reg_num)
        
        # 电池默认动作
        if self.bat_num > 0:
            if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num == float('inf'):
                # 连续电池控制
                flat_actions.extend([0.0] * self.bat_num)
            else:
                # 离散电池控制
                flat_actions.extend([0] * self.bat_num)
        
        # PV默认动作（如果启用）
        if self.pv_control_enabled and self.pv_num > 0:
            # 每个PV系统2个动作：有功功率0.0，功率因数1.0
            for _ in range(self.pv_num):
                flat_actions.extend([0.0, 1.0])
        
        return flat_actions

    def _process_mixed_actions(self, actions) -> List:
        """
        处理混合动作空间（离散 + 连续）
        
        支持的输入格式：
        1. 已分解的智能体动作列表: [agent0_action, agent1_action, ...]
        2. 2D numpy数组: (n_agents, act_dim) 
        3. 分组格式: [discrete_actions, continuous_actions]
        """
        
        # 格式1: 动作已按智能体分解 (修复核心bug: 支持numpy.ndarray)
        if isinstance(actions, (list, np.ndarray)) and len(actions) == self.n_agents:
            # 对于numpy数组，需要转换为列表格式供混合动作空间使用
            if isinstance(actions, np.ndarray):
                # 分解2D数组为智能体级动作
                agent_actions = []
                discrete_count = self.cap_num + self.reg_num + self.bat_num
                
                # 处理离散设备动作
                for i in range(discrete_count):
                    if actions.ndim == 2:
                        # 二维数组：取每个智能体的动作
                        agent_actions.append(int(actions[i].flatten()[0]))
                    else:
                        # 一维数组：直接取值
                        agent_actions.append(int(actions[i]))
                
                # 处理连续PV系统动作
                for i in range(self.pv_num):
                    agent_idx = discrete_count + i
                    if agent_idx < len(actions):
                        if actions.ndim == 2:
                            # 二维数组：每个PV智能体可能有多维动作
                            pv_action = actions[agent_idx].flatten()
                            # 确保PV动作为2维（有功功率 + 功率因数）
                            if len(pv_action) >= 2:
                                agent_actions.append(pv_action[:2])
                            else:
                                agent_actions.append(np.array([pv_action[0], 0.0]))
                        else:
                            # 一维数组：假设是标量，扩展为2维
                            action_val = actions[agent_idx]
                            if isinstance(action_val, np.ndarray):
                                if len(action_val) >= 2:
                                    agent_actions.append(action_val[:2])
                                else:
                                    agent_actions.append(np.array([action_val[0], 0.0]))
                            else:
                                agent_actions.append(np.array([float(action_val), 0.0]))
                    else:
                        # 默认PV动作
                        agent_actions.append(np.array([0.0, 0.0]))
                
                return agent_actions
            else:
                # 列表格式，直接返回
                return actions
        
        # 格式2: 分组格式 [离散动作, 连续动作]
        elif isinstance(actions, (list, tuple)) and len(actions) == 2:
            discrete_actions, continuous_actions = actions
            agent_actions = []
            
            # 处理离散设备（电容器、调压器、电池）
            discrete_actions = np.asarray(discrete_actions).flatten()
            for i in range(self.cap_num + self.reg_num + self.bat_num):
                if i < len(discrete_actions):
                    agent_actions.append(int(discrete_actions[i]))
                else:
                    agent_actions.append(0)  # 默认动作
            
            # 处理连续设备（PV系统）
            continuous_actions = np.asarray(continuous_actions).flatten()
            for i in range(self.pv_num):
                # 每个PV系统2个连续动作（有功功率 + 功率因数）
                start_idx = i * 2
                end_idx = start_idx + 2
                if end_idx <= len(continuous_actions):
                    pv_action = continuous_actions[start_idx:end_idx]
                    agent_actions.append(pv_action)
                else:
                    # 不够的情况，用默认值填充
                    agent_actions.append(np.array([0.0, 0.0]))
            
            return agent_actions
        
        # 格式3: 其他异常格式，记录并返回默认动作
        else:
            logger.warning(f"混合动作格式异常: type={type(actions)}, "
                         f"length={len(actions) if hasattr(actions, '__len__') else 'N/A'}, "
                         f"expected_agents={self.n_agents}")
            logger.debug(f"动作内容: {actions}")
            return self._get_default_actions()
    
    def _process_discrete_actions(self, actions) -> np.ndarray:
        """
        处理纯离散动作空间
        
        支持的输入格式：
        1. numpy数组: 直接展平并截取
        2. 列表/元组: 转换为numpy数组
        """
        if isinstance(actions, np.ndarray):
            # numpy数组：展平并截取到智能体数量
            return actions.flatten()[:self.n_agents]
        elif isinstance(actions, (list, tuple)):
            # 列表/元组：转换为数组格式
            actions_array = np.asarray(actions).flatten()
            return actions_array[:self.n_agents]
        else:
            # 异常格式：记录错误并返回默认动作
            logger.warning(f"纯离散动作格式异常: type={type(actions)}")
            default_actions = np.zeros(self.n_agents, dtype=int)
            return default_actions
    
    def _get_default_actions(self) -> List[Union[int, np.ndarray]]:
        """获取默认动作（错误恢复用）"""
        default_actions: List[Union[int, np.ndarray]] = []

        # 电容器和调压器默认动作（离散）
        for i in range(self.cap_num + self.reg_num + self.bat_num):
            default_actions.append(0)

        # PV系统默认动作（连续）
        if self.pv_control_enabled:
            for i in range(self.pv_num):
                default_actions.append(np.array([0.0, 0.0]))  # [有功功率, 功率因数]

        return default_actions
    
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
            safe_obs,                                             # local_obs
            safe_obs,                                             # global_state
            np.zeros((self.n_agents, 1)),                          # rewards
            np.array([True] * self.n_agents, dtype=bool),         # dones (numpy array)
            [{"error": True, "safe_mode": True}],                 # infos
            self.get_avail_actions()                              # available_actions
        )
    
    def _get_safe_reset_result(self) -> Tuple:
        """获取安全的重置结果（错误时使用）"""
        safe_obs = [np.zeros(100) for _ in range(self.n_agents)]
        return safe_obs, safe_obs, self.get_avail_actions()
    
    def _validate_step_output(self, obs: List, dones: np.ndarray, rewards: np.ndarray, info: Dict) -> None:
        """验证step输出的HAPPO兼容性"""
        try:
            # 验证观测数据
            assert isinstance(obs, list), f"观测必须是列表，得到: {type(obs)}"
            assert len(obs) == self.n_agents, f"观测长度不匹配智能体数量: {len(obs)} vs {self.n_agents}"
            
            # 验证done信号（HAPPO关键要求）
            assert isinstance(dones, np.ndarray), f"Done信号必须是numpy数组，得到: {type(dones)}"
            assert dones.dtype == bool, f"Done信号必须是布尔类型，得到: {dones.dtype}"
            assert dones.shape == (self.n_agents,), f"Done信号形状错误: {dones.shape} vs ({self.n_agents},)"
            
            # 验证奖励数据 - 修复: 奖励现在是numpy数组，形状为(n_agents, 1)
            assert isinstance(rewards, np.ndarray), f"奖励必须是numpy数组，得到: {type(rewards)}"
            assert rewards.shape == (self.n_agents, 1), f"奖励形状错误: {rewards.shape} vs ({self.n_agents}, 1)"
            
            # 验证info数据
            assert isinstance(info, dict), f"Info必须是字典，得到: {type(info)}"
            
        except AssertionError as e:
            logger.error(f"HAPPO数据验证失败: {e}")
            raise ValueError(f"HAPPO兼容性验证失败: {e}")
    
    def _get_env_action_space(self, env_action_space) -> List[Union[Discrete, Box]]:
        """分解环境动作空间给各智能体 - HAPPO异构智能体兼容"""
        agent_spaces = []
        
        if hasattr(env_action_space, 'nvec'):
            # 纯离散动作空间
            return [Discrete(n) for n in env_action_space.nvec]
        elif isinstance(env_action_space, gym.spaces.Tuple):
            # 混合动作空间（离散 + 连续）
            discrete_space = env_action_space.spaces[0]
            continuous_space = env_action_space.spaces[1] if len(env_action_space.spaces) > 1 else None
            
            # 为离散设备创建离散动作空间
            if hasattr(discrete_space, 'nvec'):
                for n in discrete_space.nvec:
                    agent_spaces.append(Discrete(n))
            
            # 为连续设备（PV系统）创建连续动作空间
            if continuous_space is not None and self.pv_control_enabled and self.pv_num > 0:
                continuous_dim = continuous_space.shape[0]
                
                # 假设每个PV系统有2个连续动作（有功功率 + 功率因数）
                expected_pv_dims = self.pv_num * 2
                if continuous_dim == expected_pv_dims:
                    # 每个PV系统独立的Box空间
                    for _ in range(self.pv_num):
                        agent_spaces.append(Box(
                            low=continuous_space.low[:2],
                            high=continuous_space.high[:2], 
                            shape=(2,), 
                            dtype=np.float32
                        ))
                else:
                    # 分配剩余维度给PV系统
                    dims_per_pv = max(1, continuous_dim // self.pv_num)
                    for i in range(self.pv_num):
                        start_idx = i * dims_per_pv
                        end_idx = min(start_idx + dims_per_pv, continuous_dim)
                        dim_size = end_idx - start_idx
                        agent_spaces.append(Box(
                            low=continuous_space.low[start_idx:end_idx],
                            high=continuous_space.high[start_idx:end_idx],
                            shape=(dim_size,),
                            dtype=np.float32
                        ))
            
            return agent_spaces
        else:
            # 默认：所有智能体使用相同的动作空间
            if isinstance(env_action_space, Discrete):
                return [env_action_space for _ in range(self.n_agents)]
            elif isinstance(env_action_space, Box):
                return [env_action_space for _ in range(self.n_agents)]
            else:
                # 最后的后备方案
                return [Discrete(2) for _ in range(self.n_agents)]
    
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
                for action in actions[:3]:
                    if isinstance(action, np.ndarray):
                        start_actions.append(f"[{', '.join([f'{x:.3f}' for x in action])}]")
                    else:
                        start_actions.append(str(action))
                for action in actions[-3:]:
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
            action_space_types = list()
            action_space_shapes = list()
            
            for i, space in enumerate(self.action_space):
                space_type = type(space).__name__
                action_space_types.append(space_type)
                
                if hasattr(space, 'shape'):
                    action_space_shapes.append(space.shape)
                elif hasattr(space, 'n'):
                    action_space_shapes.append((space.n,))
                    
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
# 保持与vvc_env_optimized.py的兼容性
OptimizedVVCEnv = VVCEnv




