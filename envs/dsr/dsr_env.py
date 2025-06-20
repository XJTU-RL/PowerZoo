# -*- coding: utf-8 -*-
"""
DSR Environment Wrapper
配电网恢复环境包装器，对接PowerZoo框架
"""

import copy
import gym
import numpy as np
from gym.spaces import Discrete, Box
from typing import List, Dict, Any, Tuple, Optional

from envs.dsr.core.dsr_core import DSRCoreEnv
from envs.dsr.core.config import DSRConfig, DEFAULT_DSR_CONFIG


class DSREnv:
    """配电网恢复环境"""
    
    def __init__(self, args: Dict[str, Any], rank: Optional[int] = None):
        """
        初始化DSR环境
        
        Args:
            args: 环境参数字典，包含DSR配置
            rank: 并行进程编号
        """
        self.args = copy.deepcopy(args)
        self.rank = rank
        
        # 解析DSR配置
        self.config = self._parse_config(args)
        
        # 创建核心环境
        self.core_env = DSRCoreEnv(self.config, worker_idx=rank)
        
        # 设置智能体信息
        self.n_agents = self.core_env.n_agents
        self.agents = list(range(self.n_agents))
        
        # 定义观测和动作空间
        self._setup_spaces()
        
        # 环境状态
        self.current_step = 0
        
        # PowerZoo兼容性设置
        self.env_name = args.get('env_name', 'dsr')
        self.useS = args.get('useS', False)
        self.use_render = args.get('use_render', False)
        self.record_node = args.get('record_node', True)
        
        # 动作是否为离散
        self.discrete = True
    
    def _parse_config(self, args: Dict[str, Any]) -> DSRConfig:
        """解析配置参数"""
        # 创建配置副本
        config_dict = DEFAULT_DSR_CONFIG.__dict__.copy()
        
        # 从args中更新配置
        # 处理env_args中的参数
        env_args = args.get('env_args', {})
        
        # 更新所有相关参数
        update_keys = [
            # 基础配置
            'system_name', 'dss_file', 'max_episode_steps', 'seed', 'use_render', 'load_noise',
            # 设备配置
            'n_dg', 'n_pv', 'n_switch', 'n_load_levels',
            # 聚合配置
            'use_load_aggregation', 'n_load_agents', 'load_aggregation_method',
            # 物理约束
            'v_min', 'v_max', 'max_load_per_step',
            # 奖励权重
            'reward_restore', 'reward_voltage', 'reward_overload', 'reward_done',
            # 故障配置
            'min_faults', 'max_faults', 'fault_scenarios',
            # 高级特性
            'use_action_mask', 'use_dynamic_network', 'record_node',
            'scale', 'worker_idx', 'useS',
            # 动作空间配置
            'pv_power_levels', 'load_action_levels', 'pv_max_power',
            # 观测空间配置
            'obs_reserved_dim', 'default_voltage',
            # 设备重置配置
            'line_disconnect_prob', 'max_faultable_lines',
            # 负荷优先级配置
            'priority_weights', 'max_priority_level',
            # 恢复判断
            'restoration_threshold', 'success_threshold',
            # 日志记录配置
            'log_interval_episodes', 'recent_episodes_window',
            # 过载检测配置
            'overload_threshold', 'use_emergency_rating', 'log_overload_details',
            # IEEE系统配置
            'ieee123_load_count'
        ]
        
        # 首先从env_args更新
        for key in update_keys:
            if key in env_args:
                config_dict[key] = env_args[key]
        
        # 然后从顶层args更新（优先级更高）
        for key in update_keys:
            if key in args:
                config_dict[key] = args[key]
        
        # 创建新的配置对象
        return DSRConfig(**config_dict)
    
    def _setup_spaces(self):
        """设置观测和动作空间"""
        # 观测空间维度
        obs_dim = self._calculate_obs_dim()
        
        # 每个智能体的观测空间
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        
        # 共享观测空间（与单智能体观测相同）
        self.share_observation_space = self.observation_space.copy()
        
        # 动作空间（初始设置，将在reset后更新）
        self.action_space = []
        self._setup_initial_action_spaces()
    
    def _setup_initial_action_spaces(self):
        """设置初始动作空间（在知道故障线路之前）"""
        self.action_space = []
        
        for i in range(self.n_agents):
            if self.core_env.agent_types[i] == 'switch':
                # 开关智能体：使用最大可能的动作数（所有可故障线路+1）
                self.action_space.append(Discrete(len(self.core_env.faultable_lines) + 1))
            
            elif self.core_env.agent_types[i] == 'pv':
                # PV智能体：配置的功率等级数量
                self.action_space.append(Discrete(self.config.pv_power_levels))
            
            elif self.core_env.agent_types[i] == 'load':
                # 负荷智能体：配置的动作等级数量
                self.action_space.append(Discrete(self.config.load_action_levels))
    
    def _update_action_spaces(self):
        """在reset后更新动作空间（知道故障线路后）"""
        for i in range(self.n_agents):
            if self.core_env.agent_types[i] == 'switch':
                # 更新开关智能体的动作空间：排除故障线路
                n_lines = len([line for line in self.core_env.faultable_lines 
                              if line not in self.core_env.fault_lines])
                self.action_space[i] = Discrete(n_lines + 1)
    
    def _calculate_obs_dim(self) -> int:
        """计算观测维度"""
        # 基础观测维度包括：
        # - 时间步信息: 2 (current_step/max_steps, restored_ratio)
        # - 母线电压信息: n_bus
        # - 设备状态信息: n_switch + n_pv + n_load
        # - 智能体特定信息: 配置的预留维度
        
        n_bus = self.core_env.n_bus
        n_devices = len(self.core_env.faultable_lines) + self.config.n_pv + len(self.core_env.load_info)
        
        return 2 + n_bus + n_devices + self.config.obs_reserved_dim
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray], 
                                                List[List[float]], List[bool], 
                                                List[Dict], List[List[int]]]:
        """
        执行一步动作
        
        Returns:
            local_obs: 局部观测
            global_state: 全局状态
            rewards: 奖励列表
            dones: 结束标志列表
            infos: 信息字典列表
            available_actions: 可用动作
        """
        # 执行动作
        obs_dict, state_dict, rewards, done, info = self.core_env.step(actions)
        
        # 转换观测格式
        local_obs = self._convert_observations(obs_dict)
        global_state = self._convert_observations(state_dict)
        
        # 转换奖励格式
        rewards_list = [[reward] for reward in rewards]
        
        # 转换结束标志格式
        dones_list = [done] * self.n_agents
        
        # 转换信息格式
        infos_list = [info] * self.n_agents
        
        # 获取可用动作
        available_actions = self.get_avail_actions()
        
        # 处理时间截断
        if done and self.core_env.current_step >= self.config.max_episode_steps:
            for info_dict in infos_list:
                info_dict["TimeLimit.truncated"] = True
                info_dict["bad_transition"] = True
        
        return local_obs, global_state, rewards_list, dones_list, infos_list, available_actions
    
    def reset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]]]:
        """
        重置环境
        
        Returns:
            observations: 初始观测
            states: 初始状态
            available_actions: 可用动作
        """
        # 重置核心环境
        obs_dict, state_dict = self.core_env.reset()
        
        # 更新动作空间（现在知道了故障线路）
        self._update_action_spaces()
        
        # 转换观测格式
        observations = self._convert_observations(obs_dict)
        states = self._convert_observations(state_dict)
        
        # 获取可用动作
        available_actions = self.get_avail_actions()
        
        return observations, states, available_actions
    
    def _convert_observations(self, obs_dict: Dict[str, Any]) -> List[np.ndarray]:
        """将观测字典转换为智能体观测列表"""
        obs_list = []
        
        for i in range(self.n_agents):
            # 构造智能体i的观测
            obs = self._build_agent_observation(i, obs_dict)
            obs_list.append(obs)
        
        return obs_list
    
    def _build_agent_observation(self, agent_id: int, obs_dict: Dict[str, Any]) -> np.ndarray:
        """构建单个智能体的观测"""
        obs_components = []
        
        # 基础时间信息
        obs_components.extend([
            obs_dict['current_step'] / obs_dict['max_steps'],  # 归一化时间步
            len(obs_dict['energized_buses']) / self.core_env.n_bus,  # 通电率
        ])
        
        # 母线电压信息（简化：每个母线取最小相电压）
        for bus_name in self.core_env.all_bus_names:
            if bus_name in obs_dict['bus_voltages']:
                voltages = obs_dict['bus_voltages'][bus_name]
                min_voltage = min(voltages) if voltages else self.config.default_voltage
                obs_components.append(min_voltage)
            else:
                obs_components.append(self.config.default_voltage)  # 配置的默认电压
        
        # 根据智能体类型添加特定观测
        agent_type = self.core_env.agent_types[agent_id]
        
        if agent_type == 'switch':
            # 开关智能体：线路状态
            for line_name in self.core_env.faultable_lines:
                if line_name in obs_dict['line_states']:
                    obs_components.append(float(obs_dict['line_states'][line_name]))
                else:
                    obs_components.append(0.0)
            
            # 补充到目标维度
            while len(obs_components) < self._calculate_obs_dim():
                obs_components.append(0.0)
        
        elif agent_type == 'pv':
            # PV智能体：PV状态和附近负荷状态
            agent_idx = self.core_env.agent_indices[agent_id]
            if agent_idx < len(self.core_env.pv_agents):
                pv_agent = self.core_env.pv_agents[agent_idx]
                obs_components.extend([
                    pv_agent['current_power'] / pv_agent['max_power'],  # 当前功率比
                    float(pv_agent['bus'] in obs_dict['energized_buses']),  # 母线通电状态
                ])
            
            # 补充到目标维度
            while len(obs_components) < self._calculate_obs_dim():
                obs_components.append(0.0)
        
        elif agent_type == 'load':
            # 负荷智能体：负荷状态和优先级信息
            agent_idx = self.core_env.agent_indices[agent_id]
            if agent_idx < len(self.core_env.load_agents):
                load_agent = self.core_env.load_agents[agent_idx]
                
                # 处理聚合负荷智能体
                managed_loads = load_agent.get('managed_loads', [])
                if managed_loads:
                    # 聚合负荷智能体的观测
                    # 1. 管理的负荷总体状态
                    enabled_count = sum(1 for load in managed_loads 
                                      if obs_dict['load_states'].get(load, False))
                    total_count = len(managed_loads)
                    restoration_ratio = enabled_count / max(total_count, 1)
                    
                    # 2. 平均优先级和总功率信息
                    avg_priority = load_agent['priority'] / self.config.max_priority_level
                    
                    # 3. 可恢复负荷比例（母线通电的负荷）
                    restorable_count = sum(1 for load in managed_loads
                                         if self.core_env.load_info[load]['bus'] in obs_dict['energized_buses'])
                    restorable_ratio = restorable_count / max(total_count, 1)
                    
                    obs_components.extend([
                        restoration_ratio,  # 已恢复负荷比例
                        avg_priority,      # 平均优先级
                        restorable_ratio,  # 可恢复负荷比例
                        float(total_count),  # 管理的负荷数量（归一化）
                    ])
                else:
                    # 单个负荷智能体的观测
                    load_name = load_agent['load_name']
                    obs_components.extend([
                        float(obs_dict['load_states'].get(load_name, False)),  # 负荷状态
                        load_agent['priority'] / self.config.max_priority_level,  # 归一化优先级
                        float(load_agent['bus'] in obs_dict['energized_buses']),  # 母线通电状态
                        1.0,  # 管理1个负荷
                    ])
            
            # 补充到目标维度
            while len(obs_components) < self._calculate_obs_dim():
                obs_components.append(0.0)
        
        # 确保观测维度正确
        target_dim = self._calculate_obs_dim()
        obs_components = obs_components[:target_dim]  # 截断
        while len(obs_components) < target_dim:  # 补齐
            obs_components.append(0.0)
        
        return np.array(obs_components, dtype=np.float32)
    
    def get_avail_actions(self) -> List[List[int]]:
        """获取所有智能体的可用动作"""
        return self.core_env.get_available_actions()
    
    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """获取单个智能体的可用动作"""
        avail_actions = self.get_avail_actions()
        return avail_actions[agent_id]
    
    def render(self):
        """渲染环境（预留接口）"""
        pass
    
    def close(self):
        """关闭环境"""
        self.core_env.close()
        print("DSR环境已关闭")
    
    def seed(self, seed: int):
        """设置随机种子"""
        self.core_env.seed(seed)
    
    @property
    def agents(self) -> List[int]:
        """智能体ID列表"""
        return list(range(self.n_agents))


# 环境创建函数
def make_dsr_env(args: Dict[str, Any], rank: Optional[int] = None) -> DSREnv:
    """创建DSR环境"""
    return DSREnv(args, rank)