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
        config = DEFAULT_DSR_CONFIG
        
        # 从args中更新配置
        if 'max_episode_steps' in args:
            config.max_episode_steps = args['max_episode_steps']
        if 'seed' in args:
            config.seed = args['seed']
        if 'system_name' in args:
            config.system_name = args['system_name']
        if 'use_render' in args:
            config.use_render = args['use_render']
        if 'load_noise' in args:
            config.load_noise = args['load_noise']
        
        return config
    
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
        
        # 动作空间
        self.action_space = []
        
        for i in range(self.n_agents):
            if self.core_env.agent_types[i] == 'switch':
                # 开关智能体：可选择的线路数量+1（不操作）
                n_lines = len([line for line in self.core_env.faultable_lines 
                              if line not in self.core_env.fault_lines])
                self.action_space.append(Discrete(n_lines + 1))
            
            elif self.core_env.agent_types[i] == 'pv':
                # PV智能体：配置的功率等级数量
                self.action_space.append(Discrete(self.config.pv_power_levels))
            
            elif self.core_env.agent_types[i] == 'load':
                # 负荷智能体：配置的动作等级数量
                self.action_space.append(Discrete(self.config.load_action_levels))
    
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
                load_name = load_agent['load_name']
                
                obs_components.extend([
                    float(obs_dict['load_states'].get(load_name, False)),  # 负荷状态
                    load_agent['priority'] / self.config.max_priority_level,  # 归一化优先级
                    float(load_agent['bus'] in obs_dict['energized_buses']),  # 母线通电状态
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