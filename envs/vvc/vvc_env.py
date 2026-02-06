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
import imageio
import glob
from envs.vvc.vvc.env_register import make_base_env, remove_parallel_dss

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class VVCEnv:
    def __init__(self, args,rank=None):#TODO: ranks是线程数 
        
        self.args = copy.deepcopy(args)
        self.env = make_base_env(args['env_name'], worker_idx=rank)#args
        self.env.seed(args['seed'] + 0)
        #智能体数量是电容、有载调压器、电池、PV数量之和（根据PV是否启用）
        pv_count = self.env.pv_num if (hasattr(self.env, 'pv_control_enabled') and self.env.pv_control_enabled) else 0
        total_agents = self.env.cap_num + self.env.reg_num + self.env.bat_num + pv_count
        agents = [i for i in range(0, total_agents)]
        self.agents = agents
        self.n_agents = len(agents)
        
        #排序顺序 - 支持CRBP架构
        self.cap_names = self.env.cap_names
        self.reg_names = self.env.reg_names
        self.bat_names = self.env.bat_names
        self.pv_names = getattr(self.env, 'pv_names', []) if (hasattr(self.env, 'pv_control_enabled') and self.env.pv_control_enabled) else []
        
        self.rank=rank#线程编号
        self.env_name=args['env_name']#便于实现多线程
        
        agents_names = self.cap_names + self.reg_names + self.bat_names + self.pv_names
        self.env.use_render=args['use_render']
        self.env.useS=args['useS']
        self.env.record_node = args['record_node']
        if args['useS']==True:
            update_orders=list(range(0,self.n_agents))
            self.ordered_agents_pairs = dict(zip(agents_names, update_orders))
            self.agents_bus=self.env.agents_bus 
        else:
            self.ordered_agents_pairs = None
            self.agents_bus=None
        self.share_observation_space = self.repeat(self.env.observation_space)
        self.observation_space = self.unwrap(self.env.observation_space)
        
        self.action_space = self.get_env_action_space(self.env.action_space)#把每个智能体的动作空间拆解出来了
        self.avail_actions = self.get_avail_actions()
        if self.env.action_space.__class__.__name__ == "Box":
            self.discrete = False
        else:
            self.discrete = True # 对所有的动作空间进行离散化处理，需要对电池进行处理，使其离散化
        
    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        支持CRBP混合动作空间处理
        """
        # 将多智能体动作转换为环境期望的格式
        env_action = self._convert_actions_to_env_format(actions)
        
        # 执行环境步骤
        obs, rew, done, info = self.env.step(env_action)
        
        if done:
            if (
                "TimeLimit.truncated" in info.keys()
                and info["TimeLimit.truncated"] == True
            ):
                info["bad_transition"] = True
        
        # 统一返回格式: rewards (n_agents, 1), dones (n_agents,), infos list of n_agents dicts
        rewards = np.array([[float(rew)]] * self.n_agents, dtype=np.float32)
        dones = np.array([bool(done)] * self.n_agents, dtype=bool)
        infos = [info] * self.n_agents
        return self.unwrap(obs), self.unwrap(obs), rewards, dones, infos, self.get_avail_actions()
    
    def _convert_actions_to_env_format(self, actions):
        """
        将多智能体动作转换为环境期望的格式
        支持离散和混合动作空间
        """
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            # 混合动作空间处理
            discrete_actions = []
            continuous_actions = []
            
            action_idx = 0
            
            # 收集离散动作（电容器 + 调压器 + 离散电池/PV）
            discrete_count = (self.env.cap_num + self.env.reg_num + 
                            (self.env.bat_num if self.env.bat_act_num < float('inf') else 0) +
                            (self.env.pv_num if (hasattr(self.env, 'pv_control_enabled') and 
                                               self.env.pv_control_enabled and 
                                               self.env.pv_act_num < float('inf')) else 0))
            
            for i in range(discrete_count):
                if action_idx < len(actions):
                    discrete_actions.append(int(actions[action_idx]))
                    action_idx += 1
            
            # 收集连续动作（连续电池 + 连续PV）
            # 连续电池动作
            if hasattr(self.env, 'bat_num') and self.env.bat_act_num == float('inf'):
                for i in range(self.env.bat_num):
                    if action_idx < len(actions):
                        continuous_actions.append(float(actions[action_idx]))
                        action_idx += 1
            
            # 连续PV动作
            if (hasattr(self.env, 'pv_control_enabled') and self.env.pv_control_enabled and 
                hasattr(self.env, 'pv_num') and self.env.pv_act_num == float('inf')):
                for i in range(self.env.pv_num):
                    if action_idx < len(actions) - 1:  # PV需要两个参数
                        continuous_actions.extend([float(actions[action_idx]), float(actions[action_idx + 1])])
                        action_idx += 2
            
            # 返回混合动作
            if discrete_actions and continuous_actions:
                return (np.array(discrete_actions), np.array(continuous_actions))
            elif discrete_actions:
                return np.array(discrete_actions)
            else:
                return np.array(continuous_actions)
        
        else:
            # 纯离散或连续动作空间
            if isinstance(actions, list):
                return np.array([float(a) if isinstance(self.action_space[i], gym.spaces.Box) else int(a) 
                               for i, a in enumerate(actions)])
            else:
                return actions.flatten() if hasattr(actions, 'flatten') else actions

    def reset(self):
        """Returns initial observations and states"""
        #self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(load_profile_idx=self.rank))
        s_obs = copy.deepcopy(obs)
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.array(avail_actions,dtype=object).tolist()
    
    def get_avail_agent_actions(self, agent_id):
        """
        返回指定智能体的可用动作
        考虑设备物理约束和当前状态
        """
        # 对于连续动作空间，返回None（表示所有动作都可用）
        if isinstance(self.action_space[agent_id], gym.spaces.Box):
            return None  # 连续动作空间不需要available actions
        
        # 获取智能体类型和索引
        agent_type, type_index = self._get_agent_type_and_index(agent_id)
        
        if agent_type == 'capacitor':
            # 电容器: 通常所有动作都可用 (0=off, 1=on)
            return [1, 1]
        
        elif agent_type == 'regulator':
            # 调压器: 需要考虑tap位置限制
            try:
                reg_name = self.reg_names[type_index]
                if hasattr(self.env, 'circuit') and reg_name in self.env.circuit.regulators:
                    reg = self.env.circuit.regulators[reg_name]
                    current_tap = reg.tap
                    min_tap, max_tap = reg.min_tap, reg.max_tap
                    
                    # 计算可用的tap位置
                    avail = [0] * self.action_space[agent_id].n
                    for tap in range(min_tap, max_tap + 1):
                        if 0 <= tap < len(avail):
                            avail[tap] = 1
                    return avail
                else:
                    return [1] * self.action_space[agent_id].n
            except (IndexError, AttributeError):
                return [1] * self.action_space[agent_id].n
        
        elif agent_type == 'battery':
            # 电池: 需要考虑SOC约束
            try:
                bat_name = self.bat_names[type_index]
                if hasattr(self.env, 'circuit') and bat_name in self.env.circuit.batteries:
                    bat = self.env.circuit.batteries[bat_name]
                    soc = bat.soc
                    
                    # 简化的SOC约束检查
                    avail = [1] * self.action_space[agent_id].n
                    
                    # 如果SOC太低，限制放电动作
                    if soc < 0.1:  # SOC < 10%
                        # 限制放电动作（具体限制取决于动作空间设计）
                        if self.action_space[agent_id].n > 16:  # 假设中间以上是放电
                            for i in range(16, self.action_space[agent_id].n):
                                avail[i] = 0
                    
                    # 如果SOC太高，限制充电动作
                    elif soc > 0.9:  # SOC > 90%
                        # 限制充电动作
                        if self.action_space[agent_id].n > 16:
                            for i in range(0, 16):
                                avail[i] = 0
                    
                    return avail
                else:
                    return [1] * self.action_space[agent_id].n
            except (IndexError, AttributeError):
                return [1] * self.action_space[agent_id].n
        
        elif agent_type == 'pv':
            # PV: 连续控制或离散控制
            if isinstance(self.action_space[agent_id], gym.spaces.Box):
                return None  # 连续PV控制
            else:
                # 离散PV控制: 通常所有动作都可用
                return [1] * self.action_space[agent_id].n
        
        else:
            # 默认情况：所有动作都可用
            return [1] * self.action_space[agent_id].n
    
    def _get_agent_type_and_index(self, agent_id):
        """
        根据智能体ID确定设备类型和在该类型中的索引
        返回: (agent_type, type_index)
        """
        current_idx = 0
        
        # 电容器
        if agent_id < current_idx + self.env.cap_num:
            return 'capacitor', agent_id - current_idx
        current_idx += self.env.cap_num
        
        # 调压器
        if agent_id < current_idx + self.env.reg_num:
            return 'regulator', agent_id - current_idx
        current_idx += self.env.reg_num
        
        # 电池
        if agent_id < current_idx + self.env.bat_num:
            return 'battery', agent_id - current_idx
        current_idx += self.env.bat_num
        
        # PV系统
        if (hasattr(self.env, 'pv_control_enabled') and self.env.pv_control_enabled and 
            agent_id < current_idx + self.env.pv_num):
            return 'pv', agent_id - current_idx
        
        return 'unknown', 0
        

    def render(self):#函数需要修改
        #self.env.render()
        pass

    def close(self):
        #self.env.close()
        remove_parallel_dss(self.env_name, self.rank)
        print("Closing the environment")

    def seed(self, seed):
        #self.env.seed(seed)
        self.env.seed(seed)#use default seed
        
    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d)
        return l 
    
    def get_env_action_space(self, env):  # 把混合动作空间分给每个单独的智能体 - 支持CRBP架构
        """
        将环境的动作空间分解为每个智能体的动作空间
        支持离散动作空间和混合动作空间（Tuple）
        """
        action_spaces = []
        
        if isinstance(env, gym.spaces.Tuple):
            # 混合动作空间：离散 + 连续
            discrete_space, continuous_space = env.spaces[0], env.spaces[1]
            
            # 处理离散动作部分
            if hasattr(discrete_space, 'nvec'):
                action_spaces.extend([Discrete(n) for n in discrete_space.nvec])
            
            # 处理连续动作部分 - 每个连续动作作为一个智能体
            if hasattr(continuous_space, 'shape') and continuous_space.shape[0] > 0:
                continuous_dim = continuous_space.shape[0]
                
                # 根据设备类型分配连续动作
                # 电池: 1维连续动作
                # PV: 2维连续动作 (有功功率 + 功率因数)
                if hasattr(self.env, 'bat_num') and hasattr(self.env, 'pv_num'):
                    bat_continuous = self.env.bat_num if self.env.bat_act_num == float('inf') else 0
                    pv_continuous = self.env.pv_num * 2 if (hasattr(self.env, 'pv_control_enabled') and 
                                                         self.env.pv_control_enabled and 
                                                         self.env.pv_act_num == float('inf')) else 0
                    
                    # 电池连续动作
                    for _ in range(bat_continuous):
                        action_spaces.append(Box(low=-1, high=1, shape=(1,), dtype=np.float32))
                    
                    # PV连续动作
                    for _ in range(self.env.pv_num if (hasattr(self.env, 'pv_control_enabled') and 
                                                    self.env.pv_control_enabled and 
                                                    self.env.pv_act_num == float('inf')) else 0):
                        action_spaces.append(Box(low=-1, high=1, shape=(2,), dtype=np.float32))  # [有功功率, 功率因数]
                else:
                    # 默认处理：每个连续维度作为一个智能体
                    for i in range(continuous_dim):
                        action_spaces.append(Box(low=-1, high=1, shape=(1,), dtype=np.float32))
        
        elif hasattr(env, 'nvec'):
            # 纯离散动作空间
            action_spaces = [Discrete(n) for n in env.nvec]
        
        elif isinstance(env, gym.spaces.Box):
            # 纯连续动作空间
            for i in range(env.shape[0]):
                action_spaces.append(Box(low=env.low[i], high=env.high[i], shape=(1,), dtype=env.dtype))
        
        else:
            raise ValueError(f"不支持的动作空间类型: {type(env)}")
        
        return action_spaces
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    