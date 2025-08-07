
import os
import gym
import numpy as np
from envs.power_envs.powerzoo_llm.circuit_system import Circuits
from envs.power_envs.powerzoo_llm.data_process.loadprofile import LoadProfile
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache, wraps
import logging
import time
from copy import deepcopy 

try:
    from envs.power_envs.powerzoo_llm.utils import get_logger, log_reward_components, log_device_actions
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    
    def log_reward_components(*args, **kwargs):
        pass
    
    def log_device_actions(*args, **kwargs):
        pass

logger = get_logger(__name__)


def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start_time
            if elapsed > 0.1:  # 记录耗时超过100ms的操作
                logger.debug(f"{func.__name__} 耗时: {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败 (耗时{elapsed:.3f}s): {e}")
            raise
    return wrapper

#### action space class ####
class ActionSpace:
    '''电容器、调压器、电池和光伏系统的动作空间封装

    属性:
        cap_num, reg_num, bat_num, pv_num (int): 电容器、调压器、电池和光伏系统的数量
        reg_act_num, bat_act_num, pv_act_num: 调压器、电池和光伏系统的动作数量
        space (gym.spaces): 来自gym的空间对象
        pv_control_enabled (bool): 是否启用光伏控制

    注意:
        当所有系统都使用离散动作时,space为MultiDiscrete类型;
        否则,space为MultiDiscrete和Box的元组,用于连续动作
    '''
    def __init__(self, CRBP_num: Tuple[int, int, int, int], RBP_act_num: Tuple[int, int, int], pv_control_enabled: bool = False):
        self.cap_num, self.reg_num, self.bat_num, self.pv_num = CRBP_num
        self.reg_act_num, self.bat_act_num, self.pv_act_num = RBP_act_num
        self.pv_control_enabled = pv_control_enabled
        
        # 缓存空间对象避免重复创建
        self._space = None
        self._initialize_space()
    
    def _initialize_space(self):
        """初始化动作空间 - 支持CRBP架构"""
        discrete_actions = ([2] * self.cap_num +  # 电容器动作 (0/1)
                          [self.reg_act_num] * self.reg_num)  # 调压器动作 (tap position)
        
        continuous_actions = []
        continuous_shape = 0
        
        # 处理电池动作空间
        if self.bat_num > 0:
            if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num < float('inf'):
                discrete_actions.extend([int(self.bat_act_num)] * self.bat_num)  # 离散电池动作
            else:
                continuous_shape += self.bat_num  # 连续电池动作
        
        # 处理光伏动作空间（如果启用）
        if self.pv_control_enabled and self.pv_num > 0:
            if isinstance(self.pv_act_num, (int, float)) and self.pv_act_num < float('inf'):
                discrete_actions.extend([int(self.pv_act_num)] * self.pv_num)  # 离散PV动作
            else:
                continuous_shape += self.pv_num * 2  # 连续PV动作 (有功功率 + 功率因数)
        
        # 构建最终动作空间
        if continuous_shape > 0:
            # 混合动作空间：离散 + 连续
            discrete_space = gym.spaces.MultiDiscrete(discrete_actions) if discrete_actions else None
            continuous_space = gym.spaces.Box(low=-1, high=1, shape=(continuous_shape,), dtype=np.float32) #PV的两个控制量都是(-1,1)
            
            if discrete_space is not None:
                self._space = gym.spaces.Tuple((discrete_space, continuous_space))#(discrete * num_crb, continuous * num_pv)
            else:
                self._space = continuous_space
        else:
            # 纯离散动作空间
            self._space = gym.spaces.MultiDiscrete(discrete_actions)
    
    @property
    def space(self):
        return self._space

    def sample(self):
        """HAPPO兼容的采样方法"""
        ss = self._space.sample()
        
        # 处理混合动作空间：保持语义分离，而非强制转换
        if isinstance(self._space, gym.spaces.Tuple):
            if len(ss) == 2:  # 离散 + 连续
                discrete_part = ss[0]  # 保持离散动作为整数
                continuous_part = ss[1].astype(np.float32)  # 连续动作保持浮点数
                # 返回混合动作列表：[离散动作数组, 连续动作数组]
                return [discrete_part, continuous_part]
            else:
                # 处理其他Tuple情况，保持原始结构
                return list(ss)
        
        # 纯离散或纯连续动作空间
        return ss

    def seed(self, seed: int):
        self._space.seed(seed)

    def dim(self) -> int:
        """返回动作空间维度 - 支持CRBP混合动作空间"""
        if isinstance(self._space, gym.spaces.Tuple):
            total_dim = 0
            for subspace in self._space.spaces:
                if hasattr(subspace, 'shape'):
                    total_dim += subspace.shape[0] if subspace.shape else subspace.n
                elif hasattr(subspace, 'nvec'):
                    total_dim += len(subspace.nvec)
                else:
                    total_dim += subspace.n
            return total_dim
        elif hasattr(self._space, 'nvec'):
            return len(self._space.nvec)
        else:
            return self._space.shape[0] if self._space.shape else self._space.n

    def CRBP_num(self) -> Tuple[int, int, int, int]:
        """返回各设备数量"""
        return self.cap_num, self.reg_num, self.bat_num, self.pv_num

    def RBP_act_num(self) -> Tuple[int, int, int]:
        """返回调压器、电池、光伏动作数量"""
        return self.reg_act_num, self.bat_act_num, self.pv_act_num
    
    def get_action_dims(self) -> Dict[str, int]:
        """返回各设备类型的动作维度"""
        return {
            'capacitor': self.cap_num,
            'regulator': self.reg_num, 
            'battery': self.bat_num,
            'pv': self.pv_num if self.pv_control_enabled else 0
        }

#### environment class ####
class Env(gym.Env):
    """Enviroment to train RL agent
    
    Attributes:
        obs (dict): Observation/state of system
        dss_folder_path (str): Path to folder containing DSS file
        dss_file (str): DSS simulation filename
        source_bus (str): the bus (with coordinates in BusCoords.csv) closest to the source
        node_size (int): the size of node in plots
        shift (int): the shift amount of labels in plots
        show_node_labels (bool): show node labels in plots
        scale (float): scale of the load profile
        wrap_observation (bool): whether to flatten obs into array at the outputs of reset & step
        observe_load (bool): whether to include the nodal loads in the observation
        
        load_profile (obj): Class for load profile management
        num_profiles (int): number of distinct profiles generated by load_profile
        horizon (int): Maximum steps in a episode
        circuit (obj): Circuit object linking to DSS simulation
        all_bus_names (list): All bus names in system
        cap_names (list): List of capacitor bus
        reg_names (list): List of regulator bus
        bat_names (list): List of battery bus
        cap_num (int): number of capacitors
        reg_num (int): number of regulators
        bat_num (int): number of batteries
        reg_act_num (int): Number of reg control actions
        bat_act_num (int): Number of bat control actions
        topology (graph): NxGraph of power system
        reward_func (obj): Class of reward fucntions
        t (int): Timestep for environment state
        ActionSpace (obj): Action space class. Use for sampling random actions
        action_space (gym.spaces): the base action space from class ActionSpace
        observation_space (gym.spaces): observation space of environment.
        
    Defined at self.step(), self.reset():
        all_load_profiles (dict): 2D array of load profile for all bus and time
    
    Defined at self.step() and used at self.plot_graph()
        self.str_action: the action string to be printed at self.plot_graph()
        
    Defined at self.build_graph():
        edges (dict): Dict of edges connecting nodes in circuit
        lines (dict): Dict of edges with components in circuit
        transformers (dict): Dictionary of transformers in system
    """  
    def __init__(self, folder_path: str, info: Dict[str, Any], dss_act: bool = False):
        super().__init__()
        
        # 基础配置
        self.obs = {}
        self.dss_folder_path = os.path.join(folder_path, info['system_name'])
        self.dss_file = info['dss_file']
        self.source_bus = info.get('source_bus', 'sourcebus')
        self.node_size = info.get('node_size', 300)
        self.shift = info.get('shift', 50)
        self.show_node_labels = info.get('show_node_labels', False)
        self.scale = info.get('scale', 1.0)
        self.wrap_observation = True
        self.observe_load = False
        self.LLM = info.get('for_LLM', False) # 是否为LLM环境
        #NOTE: 添加了智能体节点与智能体名称的对应关系
        self.agents_bus=dict()
        
        # generate load profile files
        self.load_profile = LoadProfile(\
                 info['max_episode_steps'],
                 self.dss_folder_path,
                 self.dss_file,
                 worker_idx = info['worker_idx'] if 'worker_idx' in info else None,
                 pv_data_source=info.get('pv_data_source'),
                 temperature_data_source=info.get('temperature_data_source'))

        # 生成负载数据
        self.num_profiles = self.load_profile.generate_episodes_from_existing_files(scale=self.scale)
        # 选择一个虚拟负载曲线用于电路初始化
        self.load_profile.select_load_profile(0)#在这里得到loadprofile的路径
        
        # 问题的时间范围等于负载曲线的长度
        self.horizon = info['max_episode_steps']
        self.reg_act_num = info['reg_act_num']
        self.bat_act_num = info['bat_act_num']
        assert self.horizon>=1, 'invalid horizon'
        assert self.reg_act_num>=2 and self.bat_act_num>=2, 'invalid act nums'
        
        # 添加PV控制配置
        self.pv_control_enabled = info.get('pv_control', False)
        self.pv_act_num = info.get('pv_act_num', float('inf'))  # 默认连续控制
        
        self.circuit = Circuits(os.path.join(self.dss_folder_path, self.dss_file),
                                RB_act_num=(self.reg_act_num, self.bat_act_num),
                                dss_act=dss_act)
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.cap_names = list(self.circuit.capacitors.keys())
        self.reg_names = list(self.circuit.regulators.keys())
        self.bat_names = list(self.circuit.batteries.keys())
        self.pv_names = list(self.circuit.pvs.keys()) if hasattr(self.circuit, 'pvs') else []
        
        self.cap_num = len(self.cap_names)
        self.reg_num = len(self.reg_names)
        self.bat_num = len(self.bat_names)
        self.pv_num = len(self.pv_names)
        
        assert self.cap_num>=0 and self.reg_num>=0 and self.bat_num>=0 and self.pv_num>=0 and \
               self.cap_num + self.reg_num + self.bat_num + self.pv_num>=1,'invalid CRBP_num'
        
        self.topology = self.build_graph()
        self.reward_func = self.MyReward(self, info)
        self.t = 0
        
        # create action space and observation space
        self.ActionSpace = ActionSpace( (self.cap_num, self.reg_num, self.bat_num, self.pv_num),
                                        (self.reg_act_num, self.bat_act_num, self.pv_act_num),
                                        pv_control_enabled=self.pv_control_enabled )
        self.action_space = self.ActionSpace.space
        self.reset_obs_space()
        self.useS=False
        self.use_render=False
        self.agents_bus=self.circuit.get_agent_bus_dict()

    def reset_obs_space(self, wrap_observation=True, observe_load=False):
        '''
        根据包装和负载选项重置观测空间。
        
        建议通过此函数设置 wrap_observation 和 observe_load,
        而不是直接设置属性(例如 Env.wrap_observation)。
        
        '''
        self.wrap_observation = wrap_observation
        self.observe_load = observe_load
        
        self.reset(load_profile_idx=0)
        node_num = len(np.hstack( list(self.obs['bus_voltages'].values()) )) # 节点数量
        if observe_load: nload = len(self.obs['load_profile_t'])
        
        if self.wrap_observation:
            low, high = [0.8]*node_num, [1.2]*node_num  # add voltage bound
            low, high = low+[0]*self.cap_num, high+[1]*self.cap_num # add cap bound
            low, high = low+[0]*self.reg_num, high+[self.reg_act_num]*self.reg_num # add reg bound
            low, high = low+[0,-1]*self.bat_num, high+[1,1]*self.bat_num # add bat bound
            
            # 添加PV状态边界（如果启用PV控制）
            if self.pv_control_enabled and self.pv_num > 0:
                low, high = low+[0.0,-1.0]*self.pv_num, high+[1.0,1.0]*self.pv_num  # PV有功功率 + 功率因数
            
            if observe_load: low, high = low+[0.0]*nload, high+[1.0]*nload # add load bound
            low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low, high) 
        else:
            bat_dict = {bat: gym.spaces.Box(np.array([0,-1]), np.array([1,1]), dtype=np.float32) 
                        for bat in self.obs['bat_statuses'].keys()}
            
            obs_dict = {
                'bus_voltages': gym.spaces.Box(0.8, 1.2, shape=(node_num,)),
                'cap_statuses': gym.spaces.MultiDiscrete([2]*self.cap_num),
                'reg_statuses': gym.spaces.MultiDiscrete([self.reg_act_num]*self.reg_num), 
                'bat_statuses': gym.spaces.Dict(bat_dict)
            }
            
            # 添加PV状态空间（如果启用PV控制）
            if self.pv_control_enabled and self.pv_num > 0:
                pv_dict = {pv: gym.spaces.Box(np.array([0.0,-1.0]), np.array([1.0,1.0]), dtype=np.float32) 
                          for pv in self.pv_names}
                obs_dict['pv_statuses'] = gym.spaces.Dict(pv_dict)
            
            if observe_load: obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
            self.observation_space = gym.spaces.Dict(obs_dict)

    class MyReward:
        """Reward definition class
        
        Attributes:
            env (obj): Inherits all attributes of environment 
        """
        def __init__(self, env, info: Dict[str, Any]):
            self.env = env
            self.power_w = info.get('power_w', 1.0)  # 降低功率损耗权重避免奖励scale过大
            self.cap_w = info.get('cap_w', 0.1)
            self.reg_w = info.get('reg_w', 0.1)
            self.soc_w = info.get('soc_w', 0.5)
            self.dis_w = info.get('dis_w', 0.1)
            self.pv_w = info.get('pv_w', 1.0)  # PV控制奖励权重
            
            # 约束感知奖励参数 - 为HAPPO训练优化
            self.voltage_penalty_scale = info.get('voltage_penalty_scale', 1.0)  # 电压约束惩罚系数
            self.constraint_aware = info.get('constraint_aware', True)  # 是否启用约束感知
            self.voltage_target_range = info.get('voltage_target_range', (0.95, 1.05))  # 理想电压范围
            self.progressive_penalty = info.get('progressive_penalty', True)  # 渐进式惩罚

        @performance_monitor
        def powerloss_reward(self) -> float:
            """功率损耗奖励"""
            current_loss = self.env.obs.get('power_loss', 0)
            # Handle both scalar and array types
            if hasattr(current_loss, '__len__'):
                # If it's an array, take the first element or sum
                current_loss = float(current_loss[0]) if len(current_loss) > 0 else 0.0
            else:
                current_loss = float(current_loss)
            ratio = max(0.0, min(1.0, current_loss))
            return -ratio * self.power_w

        def ctrl_reward(self, capdiff: List[float], regdiff: List[float], 
                       soc_err: List[float], discharge_err: List[float],
                       pv_diff: List[float] = None) -> float:
            """控制动作奖励 - 支持PV控制"""
            pv_diff = pv_diff or []
            
            cost = (self.cap_w * sum(capdiff) + 
                    self.reg_w * sum(regdiff) + 
                    (0.0 if self.env.t != self.env.horizon else self.soc_w * sum(soc_err)) + 
                    self.dis_w * sum(discharge_err) +
                    self.pv_w * sum(pv_diff))  # 添加PV控制成本
            return -cost

        @performance_monitor
        def voltage_reward(self, record_node: bool = False) -> Tuple[float, List[str]]:
            """电压奖励"""
            violated_nodes = []
            total_violation = 0.0
            
            bus_voltages = self.env.obs.get('bus_voltages', {})
            
            # 批量处理电压违规计算
            for name, voltages in bus_voltages.items():
                if not voltages:
                    continue
                    
                max_v = max(voltages)
                min_v = min(voltages)
                
                max_penalty = min(0, 1.05 - max_v) if max_v > 1.05 else 0
                min_penalty = min(0, min_v - 0.95) if min_v < 0.95 else 0
                
                violation = max_penalty + min_penalty
                total_violation += violation
                
                if record_node and violation != 0:
                    violated_nodes.append(name)
            
            return total_violation, violated_nodes
        
        def composite_reward(self, cd: List[float], rd: List[float], 
                            soc: List[float], dis: List[float], 
                            pv_diff: List[float] = None,
                            full: bool = True, record_node: bool = False) -> Tuple[float, Dict[str, Any]]:
            """综合奖励计算 - 约束感知版本"""
            p = self.powerloss_reward()
            v, vio_nodes = self.voltage_reward(record_node)
            t = self.ctrl_reward(cd, rd, soc, dis, pv_diff or [])
            
            # 约束感知奖励计算
            if self.constraint_aware:
                # 增强电压约束奖励
                v_enhanced = self.enhanced_voltage_reward(vio_nodes)
                # 功率平衡奖励
                power_balance_reward = self.power_balance_reward()
                # PV优化奖励
                pv_optimization_reward = self.pv_optimization_reward()
                
                summ = (t + v_enhanced + p * 0.1 + 
                       power_balance_reward + pv_optimization_reward)
                
                # 记录约束感知奖励详情
                constraint_components = {
                    'power_loss': p,
                    'voltage_enhanced': v_enhanced,
                    'control': t,
                    'power_balance': power_balance_reward,
                    'pv_optimization': pv_optimization_reward,
                    'total_constraint_aware': summ
                }
                log_reward_components(logger, constraint_components)
            else:
                summ = t + v  # 原始奖励结构
                
                # 记录基础奖励详情
                basic_components = {
                    'power_loss': p,
                    'voltage': v,
                    'control': t,
                    'total': summ
                }
                log_reward_components(logger, basic_components)
            
            info = {} if not record_node else {'violated_nodes': vio_nodes}
            if full:
                info.update({
                    'power_loss_ratio': -p / self.power_w,
                    'vol_reward': v,
                    'ctrl_reward': t,
                    'total_reward': summ
                })
                
                if self.constraint_aware:
                    info.update({
                        'voltage_violations': len(vio_nodes),
                        'voltage_compliance_rate': self._calculate_voltage_compliance(),
                        'power_balance_score': power_balance_reward,
                        'pv_utilization_score': pv_optimization_reward
                    })
            
            return summ, info
        
        def enhanced_voltage_reward(self, vio_nodes: List[str]) -> float:
            """增强电压约束奖励 - 约束感知版本"""
            bus_voltages = self.env.obs.get('bus_voltages', {})
            total_penalty = 0.0
            total_nodes = 0
            
            v_min, v_max = self.voltage_target_range
            
            for name, voltages in bus_voltages.items():
                if not voltages:
                    continue
                    
                for v in voltages:
                    total_nodes += 1
                    
                    if self.progressive_penalty:
                        # 渐进式惩罚: 越远离正常范围惩罚越大
                        if v > v_max:
                            violation_degree = (v - v_max) / (1.2 - v_max)  # 归一化违约程度
                            penalty = violation_degree ** 2 * self.voltage_penalty_scale * 2.0  # 降低惩罚系数从50到2
                        elif v < v_min:
                            violation_degree = (v_min - v) / (v_min - 0.8)
                            penalty = violation_degree ** 2 * self.voltage_penalty_scale * 2.0  # 降低惩罚系数从50到2
                        else:
                            # 在合理范围内给予奖励
                            penalty = -0.1  # 小奖励
                    else:
                        # 简单的二元惩罚
                        if v > v_max or v < v_min:
                            penalty = self.voltage_penalty_scale * 1.0  # 降低惩罚系数从10到1
                        else:
                            penalty = -0.01  # 降低奖励避免影响过大
                    
                    total_penalty += penalty
            
            # 平均化惩罚
            avg_penalty = total_penalty / max(total_nodes, 1)
            return -avg_penalty
        
        def power_balance_reward(self) -> float:
            """功率平衡奖励 - 鼓励系统功率平衡"""
            power_loss_ratio = abs(self.env.obs.get('power_loss', 0))
            
            # Handle both scalar and array types
            if hasattr(power_loss_ratio, '__len__'):
                # If it's an array, take the first element
                power_loss_ratio = float(power_loss_ratio[0]) if len(power_loss_ratio) > 0 else 0.0
            else:
                power_loss_ratio = float(power_loss_ratio)
            
            # 功率损耗越低奖励越大
            if power_loss_ratio < 0.02:  # <2%损耗
                return 5.0
            elif power_loss_ratio < 0.05:  # <5%损耗
                return 2.0
            elif power_loss_ratio < 0.10:  # <10%损耗
                return 0.0
            else:
                return -power_loss_ratio * 20  # 高损耗惩罚
        
        def pv_optimization_reward(self) -> float:
            """光伏优化奖励 - 鼓励合理使用PV系统"""
            if not (self.env.pv_control_enabled and self.env.pv_num > 0):
                return 0.0
            
            pv_statuses = self.env.obs.get('pv_statuses', {})
            if not pv_statuses:
                return 0.0
            
            total_reward = 0.0
            for pv_name, status in pv_statuses.items():
                if len(status) >= 2:
                    p_ratio, pf = status[0], status[1]
                    
                    # 鼓励高功率输出（在高辐照时）
                    power_reward = p_ratio * 2.0
                    
                    # 鼓励合理的功率因数（接近单位功率因数）
                    pf_penalty = abs(pf - 1.0) * 1.0
                    
                    # 鼓励电压支撑（当系统电压低时）
                    voltage_support_reward = 0.0
                    bus_voltages = self.env.obs.get('bus_voltages', {})
                    avg_voltage = np.mean([np.mean(v) for v in bus_voltages.values() if v])
                    if avg_voltage < 0.98 and pf > 0.95:  # 低电压时发出无功
                        voltage_support_reward = 1.0
                    
                    total_reward += power_reward - pf_penalty + voltage_support_reward
            
            return total_reward / max(self.env.pv_num, 1)
        
        def _calculate_voltage_compliance(self) -> float:
            """计算电压合格率"""
            bus_voltages = self.env.obs.get('bus_voltages', {})
            total_measurements = 0
            compliant_measurements = 0
            
            v_min, v_max = self.voltage_target_range
            
            for voltages in bus_voltages.values():
                for v in voltages:
                    total_measurements += 1
                    if v_min <= v <= v_max:
                        compliant_measurements += 1
            
            return compliant_measurements / max(total_measurements, 1)

    @performance_monitor
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """环境步进
        
        Args:
            action: Integer array of actions for capacitors, regulators and batteries
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        action_idx = 0
        self.str_action = '' # the action string to be printed at self.plot_graph()
        
        ### capacitor control
        if self.cap_num>0:
            statuses = action[action_idx:action_idx+self.cap_num]
            capdiff = self.circuit.set_all_capacitor_statuses(statuses)
            cap_statuses = {cap:status for cap, status in \
                            zip(self.circuit.capacitors.keys(), statuses)}
            action_idx += self.cap_num
            self.str_action += 'Cap Status:'+str(statuses)
            
            # 记录电容器动作详情
            for i, (cap_name, status) in enumerate(cap_statuses.items()):
                old_status = getattr(self.circuit.capacitors[cap_name], 'status', 0)
                log_device_actions(logger, "Capacitor", cap_name, old_status, status, capdiff[i])
        else: capdiff, cap_statuses = [], dict()

        ### regulator control
        if self.reg_num>0:
            tapnums = action[action_idx:action_idx+self.reg_num]
            # 获取旧的抽头值用于日志记录
            old_tapnums = [self.circuit.regulators[reg].tap for reg in self.reg_names]
            regdiff = self.circuit.set_all_regulator_tappings(tapnums)
            reg_statuses = {reg:self.circuit.regulators[reg].tap \
                            for reg in self.reg_names}
            action_idx += self.reg_num
            self.str_action += '调压器抽头状态'+str(tapnums)
            
            # 记录调压器动作详情
            for i, reg_name in enumerate(self.reg_names):
                new_tap = reg_statuses[reg_name]
                log_device_actions(logger, "Regulator", reg_name, old_tapnums[i], new_tap, regdiff[i])
        else: regdiff, reg_statuses = [], dict()

        ### battery control
        if self.bat_num>0:
            if isinstance(self.action_space, gym.spaces.Tuple) and isinstance(self.bat_act_num, (int, float)) and self.bat_act_num == float('inf'):
                # 连续电池控制 - 从连续动作部分获取
                continuous_start = len(action) - (self.bat_num + (self.pv_num * 2 if self.pv_control_enabled else 0))
                bat_actions = action[continuous_start:continuous_start + self.bat_num]
            else:
                # 离散电池控制
                bat_actions = action[action_idx:action_idx+self.bat_num]
                action_idx += self.bat_num
            
            # 获取旧的电池状态用于日志记录
            old_bat_states = {}
            for bat_name in self.bat_names:
                if bat_name in self.circuit.batteries:
                    bat = self.circuit.batteries[bat_name]
                    if hasattr(bat, 'kw'):
                        old_bat_states[bat_name] = bat.kw
                    elif hasattr(bat, 'state'):
                        old_bat_states[bat_name] = bat.state
                    else:
                        old_bat_states[bat_name] = 0
            
            self.circuit.set_all_batteries_before_solve(bat_actions)
            self.str_action += 'Bat Status:'+str(bat_actions)
            
            # 记录电池动作详情
            for i, bat_name in enumerate(self.bat_names):
                if bat_name in self.circuit.batteries:
                    bat = self.circuit.batteries[bat_name]
                    new_state = getattr(bat, 'kw', getattr(bat, 'state', 0))
                    old_state = old_bat_states.get(bat_name, 0)
                    diff = abs(new_state - old_state) if isinstance(new_state, (int, float)) else 0
                    log_device_actions(logger, "Battery", bat_name, old_state, new_state, diff)
        
        ### PV control (如果启用)
        if self.pv_control_enabled and self.pv_num > 0:
            if isinstance(self.action_space, gym.spaces.Tuple) and isinstance(self.pv_act_num, (int, float)) and self.pv_act_num == float('inf'):
                # 连续PV控制 - 从连续动作部分获取
                continuous_start = len(action) - (self.pv_num * 2)
                pv_actions = action[continuous_start:]
                # 将PV动作重新整形为 [pv_num, 2] 格式 (有功功率, 功率因数)
                pv_actions = np.array(pv_actions).reshape(self.pv_num, 2)
            else:
                # 离散PV控制
                pv_actions = action[action_idx:action_idx+self.pv_num]
                action_idx += self.pv_num
            
            # 设置PV系统控制
            self.circuit.set_all_pvs_before_solve(pv_actions)
                            
            self.str_action += 'PV Status:'+str(pv_actions)

        # DSS求解（添加错误处理）
        try:
            logger.debug(f"开始DSS求解 - 时步: {self.t}")
            self.circuit.dss.ActiveCircuit.Solution.Solve()
            
            # 检查求解状态
            converged = self.circuit.dss.ActiveCircuit.Solution.Converged
            if not converged:
                logger.warning(f"DSS求解未收敛 - 时步: {self.t}")
            else:
                logger.debug(f"DSS求解成功收敛 - 时步: {self.t}")
                
        except Exception as e:
            logger.error(f"DSS求解失败: {e}")
            # 返回安全的默认结果
            return self._get_safe_step_result()

        ### update battery kWh. record soc_err and discharge_err
        if self.bat_num>0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        else: soc_errs, dis_errs, bat_statuses = [], [], dict()

        ### update time step
        self.t += 1 
        logger.debug(f"环境步骤完成 - 时步: {self.t}")
 
        ### Update obs ###
        bus_voltages = dict()
        voltage_violations = 0
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0]
            
            # 统计电压违规情况
            for v in bus_voltages[bus_name]:
                if v < 0.95 or v > 1.05:
                    voltage_violations += 1
        
        if voltage_violations > 0:
            logger.warning(f"电压违规节点数: {voltage_violations} - 时步: {self.t}")
        
        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        
        # 更新PV状态（如果启用PV控制）
        if self.pv_control_enabled and self.pv_num > 0:
            pv_statuses = {}
            for pv_name in self.pv_names:
                if pv_name in self.circuit.pvs:
                    pv = self.circuit.pvs[pv_name]
                    # 获取PV系统状态: [有功功率比例, 功率因数]
                    if hasattr(pv, 'get_status'):
                        pv_statuses[pv_name] = pv.get_status()
                    else:
                        # 如果没有get_status方法，使用默认值
                        pv_statuses[pv_name] = [0.5, 1.0]  # 默认50%功率，单位功率因数
                else:
                    pv_statuses[pv_name] = [0.0, 1.0]  # PV系统不存在时的默认值
            self.obs['pv_statuses'] = pv_statuses
        else:
            self.obs['pv_statuses'] = {}
            
        # 使用新的正确方法计算功率损失百分比
        self.obs['power_loss'] = self.circuit.calculate_loss_percentage() / 100.0  # 转换为小数形式
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t%self.horizon].to_dict()

        done = (self.t == self.horizon)

        # 计算PV控制差异（如果启用）
        pv_diffs = []
        if self.pv_control_enabled and self.pv_num > 0:
            # PV diffs are already calculated by the circuit, just pass empty list for now
            # In the future, this can be enhanced to track actual PV control differences
            pass

        reward, info = self.reward_func.composite_reward(capdiff, regdiff,\
                                                         soc_errs, dis_errs, pv_diffs)
        # avoid dividing by zero
        info.update( {'av_cap_err': sum(capdiff)/(self.cap_num+1e-10),
                      'av_reg_err': sum(regdiff)/(self.reg_num+1e-10),
                      'av_dis_err': sum(dis_errs)/(self.bat_num+1e-10),
                      'av_soc_err': sum(soc_errs)/(self.bat_num+1e-10),
                      'av_soc': sum([soc for soc, _ in bat_statuses.values()])/ \
                                   (self.bat_num+1e-10)  })
        
        # 添加logger需要的信息字段
        try:
            total_loss = self.circuit.total_loss()
            total_power = self.circuit.total_power()
            total_load = self.circuit.total_load_power()
            
            info['power_loss_kw'] = total_loss[0] if len(total_loss) > 0 else 0.0
            info['total_power_kw'] = abs(total_power[0]) if len(total_power) > 0 else 100.0
            info['total_load_kw'] = total_load[0] if len(total_load) > 0 else 100.0
            info['power_loss_kvar'] = total_loss[1] if len(total_loss) > 1 else 0.0
            info['total_power_kvar'] = abs(total_power[1]) if len(total_power) > 1 else 50.0
            info['total_load_kvar'] = total_load[1] if len(total_load) > 1 else 50.0
            info['power_loss_ratio'] = self.circuit.calculate_loss_percentage() / 100.0
            info['power_loss_percentage'] = self.circuit.calculate_loss_percentage()  # 添加百分比形式
        except Exception as e:
            logger.warning(f"获取功率信息时出错: {e}")
            info['power_loss_kw'] = 0.0
            info['total_power_kw'] = 100.0
            info['total_load_kw'] = 100.0
            info['power_loss_kvar'] = 0.0 
            info['total_power_kvar'] = 50.0
            info['total_load_kvar'] = 50.0
            info['power_loss_ratio'] = 0.0
            info['power_loss_percentage'] = 0.0
        
        # 电压违规计数和率统计
        info['voltage_violation_count'] = voltage_violations
        
        # 计算单步单bus平均电压违规数
        total_buses = len(bus_voltages) if bus_voltages else 1
        info['voltage_violations_per_bus'] = voltage_violations / max(total_buses, 1)
        
        # 计算电压违规率（违规bus数/总bus数）
        if bus_voltages:
            violated_buses = sum(1 for v in bus_voltages.values() if v < 0.95 or v > 1.05)
            info['voltage_violation_rate'] = violated_buses / total_buses
        else:
            info['voltage_violation_rate'] = 0.0
        
        # PV利用率 - 修复：确保利用率在0-1范围内作为比率
        info['pv_utilization'] = 0.0
        if self.pv_control_enabled and self.pv_num > 0 and 'pv_statuses' in self.obs:
            pv_powers = [status[0] for status in self.obs['pv_statuses'].values() if len(status) > 0]
            if pv_powers:
                # 确保每个功率比率都在0-1范围内，然后转换为百分比
                valid_powers = []
                for power in pv_powers:
                    # 检查并修正异常值
                    if isinstance(power, (int, float)) and not np.isnan(power):
                        corrected_power = min(max(power, 0.0), 1.0)
                        valid_powers.append(corrected_power)
                        if power != corrected_power:
                            logger.warning(f"PV功率比率异常值已修正: {power:.3f} -> {corrected_power:.3f}")
                    else:
                        logger.warning(f"PV功率比率无效值: {power}, 将使用0.0")
                        valid_powers.append(0.0)
                
                if valid_powers:
                    utilization = np.mean(valid_powers)
                    info['pv_utilization'] = utilization  # 保持为0-1的比率
                    logger.debug(f"PV利用率计算: 有效功率比率={valid_powers}, 平均利用率={utilization:.2%}")
                else:
                    info['pv_utilization'] = 0.0
                    logger.warning("没有有效的PV功率比率数据")
        
        # 电池SOC
        info['battery_avg_soc'] = 0.5
        if self.bat_num > 0 and bat_statuses:
            soc_values = [soc for soc, _ in bat_statuses.values()]
            if soc_values:
                info['battery_avg_soc'] = np.mean(soc_values)
        
        #使用无功电压敏感度矩阵
        if self.useS==True:
            self.Y=self.circuit.get_Y_matrix()
            self.agents_bus=self.circuit.get_agent_bus_dict()
            S=self.circuit.get_node_sensity(self.Y)
            # 保留 S 中在 agent_bus_dict 的值中出现的键
            filtered_S = {key: value for key, value in S.items() if any(key in values for values in self.agents_bus.values())}#或者就不按字典写了，直接变成一个数组算辽，数组的每个位置对应一个智能体节点的位置
            info['S']=filtered_S#一个episode算一次更新顺序，然后insert一下
        if self.use_render==True:
            info['bus_voltages']=bus_voltages
            self.agents_bus=self.circuit.get_agent_bus_dict()
            info['agents_bus']=self.agents_bus
            info['powerloss']=self.circuit.total_loss()[0]
            info['powerloss_reward']= -self.circuit.calculate_loss_percentage() / 100.0 * 10
        
        # if self.LLM:
        #     # PV834在34Bus系统中不存在，需要根据实际系统配置修改
        #     # pvpower=self.circuit.dss.ActiveCircuit.CktElements('PVSystem.PV834').TotalPowers[0]
        #     pass
        
        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, done, info
        else:
            return self.obs, reward, done, info
    
    def _get_safe_step_result(self) -> Tuple:
        """获取安全的步进结果（错误时使用）"""
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
            safe_obs = np.zeros(self.observation_space.shape)
        else:
            safe_obs = np.zeros(100)  # 假设观测维度
        safe_info = {"error": True, "vol_reward": 0, "ctrl_reward": 0}
        return safe_obs, 0.0, True, safe_info
    
    def _get_safe_reset_result(self):
        """获取安全的重置结果（错误时使用）"""
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'shape'):
            return np.zeros(self.observation_space.shape)
        else:
            return np.zeros(100)  # 假设观测维度

    @performance_monitor
    def reset(self, load_profile_idx: int = 0, irrad_train: int = 19, hours: str = '14to15/test'):
        """环境重置
        
        Args:
            load_profile_idx: ID number for load profile
            irrad_train: 光伏辐照度训练参数
            hours: 时间参数
        
        Returns:
            observation: wrapped observation
        """
        ###reset time
        self.t = 0
 
        ### choose load profile
        try:
            self.load_profile.select_load_profile(load_profile_idx)
            self.all_load_profiles = self.load_profile.get_load_profile_data(load_profile_idx)
        except Exception as e:
            logger.error(f"负载配置文件选择失败: {e}")
            return self._get_safe_reset_result()
            
        if self.LLM:#在此处引入光伏的可变性。
            try:
                logger.info(f"设置光伏参数: irrad_train={irrad_train}, episode_idx={load_profile_idx}")
                # 优先使用新的统一方法
                pv_success = self.load_profile.select_pv_temperature_profile(load_profile_idx)
                if not pv_success:
                    # 如果新方法失败，回退到原来的方法
                    logger.warning("使用原始光伏配置方法")
                    self.load_profile.select_irradiance_profile(style=irrad_train,hours=hours)
            except Exception as e:
                logger.warning(f"光伏配置失败: {e}")
                
        ### re-compile dss and reset batteries
        try:
            self.circuit.reset()
        except Exception as e:
            logger.error(f"电路重置失败: {e}")
            return self._get_safe_reset_result()

        ### node voltages
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0]
        self.obs['bus_voltages'] = bus_voltages

        ### status of capacitor
        cap_statuses = {name:cap.status for name, cap in self.circuit.capacitors.items()}
        self.obs['cap_statuses'] = cap_statuses
        
        ### status of regulator
        reg_statuses = {name:reg.tap for name, reg in self.circuit.regulators.items()}
        self.obs['reg_statuses'] = reg_statuses

        ### status of battery
        bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        self.obs['bat_statuses'] = bat_statuses
        
        ### status of PV systems (如果启用PV控制)
        if self.pv_control_enabled and self.pv_num > 0:
            pv_statuses = {}
            for pv_name in self.pv_names:
                if pv_name in self.circuit.pvs:
                    pv = self.circuit.pvs[pv_name]
                    # 初始化PV状态: [有功功率比例, 功率因数]
                    if hasattr(pv, 'get_status'):
                        pv_statuses[pv_name] = pv.get_status()
                    else:
                        pv_statuses[pv_name] = [0.5, 1.0]  # 初始值: 50%功率, 单位功率因数
                else:
                    pv_statuses[pv_name] = [0.0, 1.0]  # PV系统不存在时的默认值
            self.obs['pv_statuses'] = pv_statuses
        else:
            self.obs['pv_statuses'] = {}

        ### total power loss - 使用正确的计算方法
        self.obs['power_loss'] = self.circuit.calculate_loss_percentage() / 100.0  # 转换为小数形式
        
        ### time step tracker
        self.obs['time'] = self.t

        ### load for current timestep
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t].to_dict()

        ### Edge weight
        #self.obs['Y_matrix'] = self.circuit.edge_weight

        if self.wrap_observation:
            return self.wrap_obs(self.obs).astype(np.float32)
        else:
            return self.obs
    
    def dss_step(self):
        assert self.circuit.dss_act == True, 'Env.circuit.dss_act must be True'

        ### update time step
        prev_states = self.circuit.get_all_capacitor_statuses()
        prev_tapnums = self.circuit.get_all_regulator_tapnums()
        
        self.circuit.dss.ActiveCircuit.Solution.Solve()
        
        self.t += 1 
        cap_statuses = self.circuit.get_all_capacitor_statuses()
        reg_statuses = self.circuit.get_all_regulator_tapnums()
        capdiff = np.array([abs(prev_states[c]-cap_statuses[c]) for c in prev_states])
        regdiff = np.array([abs(prev_tapnums[r]-reg_statuses[r]) for r in prev_tapnums])

        # OpenDSS does not control batteries
        soc_errs, dis_errs, bat_statuses = [], [], dict()

        ### Update obs ###
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0]
        
        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        # 使用新的正确方法计算功率损失百分比
        self.obs['power_loss'] = self.circuit.calculate_loss_percentage() / 100.0  # 转换为小数形式
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t%self.horizon].to_dict()

        done = (self.t == self.horizon)

        # 为dss_step方法添加空的PV差异列表（因为dss_step不处理PV控制）
        pv_diffs = []

        reward, info = self.reward_func.composite_reward(capdiff, regdiff,\
                                                         soc_errs, dis_errs, pv_diffs)
        # avoid dividing by zero
        info.update( {'av_cap_err': sum(capdiff)/(self.cap_num+1e-10),
                      'av_reg_err': sum(regdiff)/(self.reg_num+1e-10),
                      'av_dis_err': sum(dis_errs)/(self.bat_num+1e-10),
                      'av_soc_err': sum(soc_errs)/(self.bat_num+1e-10),
                      'av_soc': sum([soc for soc, _ in bat_statuses.values()])/ \
                                   (self.bat_num+1e-10)  })
        
        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, done, info
        else:
            return self.obs, reward, done, info

    def wrap_obs(self, obs):
        """ Wrap the observation dictionary (i.e., self.obs) to a numpy array - 支持CRBP架构
        
        Attribute:
            obs: the observation dictionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']
        
        # 添加PV状态（如果启用PV控制）
        if self.pv_control_enabled and self.pv_num > 0:
            key_obs.append('pv_statuses')
            
        if self.observe_load: key_obs.append('load_profile_t')

        mod_obs = []
        for var_dict in key_obs:
            # node voltage is a dict of dict, we only take minimum phase node voltage
            #if var_dict == 'bus_voltages': 
            #    for values in obs[var_dict].values():
            #        mod_obs.append(min(values))
            if var_dict in \
                ['bus_voltages','cap_statuses','reg_statuses', 'bat_statuses', 'pv_statuses', 'load_profile_t']:
                mod_obs = mod_obs + list(obs[var_dict].values())
            elif var_dict == 'power_loss_ratio':
                mod_obs.append(obs['power_loss_ratio'])
        return np.hstack(mod_obs)

    def get_obs(self, wrapped: bool = True):
        """
        返回当前环境的观测值。

        参数:
            wrapped (bool):
                - True:  返回扁平化后的 numpy 数组；
                - False: 返回原始的 obs 字典，包含 bus_voltages、cap_statuses、reg_statuses、bat_statuses 等所有字段。

        返回:
            numpy.ndarray 或 dict
        """
        if wrapped:
            # 与 reset()/step() 返回格式一致
            return self.wrap_obs(self.obs).astype(np.float32)
        else:
            # 原始观测字典
            return self.obs

    @performance_monitor
    def build_graph(self):
        """构建网络图
        
        Returns:
            Graph: Network graph
        """
        self.lines = dict()
        self.circuit.dss.ActiveCircuit.Lines.First
        while(True):
            bus1 = self.circuit.dss.ActiveCircuit.Lines.Bus1.split('.', 1)[0].lower()
            bus2 = self.circuit.dss.ActiveCircuit.Lines.Bus2.split('.', 1)[0].lower()
            line_name = self.circuit.dss.ActiveCircuit.Lines.Name.lower()
            self.lines[line_name] = (bus1, bus2)
            if self.circuit.dss.ActiveCircuit.Lines.Next==0:
                break

        transformer_names = self.circuit.dss.ActiveCircuit.Transformers.AllNames
        self.transformers = dict()
        for transformer_name in transformer_names:
            self.circuit.dss.ActiveCircuit.SetActiveElement('Transformer.' + transformer_name)
            buses = self.circuit.dss.ActiveCircuit.ActiveElement.BusNames
            #assert len(buses) == 2, 'Transformer {} has more than two terminals'.format(transformer_name)
            bus1 = buses[0].split('.', 1)[0].lower()
            bus2 = buses[1].split('.', 1)[0].lower()
            self.transformers[transformer_name] = (bus1, bus2)

        self.edges = [frozenset(edge) for _, edge in self.transformers.items()] + [frozenset(edge) for _, edge in self.lines.items()]
        if len(self.edges) != len(set(self.edges)):
            print('There are ' + str(len(self.edges)) + ' edges and ' + str(len(set(self.edges))) + ' unique edges. Overlapping transformer edges')

        self.circuit.topology.add_edges_from(self.edges)
        
        return self.circuit.topology

    def plot_graph(self, node_bound='minimum', 
                   vmin=0.95, vmax=1.05, 
                   cmap='jet', figsize=(18,12), 
                   text_loc_x=0, text_loc_y=400,
                   node_size=None, shift=None,
                   show_node_labels=None,
                   show_voltages=True,
                   show_controllers=True,
                   show_actions=False):
        """Function to plot system graph with voltage as node intensity
        
        Args:
            node_bound (str): Determine to plot max/min node voltage for nodes with more than one phase
            vmin (float): Min heatmap intensity
            vmax (float): Max heatmap intensity
            cmap (str): Colormap
            figsize (tuple): Figure size
            text_loc_x (int): x-coordinate for timestamp
            text_loc_y (int): y-coordinate for timestamp
            node_size (int): Node size. If None, initialize with environment setting
            shift (int): shift of node label. If None, initialize with environment setting
            show_node_labels (bool): show node label. If None, initialize with environment setting
            show_voltages (bool): show voltages
            show_controllers (bool): show controllers
            show_actions (bool): show actions
        
        Returns:
            fig: Matplotlib figure
            pos: dictionary of node positions
            
        """
        node_size = self.node_size if node_size is None else node_size
        shift = self.shift if shift is None else shift
        show_node_labels = self.show_node_labels if show_node_labels is None else show_node_labels
        
        #get normalized node voltages
        voltages, nodes = [], []
        pos = dict()

        assert node_bound in ['maximum', 'minimum'], 'invalid node_bound'
        for busname in self.all_bus_names:
            self.circuit.dss.Circuits.SetActiveBus(busname)
            if not self.circuit.dss.Circuits.Buses.Coorddefined: continue
            x = self.circuit.dss.Circuits.Buses.x
            y = self.circuit.dss.Circuits.Buses.y

            pos[busname] = (x,y)
            nodes.append(busname)
            bus_volts = [self.circuit.dss.Circuits.Buses.puVmagAngle[i] for i in range(len(self.circuit.dss.Circuits.Buses.puVmagAngle)) if i%2==0]
            if node_bound == 'minimum':
                voltages.append(min(bus_volts))
            elif node_bound == 'maximum':
                voltages.append(max(bus_volts))

        fig = plt.figure(figsize=figsize)
        graph = nx.Graph()

        # local lines, transformers and edges
        HasLocation = lambda p: (p[0] in pos and p[1] in pos)
        loc_lines = [pair for pair in self.lines.values() if HasLocation(pair)]
        loc_trans = [pair for pair in self.transformers.values() if HasLocation(pair)]

        graph.add_edges_from(loc_lines + loc_trans)
        nx.draw_networkx_edges(graph, pos, loc_lines, edge_color='k', width=3, label='lines')
        nx.draw_networkx_edges(graph, pos, loc_trans, edge_color='r', width=3, label='transformers')
        if show_voltages:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=voltages, vmin=vmin, vmax=vmax, cmap=cmap, node_size=node_size)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm)
        else:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=np.ones(len(voltages)), vmin=vmin, vmax=vmax, cmap=cmap, node_size=node_size)

        if show_node_labels:
            node_labels = {node:node for node in pos}
            nx.draw_networkx_labels(graph, pos, labels= node_labels, font_size=15)

        # show source bus
        loc={self.source_bus:(pos[self.source_bus][0]+shift, pos[self.source_bus][1]-shift)}
        nx.draw_networkx_labels(graph, loc, labels={self.source_bus:'src'}, font_size=15)

        if show_controllers:
            if self.cap_num>0:
                labels = {self.circuit.capacitors[cap].bus1:'cap' for cap in self.cap_names}
                labels = {k:v for k,v in labels.items() if k in pos } # remove missing pos
                loc = {bus:(pos[bus][0]+shift,pos[bus][1]+shift) for bus in labels.keys()}
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15, 
                                        font_color='darkorange')
            if self.bat_num>0:
                labels = {self.circuit.batteries[bat].bus1:'bat' for bat in self.bat_names}
                labels = {k:v for k,v in labels.items() if k in pos } # remove missing pos
                loc = {bus:(pos[bus][0]+shift,pos[bus][1]+shift) for bus in labels.keys()}
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15, 
                                        font_color='darkviolet')
            if self.reg_num>0:
                regs = self.circuit.regulators
                labels = {(regs[r].bus1, regs[r].bus2):'reg' for r in self.reg_names}
                # accept if one of the edge's node is in pos
                labels = {k:v for k,v in labels.items() if (k[0] in pos or k[1] in pos) }
                
                loc = dict()
                for key in labels.keys():
                    b1, b2 = key
                    lx, ly, count = 0.0, 0.0, 0
                    for b in list(key):
                        if b in pos:
                            ll = pos[b]
                            lx, ly, count = lx+ll[0], ly+ll[1], count+1
                    lx, ly = lx/count, ly/count
                    loc[key] = (lx + shift, ly + shift)
                nx.draw_networkx_labels(graph, loc, labels=labels, font_size=15, 
                                        font_color='darkred')


        
        if show_actions:
            plt.text(text_loc_x, text_loc_y, s='t='+str(self.t)+' Action: '+ self.str_action, 
                     fontsize=18)
        elif show_voltages:
            plt.text(text_loc_x, text_loc_y, s='t='+str(self.t), fontsize=18)

        return fig, pos

    def seed(self, seed):
        self.ActionSpace.seed(seed)

    def random_action(self):
        """Samples random action
        
        Returns:
            Array: Random control actions
        """
        return self.ActionSpace.sample()

    def dummy_action(self):
        return [1]*self.cap_num + \
               [self.reg_act_num]*self.reg_num + \
               [0.0 if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num==np.inf else int(self.bat_act_num)//2]*self.bat_num
        
    def load_base_kW(self):
        '''
        get base kW of load objects.
        see class Load in circuit.py for details on Load.feature
        '''
        basekW = dict()
        for load in self.circuit.loads.keys():
            basekW[load[5:]] = self.circuit.loads[load].feature[1]
        return basekW
    

# 日志信息
logger.info("PowerZoo环境类初始化完成")
