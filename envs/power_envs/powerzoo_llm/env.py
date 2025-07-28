
import os
import gym
import numpy as np
from envs.power_envs.powerzoo_llm.circuit import Circuits
from envs.power_envs.powerzoo_llm.loadprofile import LoadProfile
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
    from envs.power_envs.powezoo_llm.utils import get_logger
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

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


def FFT_selection(vio_nodes, dist_matrix, k=10):
    '''
    Farthest first traversal to select batteries from the violated nodes.
    和choose_batteries函数配合选择电池位置
    Arguments:
        vio_nondes (list): bus names with puVoltage<0.95
        dist_matrix (np.array): the pairwise distance matrix of the violated nodes
        k (int): number of batteries

    Returns:
        list of the names of the chosen nodes
    '''
    assert k>1, 'invalid k'
    if len(vio_nodes)<=1: return vio_nodes

    # for >=2 number of violated nodes
    # random initial point
    chosen = [ np.random.randint(len(vio_nodes)) ]
    
    # construct dist_map and the second point
    dist_map = dict()
    max_dist = p = 0
    for i in range(len(vio_nodes)):
        if i != chosen[-1]:
            dist = dist_matrix[i,chosen[-1]]
            dist_map[i] = dist
            if dist > max_dist:
                max_dist = dist
                p = i
    del dist_map[p]
    chosen.append(p)
    
    for kk in range(2, k):
        if len(dist_map)==0: break
            
        # update 'dist_map', 'p'
        max_dist = p = 0
        for pt, val in dist_map.items():
            dist = min( val, dist_matrix[pt,chosen[-1]])
            if dist < val:
                dist_map[pt] = dist
            if dist > max_dist:
                max_dist = dist
                p = pt
        del dist_map[p]
        chosen.append(p)
    return [vio_nodes[c] for c in chosen]

def choose_batteries(env, k=10, on_plot=True, node_bound='minimum'):
    '''
    Choose battery locations
    
    Arguments:
        env (obj): the environment object
        k (int): number of battery to allocate
        on_plot (bool): allocate battery on the nodes shown in the pos
        node_bound (str): Determine to plot max/min node voltage for nodes with more than one phase

    Returns:
        list of the names of the chosen nodes
    '''
    assert node_bound in ['minimum','maximum'], 'invalid node_bound'
    
    graph = nx.Graph()
    graph.add_edges_from(list(env.lines.values()) + list(env.transformers.values()))
    lens = dict( nx.shortest_path_length(graph) )

    if node_bound == 'minimum':
        nv = {bus: min(volts) for bus, volts in env.obs['bus_voltages'].items()}
    else:
        nv = {bus: max(volts) for bus, volts in env.obs['bus_voltages'].items()}
    
    if on_plot: 
        _, pos = env.plot_graph(show_voltages=False)
        nv = {bus:volts for bus, volts in nv.items() if bus in pos}

    vio_nodes = [bus for bus, vol in nv.items() if vol<0.95]
    dist = np.zeros((len(vio_nodes), len(vio_nodes)))
    for i, b1 in enumerate(vio_nodes):
        for j, b2 in enumerate(vio_nodes):
            dist[i,j] = lens[b1][b2]
        
    choice = FFT_selection(vio_nodes, dist, k)
    return choice


def get_basekv(env, buses):
    #buses = ['l3160098', 'l3312692', 'l3091052', 'l3065696', 'l3235247', 'l3066804', 'l3251854', 'l2785537', 'l2839331', 'm1069509']
    ans = []
    for busname in buses:
        env.circuit.dss.Circuits.SetActiveBus(busname)
        ans.append( env.circuit.dss.Circuits.Buses.kVBase )
    print(ans)
    


#### action space class ####
class ActionSpace:
    '''Action Space Wrapper for Capacitors, Regulators, and Batteries

    Attributes:
        cap_num, reg_num, bat_num (int): number of capacitors, regulators, and batteries.
        reg_act_num, bat_act_num: number of actions for regulators and batteries.
        space (gym.spaces): the space object from gym

    Note:
        space is MultiDiscrete if using the discrete battery;
        otherwise, space is a tuple of MultiDiscrete and Box
    '''
    def __init__(self, CRB_num: Tuple[int, int, int], RB_act_num: Tuple[int, int], optimization_level: str = "standard"):
        self.cap_num, self.reg_num, self.bat_num = CRB_num
        self.reg_act_num, self.bat_act_num = RB_act_num
        self.optimization_level = optimization_level
        
        # 缓存空间对象避免重复创建（优化功能）
        self._space = None
        self._initialize_space()
    
    def _initialize_space(self):
        """初始化动作空间"""
        if self.bat_act_num < float('inf'):
            # discrete battery，把电容器，电源，调压器的动作拼接起来
            self._space = gym.spaces.MultiDiscrete(
                [2] * self.cap_num + # 电容器动作 
                [self.reg_act_num] * self.reg_num + # 调压器动作
                [self.bat_act_num] * self.bat_num # 电池动作
            )
        else:
            # continuous battery
            self._space = gym.spaces.Tuple((
                gym.spaces.MultiDiscrete([2] * self.cap_num + [self.reg_act_num] * self.reg_num),
                gym.spaces.Box(low=-1, high=1, shape=(self.bat_num,))
            ))
    
    @property
    def space(self):
        return self._space

    def sample(self):
        """优化的采样方法"""
        ss = self._space.sample()
        if self.bat_act_num == np.inf:
            return np.concatenate(ss)
        return ss

    def seed(self, seed: int):
        self._space.seed(seed)

    def dim(self) -> int:
        """返回动作空间维度"""
        if self.bat_act_num == np.inf:
            return self._space[0].shape[0] + self._space[1].shape[0]
        return self._space.shape[0]

    def CRB_num(self) -> Tuple[int, int, int]:
        return self.cap_num, self.reg_num, self.bat_num

    def RB_act_num(self) -> Tuple[int, int]:
        return self.reg_act_num, self.bat_act_num

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
    def __init__(self, folder_path: str, info: Dict[str, Any], dss_act: bool = False, 
                 optimization_level: str = "standard"):
        super().__init__()
        
        # 基础配置
        self.obs = {}
        self.optimization_level = optimization_level
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
        self.irrad_dss = info.get('irrad_dss', None)
        
        # 性能缓存（优化功能）
        self._performance_cache = {} if optimization_level in ["high", "maximum"] else None
        
        #TODO:添加了智能体节点与智能体名称的对应关系
        self.agents_bus=dict()
        
        # generate load profile files
        self.load_profile = LoadProfile(\
                 info['max_episode_steps'],
                 self.dss_folder_path,
                 self.dss_file,
                 worker_idx = info['worker_idx'] if 'worker_idx' in info else None,
                 irrad_dss=irrad_dss if self.LLM else None)

        self.num_profiles = self.load_profile.gen_loadprofile(scale=self.scale)
        # choose a dummy load profile for the initialization of the circuit
        self.load_profile.choose_loadprofile(0)#在这里得到loadprofile的路径
        
        # problem horizon is the length of load profile
        self.horizon = info['max_episode_steps']
        self.reg_act_num = info['reg_act_num']
        self.bat_act_num = info['bat_act_num']
        assert self.horizon>=1, 'invalid horizon'
        assert self.reg_act_num>=2 and self.bat_act_num>=2, 'invalid act nums'
        
        self.circuit = Circuits(os.path.join(self.dss_folder_path, self.dss_file),
                                RB_act_num=(self.reg_act_num, self.bat_act_num),
                                dss_act=dss_act)
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.cap_names = list(self.circuit.capacitors.keys())
        self.reg_names = list(self.circuit.regulators.keys())
        self.bat_names = list(self.circuit.batteries.keys())
        self.cap_num = len(self.cap_names)
        self.reg_num = len(self.reg_names)
        self.bat_num = len(self.bat_names)
        assert self.cap_num>=0 and self.reg_num>=0 and self.bat_num>=0 and \
               self.cap_num + self.reg_num + self.bat_num>=1,'invalid CRB_num'
        
        self.topology = self.build_graph()
        self.reward_func = self.MyReward(self, info)
        self.t = 0
        
        # create action space and observation space
        self.ActionSpace = ActionSpace( (self.cap_num, self.reg_num, self.bat_num),
                                        (self.reg_act_num, self.bat_act_num),
                                        optimization_level=optimization_level )
        self.action_space = self.ActionSpace.space
        self.reset_obs_space()
        self.useS=False
        self.use_render=False
        self.Y=self.circuit.get_Y_matrix()
        self.agents_bus=self.circuit.get_agent_bus_dict()
        #TODO:S修改，在此添加条件判断
    def reset_obs_space(self, wrap_observation=True, observe_load=False):
        '''
        reset the observation space based on the option of wrapping and load.
        
        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        
        '''
        self.wrap_observation = wrap_observation
        self.observe_load = observe_load
        
        self.reset(load_profile_idx=0)
        #nnode = len(self.obs['bus_voltages'])
        nnode = len(np.hstack( list(self.obs['bus_voltages'].values()) ))
        if observe_load: nload = len(self.obs['load_profile_t'])
        
        if self.wrap_observation:
            low, high = [0.8]*nnode, [1.2]*nnode  # add voltage bound
            low, high = low+[0]*self.cap_num, high+[1]*self.cap_num # add cap bound
            low, high = low+[0]*self.reg_num, high+[self.reg_act_num]*self.reg_num # add reg bound
            low, high = low+[0,-1]*self.bat_num, high+[1,1]*self.bat_num # add bat bound
            if observe_load: low, high = low+[0.0]*nload, high+[1.0]*nload # add load bound
            low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low, high) 
        else:
            bat_dict = {bat: gym.spaces.Box(np.array([0,-1]), np.array([1,1]), dtype=np.float32) 
                        for bat in self.obs['bat_statuses'].keys()}
            obs_dict = {
                'bus_voltages': gym.spaces.Box(0.8, 1.2, shape=(nnode,)),
                'cap_statuses': gym.spaces.MultiDiscrete([2]*self.cap_num),
                'reg_statuses': gym.spaces.MultiDiscrete([self.reg_act_num]*self.cap_num),
                'bat_statuses': gym.spaces.Dict(bat_dict)
            }
            if observe_load: obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
            self.observation_space = gym.spaces.Dict(obs_dict)

    class MyReward:
        """Reward definition class（集成优化功能）
        
        Attributes:
            env (obj): Inherits all attributes of environment 
        """
        def __init__(self, env, info: Dict[str, Any]):
            self.env = env
            self.power_w = info.get('power_w', 10.0)
            self.cap_w = info.get('cap_w', 0.1)
            self.reg_w = info.get('reg_w', 0.1)
            self.soc_w = info.get('soc_w', 0.5)
            self.dis_w = info.get('dis_w', 0.1)
            
            # 缓存上一次的计算结果（优化功能）
            self._last_power_loss = None
            self._last_voltages = None

        @performance_monitor
        def powerloss_reward(self) -> float:
            """功率损耗奖励（带缓存）"""
            current_loss = self.env.obs.get('power_loss', 0)
            if self._last_power_loss == current_loss:
                return -current_loss * self.power_w
            
            self._last_power_loss = current_loss
            ratio = max(0.0, min(1.0, current_loss))
            return -ratio * self.power_w

        def ctrl_reward(self, capdiff: List[float], regdiff: List[float], 
                       soc_err: List[float], discharge_err: List[float]) -> float:
            """控制动作奖励"""
            cost = (self.cap_w * sum(capdiff) + 
                    self.reg_w * sum(regdiff) + 
                    (0.0 if self.env.t != self.env.horizon else self.soc_w * sum(soc_err)) + 
                    self.dis_w * sum(discharge_err))
            return -cost

        @performance_monitor
        def voltage_reward(self, record_node: bool = False) -> Tuple[float, List[str]]:
            """电压奖励（优化版）"""
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
                            full: bool = True, record_node: bool = False) -> Tuple[float, Dict[str, Any]]:
            """综合奖励计算"""
            p = self.powerloss_reward()
            v, vio_nodes = self.voltage_reward(record_node)
            t = self.ctrl_reward(cd, rd, soc, dis)
            
            summ = t + v  # 只包含控制和电压奖励
            
            info = {} if not record_node else {'violated_nodes': vio_nodes}
            if full:
                info.update({
                    'power_loss_ratio': -p / self.power_w,
                    'vol_reward': v,
                    'ctrl_reward': t
                })
            
            return summ, info

    @performance_monitor
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """优化的环境步进
        
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
            #print("self.cap_num.statues=",statuses)#TODO:打印电容数量
            capdiff = self.circuit.set_all_capacitor_statuses(statuses)
            cap_statuses = {cap:status for cap, status in \
                            zip(self.circuit.capacitors.keys(), statuses)}
            action_idx += self.cap_num
            self.str_action += 'Cap Status:'+str(statuses)
        else: capdiff, cap_statuses = [], dict()

        ### regulator control
        if self.reg_num>0:
            tapnums = action[action_idx:action_idx+self.reg_num]
            regdiff = self.circuit.set_all_regulator_tappings(tapnums)
            reg_statuses = {reg:self.circuit.regulators[reg].tap \
                            for reg in self.reg_names}
            action_idx += self.reg_num
            self.str_action += 'Reg Tap Status:'+str(tapnums)
        else: regdiff, reg_statuses = [], dict()

        ### battery control
        if self.bat_num>0:
            states = action[action_idx:]
            self.circuit.set_all_batteries_before_solve(states)
            self.str_action += 'Bat Status:'+str(states)

        # DSS求解（添加错误处理）
        try:
            self.circuit.dss.ActiveCircuit.Solution.Solve()
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
 
        ### Update obs ###
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0]
        
        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        self.obs['power_loss'] = - self.circuit.total_loss()[0]/self.circuit.total_power()[0]
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t%self.horizon].to_dict()

        done = (self.t == self.horizon)

        reward, info = self.reward_func.composite_reward(capdiff, regdiff,\
                                                         soc_errs, dis_errs)
        # avoid dividing by zero
        info.update( {'av_cap_err': sum(capdiff)/(self.cap_num+1e-10),
                      'av_reg_err': sum(regdiff)/(self.reg_num+1e-10),
                      'av_dis_err': sum(dis_errs)/(self.bat_num+1e-10),
                      'av_soc_err': sum(soc_errs)/(self.bat_num+1e-10),
                      'av_soc': sum([soc for soc, _ in bat_statuses.values()])/ \
                                   (self.bat_num+1e-10)  })
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
            info['powerloss_reward']=- self.circuit.total_loss()[0]/self.circuit.total_power()[0]*10
        
        if self.LLM:
            pvpower=self.circuit.dss.ActiveCircuit.CktElements('PVSystem.PV834').TotalPowers[0]
        
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
        """优化的环境重置
        
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
            self.load_profile.choose_loadprofile(load_profile_idx)
            self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)
        except Exception as e:
            logger.error(f"负载配置文件选择失败: {e}")
            return self._get_safe_reset_result()
            
        if self.LLM:#在此处引入光伏的可变性。
            try:
                logger.info(f"设置光伏参数: irrad_train={irrad_train}")
                self.load_profile.choose_irrad_profile(style=irrad_train,hours=hours)
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

        ### total power loss
        self.obs['power_loss'] = -self.circuit.total_loss()[0]/self.circuit.total_power()[0]
        
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
        self.obs['power_loss'] = - self.circuit.total_loss()[0]/self.circuit.total_power()[0]
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t%self.horizon].to_dict()

        done = (self.t == self.horizon)

        reward, info = self.reward_func.composite_reward(capdiff, regdiff,\
                                                         soc_errs, dis_errs)
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
        """ Wrap the observation dictionary (i.e., self.obs) to a numpy array
        
        Attribute:
            obs: the observation dictionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']
        if self.observe_load: key_obs.append('load_profile_t')

        mod_obs = []
        for var_dict in key_obs:
            # node voltage is a dict of dict, we only take minimum phase node voltage
            #if var_dict == 'bus_voltages': 
            #    for values in obs[var_dict].values():
            #        mod_obs.append(min(values))
            if var_dict in \
                ['bus_voltages','cap_statuses','reg_statuses', 'bat_statuses', 'load_profile_t']:
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
        """构建网络图（优化版）
        
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
        # print(len(self.circuit.topology.nodes))
        # print(self.circuit.topology.nodes)
        # print(len(self.circuit.topology.edges))
        # print(self.circuit.topology.edges)

        # self.adj_mat = nx.adjacency_matrix(self.circuit.topology)
        # print(self.adj_mat.todense())
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
               [0.0 if self.bat_act_num==np.inf else self.bat_act_num//2]*self.bat_num
        
    def load_base_kW(self):
        '''
        get base kW of load objects.
        see class Load in circuit.py for details on Load.feature
        '''
        basekW = dict()
        for load in self.circuit.loads.keys():
            basekW[load[5:]] = self.circuit.loads[load].feature[1]
        return basekW
    
    # === 优化级别配置方法 ===
    def set_optimization_level(self, level: str):
        """设置优化级别
        
        Args:
            level: 优化级别 ("minimal", "standard", "high", "maximum")
        """
        valid_levels = ["minimal", "standard", "high", "maximum"]
        if level not in valid_levels:
            raise ValueError(f"Invalid optimization level: {level}. Must be one of {valid_levels}")
        
        self.optimization_level = level
        
        # 根据优化级别调整缓存设置
        if level in ["high", "maximum"]:
            if self._performance_cache is None:
                self._performance_cache = {}
        else:
            self._performance_cache = None
        
        logger.info(f"优化级别设置为: {level}")


# 日志信息
logger.info("PowerZoo环境类已成功集成优化功能")
