# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import os
import gym
import numpy as np
from envs.powerzoo.powerzoo.circuit import Circuits
from envs.powerzoo.powerzoo.loadprofile import LoadProfile
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

#### helper functions ####

def plotting(env, profile, episode_step, show_voltages=True):
    """ Plot network status with a load profile at an episode step
    
    Args:
        env (obj): the environment object
        profile (int): the load profile number
        episode_step (int): the step number in the episode
        show_voltages (bool): show voltages or not
    """
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd,'plots')):
        os.makedirs(os.path.join(cwd,'plots'))
    
    fig, _ = env.plot_graph(show_voltages=show_voltages)
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(cwd,'plots/' + str(profile).zfill(3) +'_'+ str(episode_step) + '.png'))
    plt.close()

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
    def __init__(self, CRB_num, RB_act_num):
        self.cap_num, self.reg_num, self.bat_num = CRB_num#三元组，包括了电容器数量，调压器数量和电池数量，如果要改为多智能体模式，则应该给每个智能体一个actionspace
        #我的想法是造三个类，电容器类，电池类和调压器类，或者就把他们写死
        self.reg_act_num, self.bat_act_num = RB_act_num#二元组，包含了调压器和电池的动作数量

        if self.bat_act_num < float('inf'):
            # discrete battery，把电容器，电源，调压器的动作拼接起来
            self.space = gym.spaces.MultiDiscrete(\
                           [2]*self.cap_num + \
                           [self.reg_act_num]*self.reg_num + \
                           [self.bat_act_num]*self.bat_num      )
        else:
            # continuous battery
            self.space = gym.spaces.Tuple((\
               gym.spaces.MultiDiscrete([2]*self.cap_num + [self.reg_act_num]*self.reg_num),\
               gym.spaces.Box(low=-1, high=1, shape=(self.bat_num,)) ))

    def sample(self):#联合采样
        ss = self.space.sample()
        if self.bat_act_num == np.inf:
            return np.concatenate(ss)#如果电池是连续动作，就把他们拼接在一起
        return ss#否则都是离散动作不用拼接

    def seed(self, seed):
        self.space.seed(seed)

    def dim(self):#用于返回动作空间的维数
        if self.bat_act_num == np.inf:
            return self.space[0].shape[0] + self.space[1].shape[0]
        return self.space.shape[0]

    def CRB_num(self):
        return self.cap_num, self.reg_num, self.bat_num

    def RB_act_num(self):
        return self.reg_act_num, self.bat_act_num

#### environment class ####
class Env(gym.Env):

    """训练 RL 代理的环境   
    Attributes:
        obs (dict): 系统的观测/状态
        dss_folder_path (str): 包含DSS文件的文件夹路径
        dss_file (str): DSS仿真文件名
        source_bus (str): 距离电源最近的母线（在BusCoords.csv中有坐标）
        node_size (int): 绘图中节点的大小
        shift (int): 绘图中标签的偏移量
        show_node_labels (bool): 是否在绘图中显示节点标签
        scale (float): 负荷特性的比例
        wrap_observation (bool): 是否在reset和step的输出中将观测展平为数组
        observe_load (bool): 是否在观测中包含节点负荷
        load_profile (obj): 负荷特性管理类
        num_profiles (int): load_profile生成的不同特性的数量
        horizon (int): 每个episode的最大步数
        circuit (obj): 连接到DSS仿真的电路对象
        all_bus_names (list): 系统中所有母线的名称
        cap_names (list): 电容器母线列表
        reg_names (list): 调节器母线列表
        bat_names (list): 电池母线列表
        cap_num (int): 电容器数量
        reg_num (int): 调节器数量
        bat_num (int): 电池数量
        reg_act_num (int): 调节器控制动作的数量
        bat_act_num (int): 电池控制动作的数量
        topology (graph): 电力系统的NxGraph
        reward_func (obj): 奖励函数类
        t (int): 环境状态的当前时间步
        ActionSpace (obj): 动作空间类。用于采样随机动作
        action_space (gym.spaces): 来自ActionSpace类的基动作空间
        observation_space (gym.spaces): 环境的观测空间。

        
    在self.step()和self.reset()中定义:
        all_load_profiles (dict): 所有母线和时间的负荷特性二维数组
    
    在self.step()中定义并在self.plot_graph()中使用:
        self.str_action: 在self.plot_graph()中打印的动作字符串
        
    在self.build_graph()中定义:
        edges (dict): 连接电路中节点的边字典
        lines (dict): 电路中包含组件的边字典
        transformers (dict): 系统中变压器的字典

    """  
    def __init__(self, folder_path, info, dss_act=False):
        super().__init__()
        self.obs = dict()
        self.dss_folder_path = os.path.join(folder_path, info['system_name'])
        self.dss_file = info['dss_file']
        self.source_bus = info['source_bus']
        self.node_size = info['node_size']
        self.shift = info['shift']
        self.show_node_labels = info['show_node_labels']
        self.scale = info['scale'] if 'scale' in info else 1.0
        self.wrap_observation = True
        self.observe_load = False
        self.use_load_noise=info['load_noise']
        
        #添加了智能体节点与智能体名称的对应关系
        self.agents_bus=dict()
        
        # 生成负载配置文件
        self.load_profile = LoadProfile(\
                 info['max_episode_steps'],
                 self.dss_folder_path,
                 self.dss_file,
                 self.use_load_noise,
                 worker_idx = info['worker_idx'] if 'worker_idx' in info else None)

        self.num_profiles = self.load_profile.gen_loadprofile(use_noise=self.use_load_noise,scale=self.scale)
        # choose a dummy load profile for the initialization of the circuit
        self.load_profile.choose_loadprofile(0,self.use_load_noise)
        
        # 问题范围是负载曲线的长度
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
                                        (self.reg_act_num, self.bat_act_num) )
        self.action_space = self.ActionSpace.space
        self.reset_obs_space()

        self.record_node = True
        #TODO:S修改，在此添加条件判断
        self.useS=False
        self.use_render=False
        self.agents_bus=self.circuit.get_agent_bus_dict()
        
    def reset_obs_space(self, wrap_observation=True, observe_load=False):
        '''
        reset the observation space based on the option of wrapping and load.
        
        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        
        根据打包选项和负荷选项重置观测空间。
        建议通过此函数设置wrap_observation和observe_load，而不是直接从属性（例如，Env.wrap_observation）设置。
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
        """Reward definition class
        
        Attributes:
            env (obj): Inherits all attributes of environment 
        """
        def __init__(self, env, info):
            self.env = env
            self.power_w = info['power_w']
            self.cap_w = info['cap_w']
            self.reg_w = info['reg_w']
            self.soc_w = info['soc_w']
            self.dis_w = info['dis_w']

        def powerloss_reward(self):

            # 整个系统在某一时间步powerloss的惩罚

            #loss = self.env.circuit.total_loss()[0] # a postivie float
            #gen = self.env.circuit.total_power()[0] # a negative float
            ratio = max(0.0, min(1.0, self.env.obs['power_loss_ratio']) )
            return -ratio * self.power_w

        def ctrl_reward(self, capdiff, regdiff, soc_err, discharge_err):
            # 错误动作处罚
            ## capdiff: abs(current_cap_state - new_cap_state)
            ## regdiff: abs(current_reg_tap_num - new_reg_tap_num)
            ## soc_err: abs(soc - initial_soc)
            ## discharge_err: max(0, kw) / max_kw
            ### discharge_err > 0 means discharging
            cost =  self.cap_w * sum(capdiff) + \
                    self.reg_w * sum(regdiff) + \
                    (0.0 if self.env.t != self.env.horizon else self.soc_w * sum(soc_err)) + \
                    self.dis_w * sum(discharge_err)
            return -cost

        def voltage_reward(self, record_node = False):

            # 节点电压超出 [0.95, 1.05] 范围的惩罚
            violated_nodes = []
            total_violation_num = 0
            for name, voltages in self.env.obs['bus_voltages'].items():
                max_penalty = min(0, 1.05 - max(voltages)) #penalty is negative if above max
                min_penalty = min(0, min(voltages) - 0.95) #penalty is negative if below min
                total_violation_num += (max_penalty + min_penalty)
                if record_node and (max_penalty != 0 or min_penalty != 0):
                    violated_nodes.append(name)
            return total_violation_num, violated_nodes
        
        def get_powerloss_info(self):
            """
            返回功率损耗相关的物理值。

            Returns:
                dict: 包含功率损耗比和总损耗值的字典。
            """
            # 获取功率损耗比（0 到 1 之间）
            power_loss_ratio = max(0.0, min(1.0, self.env.obs['power_loss_ratio']))
            # 获取总功率损耗值（假设可以从环境获取 total_loss）
            total_loss = self.env.circuit.total_loss()[0]  # 正值，单位可以是 kW
            # 获取总发电功率值（假设可以从环境获取 total_power）
            total_power = -self.env.circuit.total_power()[0]  # 负值，单位可以是 kW
            return {
                "power_loss_ratio": power_loss_ratio,
                "total_power_loss": total_loss,
                "total_generation_power": total_power
            }
            
        def composite_reward(self, cd, rd, soc, dis, record_node=True):
            """
            计算当前状态的综合奖励。

            参数:
                cd: 控制差异指标
                rd: 调节差异指标
                soc: 电池荷电状态指标
                dis: 放电指标
                full_info (bool): 如果为 True，返回的信息中包含详细的奖励组成部分。
                record_node (bool): 如果为 True，返回的信息中包含违规节点的相关信息。

            返回:
                summ (float): 总奖励值。
                info (dict): 奖励的组成部分以及可选的节点违规详情。
            """
            # 计算各部分奖励
            p = self.powerloss_reward()  # 功率损耗奖励
            v, vio_nodes = self.voltage_reward(record_node)  # 电压相关的奖励与违规节点
            t = self.ctrl_reward(cd, rd, soc, dis)  # 控制相关奖励
            summ = p + v + t  # 综合奖励为各部分奖励的总和

            # 初始化信息字典
            info = {'violated_nodes': vio_nodes} if record_node else {}
            
            try:
                # 添加详细的奖励组成部分
                info.update({
                    'power_loss_ratio': -p / (self.power_w or 1e-6),  # 防止除以零
                    'vol_reward': v,
                    'ctrl_reward': t
                })
            except ZeroDivisionError:
                raise ValueError("self.power_w 不能为零，无法计算 power_loss_ratio")

            return summ, info

    def step(self, action):
        """执行环境的一步操作，并调用 OpenDSS 求解器更新状态。

        Args:
            action [array]: 包含电容器、调压器和电池的整数数组，每个元素表示对应设备的控制指令。
        
        Returns:
            tuple: (观测值, 奖励, 是否结束标志, 附加信息字典)
            - self.wrap_obs(self.obs): 当前时间步的观测值（可能被包装）。
            - reward: 当前时间步的奖励（float）。
            - done: 当前 episode 是否结束（bool）。
            - info: 附加信息字典，包含奖励组成和设备状态误差等。
        """
        action_idx = 0
        self.str_action = ''  # 用于记录当前动作字符串，便于后续打印和调试。

        ### 电容器控制 ###
        if self.cap_num > 0:  # 如果存在电容器
            # 提取动作中与电容器相关的部分
            statuses = action[action_idx:action_idx + self.cap_num]
            # 更新所有电容器状态，返回状态变化量
            capdiff = self.circuit.set_all_capacitor_statuses(statuses)
            # 构建电容器状态字典，key 为电容器名称，value 为状态
            cap_statuses = {cap: status for cap, status in zip(self.circuit.capacitors.keys(), statuses)}
            action_idx += self.cap_num  # 更新动作索引位置
            self.str_action += 'Cap Status:' + str(statuses)  # 添加到动作字符串
        else:  # 如果没有电容器
            capdiff, cap_statuses = [], dict()

        ### 调压器控制 ###
        if self.reg_num > 0:  # 如果存在调压器
            # 提取动作中与调压器相关的部分
            tapnums = action[action_idx:action_idx + self.reg_num]
            # 更新所有调压器的分接头位置，返回状态变化量
            regdiff = self.circuit.set_all_regulator_tappings(tapnums)
            # 构建调压器状态字典，key 为调压器名称，value 为当前 tap 值
            reg_statuses = {reg: self.circuit.regulators[reg].tap for reg in self.reg_names}
            action_idx += self.reg_num  # 更新动作索引位置
            self.str_action += 'Reg Tap Status:' + str(tapnums)  # 添加到动作字符串
        else:  # 如果没有调压器
            regdiff, reg_statuses = [], dict()

        ### 电池控制 ###
        if self.bat_num > 0:  # 如果存在电池
            # 提取动作中与电池相关的部分
            states = action[action_idx:]
            # 在解算前设置所有电池的目标状态
            self.circuit.set_all_batteries_before_solve(states)
            self.str_action += 'Bat Status:' + str(states)  # 添加到动作字符串

        ### 调用 OpenDSS 解算器 ###
        self.circuit.dss.ActiveCircuit.Solution.Solve()  # 触发潮流计算，更新系统状态

        ### 更新电池状态并记录误差 ###
        if self.bat_num > 0:  # 如果存在电池
            # 设置所有电池的实际状态，返回 SOC 和放电误差
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            # 构建电池状态字典，包含 SOC 和放电功率比
            bat_statuses = {name: [bat.soc, -1 * bat.actual_power() / bat.max_kw] for name, bat in self.circuit.batteries.items()}
        else:  # 如果没有电池
            soc_errs, dis_errs, bat_statuses = [], [], dict()

        ### 更新时间步 ###
        self.t += 1  # 增加当前时间步计数

        ### 更新观测值 ###
        bus_voltages = dict()  # 存储总线电压
        for bus_name in self.all_bus_names:
            # 获取指定总线的电压值
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            # 仅保留奇数索引的电压值（相当于相电压）
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i % 2 == 0]
        
        # 更新观测值字典
        self.obs['bus_voltages'] = bus_voltages  # 总线电压
        self.obs['cap_statuses'] = cap_statuses  # 电容器状态
        self.obs['reg_statuses'] = reg_statuses  # 调压器状态
        self.obs['bat_statuses'] = bat_statuses  # 电池状态
        self.obs['power_loss_ratio'] = - self.circuit.total_loss()[0] / self.circuit.total_power()[0]  # 功率损耗比
        # # 电路的总损耗，并将结果存储在self.obs字典中
        # self.obs['power_loss_kw']= self.circuit.total_loss()[0]
        # self.obs['power_loss_kvar']= self.circuit.total_loss()[1]
        # # 电路的总功率，并将结果存储在self.obs字典中
        # self.obs['total_power_kw']= self.circuit.total_power()[0]
        # self.obs['total_power_kvar']= self.circuit.total_power()[1]
        
        self.obs['time'] = self.t  # 当前时间步
        if self.observe_load:  # 如果观察负载
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t % self.horizon].to_dict()  # 当前时间步的负载信息

        ### 判断是否结束 ###
        done = (self.t == self.horizon)  # 当前时间步是否达到最大时间步

        ### 奖励和信息计算 ###
        reward, info = self.reward_func.composite_reward(capdiff, regdiff, soc_errs, dis_errs, 
                                                         record_node=self.record_node)  # 计算奖励和附加信息

        # 更新附加信息字典
        info.update({
            'av_cap_err': sum(capdiff) / (self.cap_num + 1e-10),  # 平均电容器误差
            'av_reg_err': sum(regdiff) / (self.reg_num + 1e-10),  # 平均调压器误差
            'av_dis_err': sum(dis_errs) / (self.bat_num + 1e-10),  # 平均放电误差
            'av_soc_err': sum(soc_errs) / (self.bat_num + 1e-10),  # 平均 SOC 误差
            'av_soc': sum([soc for soc, _ in bat_statuses.values()]) / (self.bat_num + 1e-10),  # 平均 SOC
            'capacitor_ctrl':sum(capdiff),
            'regulator_ctrl':sum(regdiff),
            'discharge_ctrl':sum(dis_errs)
        })
        # self.obs['power_loss_kw']= self.circuit.total_loss()[0]
        # self.obs['power_loss_kvar']= self.circuit.total_loss()[1]
        # # 电路的总功率，并将结果存储在self.obs字典中
        # self.obs['total_power_kw']= self.circuit.total_power()[0]
        # self.obs['total_power_kvar']= self.circuit.total_power()[1]
        
        info['power_loss_kw']= self.circuit.total_loss()[0]
        info['power_loss_kvar']= self.circuit.total_loss()[1]
        info['total_power_kw']= self.circuit.total_power()[0]
        info['total_power_kvar']= self.circuit.total_power()[1]
        
        

        ### 可选：计算无功电压敏感度矩阵 ###
        if self.useS:
            Y = self.circuit.get_Y_matrix_acc()  # 获取 Y 矩阵
            self.agents_bus = self.circuit.get_agent_bus_dict()  # 获取智能体对应总线
            S = self.circuit.get_node_sensity_acc(Y)  # 计算无功电压敏感度
            # 过滤敏感度矩阵，仅保留与智能体相关的节点
            filtered_S = {key: value for key, value in S.items() if any(key in values for values in self.agents_bus.values())}
            info['S'] = filtered_S  # 将过滤后的敏感度矩阵加入信息字典

        ### 可选：记录更多可视化信息 ###
        if self.use_render:
            info['bus_voltages'] = bus_voltages  # 总线电压
            self.agents_bus = self.circuit.get_agent_bus_dict()  # 智能体总线
            info['agents_bus'] = self.agents_bus  # 加入智能体总线信息
            info['powerloss'] = self.circuit.total_loss()[0]  # 功率损耗

        ### 返回结果 ###
        if self.wrap_observation:  # 如果需要包装观测值
            return self.wrap_obs(self.obs), reward, done, info
        else:  # 否则直接返回原始观测值
            return self.obs, reward, done, info

    def reset(self, load_profile_idx=0):
        """Reset state of enviroment for new episode
        
        Args:
            load_profile_idx (int, optional): ID number for load profile
        
        Returns:
            numpy array: wrapped observation
        """
        ###reset time
        self.t = 0
 
        ### choose load profile
        self.load_profile.choose_loadprofile(load_profile_idx,self.use_load_noise)
        self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)
        
        ### re-compile dss and reset batteries
        self.circuit.reset()

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
        self.obs['power_loss_ratio'] = -self.circuit.total_loss()[0]/self.circuit.total_power()[0]
        
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
            return self.obs.astype(np.float32)
    
    def dss_step(self):
        """执行环境的一个时间步更新，通过 OpenDSS 求解器更新系统状态。

        Returns:
            tuple: (观测值, 奖励, 是否结束标志, 附加信息字典)
            - self.wrap_obs(self.obs): 当前时间步的观测值（可能被包装）。
            - reward: 当前时间步的奖励（float）。
            - done: 当前 episode 是否结束（bool）。
            - info: 附加信息字典，包含奖励组成和设备状态误差等。
        """
        # 确保 OpenDSS 模式已激活
        assert self.circuit.dss_act == True, 'Env.circuit.dss_act must be True'

        ### 更新时间步之前的状态 ###
        # 获取当前电容器的状态
        prev_states = self.circuit.get_all_capacitor_statuses()
        # 获取当前调压器的分接头状态
        prev_tapnums = self.circuit.get_all_regulator_tapnums()

        ### 调用 OpenDSS 求解器 ###
        # 调用 OpenDSS 的解算器，计算潮流并更新系统状态
        self.circuit.dss.ActiveCircuit.Solution.Solve()

        ### 更新时间步计数 ###
        self.t += 1  # 当前时间步加 1

        ### 更新设备状态 ###
        # 获取更新后的电容器状态
        cap_statuses = self.circuit.get_all_capacitor_statuses()
        # 获取更新后的调压器分接头状态
        reg_statuses = self.circuit.get_all_regulator_tapnums()
        # 计算电容器状态变化量（绝对值差异）
        capdiff = np.array([abs(prev_states[c] - cap_statuses[c]) for c in prev_states])
        # 计算调压器分接头变化量（绝对值差异）
        regdiff = np.array([abs(prev_tapnums[r] - reg_statuses[r]) for r in prev_tapnums])

        ### 电池状态更新（未控制） ###
        # OpenDSS 不控制电池，因此电池相关状态为空
        soc_errs, dis_errs, bat_statuses = [], [], dict()

        ### 更新观测值 ###
        bus_voltages = dict()  # 初始化总线电压字典
        for bus_name in self.all_bus_names:  # 遍历所有总线
            # 获取指定总线的电压值
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            # 仅保留奇数索引（相电压）
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i % 2 == 0]
        
        # 将总线电压和其他状态信息存入观测值字典
        self.obs['bus_voltages'] = bus_voltages  # 总线电压
        self.obs['cap_statuses'] = cap_statuses  # 电容器状态
        self.obs['reg_statuses'] = reg_statuses  # 调压器状态
        self.obs['bat_statuses'] = bat_statuses  # 电池状态
        
        self.obs['power_loss_ratio'] = - self.circuit.total_loss()[0] / self.circuit.total_power()[0]  # 功率损耗比
        # 电路的总损耗，并将结果存储在self.obs字典中
        self.obs['power_loss_kw']= self.circuit.total_loss()[0]
        self.obs['power_loss_kvar']= self.circuit.total_loss()[1]
        # 电路的总功率，并将结果存储在self.obs字典中
        self.obs['total_power_kw']= self.circuit.total_power()[0]
        self.obs['total_power_kvar']= self.circuit.total_power()[1]
        
        self.obs['time'] = self.t  # 当前时间步
        if self.observe_load:  # 如果需要观察负载
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t % self.horizon].to_dict()  # 当前时间步负载数据

        ### 判断是否结束 ###
        # 判断当前时间步是否达到最大时间步
        done = (self.t == self.horizon)

        ### 计算奖励和附加信息 ###
        # 计算奖励值和附加信息，主要基于电容器和调压器的变化量
        reward, info = self.reward_func.composite_reward(capdiff, regdiff, soc_errs, dis_errs)

        # 更新附加信息字典，包括各项平均误差和电池状态
        info.update({
            'av_cap_err': sum(capdiff) / (self.cap_num + 1e-10),  # 平均电容器误差
            'av_reg_err': sum(regdiff) / (self.reg_num + 1e-10),  # 平均调压器误差
            'av_dis_err': sum(dis_errs) / (self.bat_num + 1e-10),  # 平均放电误差
            'av_soc_err': sum(soc_errs) / (self.bat_num + 1e-10),  # 平均 SOC 误差
            'av_soc': sum([soc for soc, _ in bat_statuses.values()]) / (self.bat_num + 1e-10)  # 平均 SOC
        })

        ### 返回结果 ###
        if self.wrap_observation:  # 如果需要包装观测值
            return self.wrap_obs(self.obs), reward, done, info
        else:  # 否则直接返回原始观测值
            return self.obs, reward, done, info

    def wrap_obs(self, obs):
        """ Wrap the observation dictionary (i.e., self.obs) to a numpy array
        
        Attribute:
            obs: the observation distionary generated at self.reset() and self.step()
        
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

    def build_graph(self):
        """Constructs a NetworkX graph for downstream use
        
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
