# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT
import os
import gymnasium as gym

from envs.lfc.IEEE9BusSystem.loadprofile import LoadProfile#这里有一个相对路径的问题，python不能直接引用父目录的文件，当前的地址为LFC_gym文件
from envs.lfc.IEEE9BusSystem.opendss_python import circuit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 模块目录：.slx 模型与 sampled_180s_data.csv 所在
import time
from datetime import datetime
import pandas as pd

class LFCEnv(gym.Env): 
    '''
    action_space:version1:loadshedding ratio(-0.15,0)把所有这些打包进matlab的相同工作区的不同列
    observation_space:设计观察空间(首先假设所建模型并非局部观察)
    generator active power:(3个电机)pu
    mechanical power:pu(3个电机)pu
    bus voltage:使用rms获取所有节点功率的有效值pu
    load power:上一时刻的负载功率(三个负载,有功and无功)
    节点频率：可加可不加
    和频率调节强相关的量
    horizon (int): Maximum steps in a episode
    选择采样频率为100hz即10ms0.01s(load shedding,二次调频可以选择为3-5s),假设仿真时长为10s,则最大环境步数为1000步。
    '''    
    def __init__(self,info,ranks):#初始化动作空间和观察空间
        #在定义action_space前，需要先引入opendss的模块
        #目前我还没特别想好observation_space的设定，先设定为3个电机功率，3个电机机械功率，3个节点电压，3个节点频率，3个负载功率，配网节点光伏节点的有功输出以及最大有功输出,暂时先不加电压
        self.obs = dict()
        assert info.get('dss_folder_path'), (
            "dss_folder_path 未设置：lfc.yaml 默认为 null（P0-4），"
            "运行时必须通过 --dss_folder_path 或配置显式传入 DSS 案例目录")
        self.dss_folder_path=info['dss_folder_path']
        self.dss_file=info['dss_file']
        self.horizon = info['max_episode_steps']
        
        if ranks is not None:
            base_file = os.path.join(self.dss_folder_path, info['dss_file'])
            assert os.path.exists(base_file), base_file + ' does not exist'
            # 确保原始 DSS 文件存在，如果不存在抛出断言错误
            with open(base_file, 'r') as fin, open(base_file[:-4] + '_' + str(ranks) + '.dss', 'w') as fout:
            # 创建一个新的 DSS 文件，文件名基于 worker_idx 添加后缀
            # 例如: 如果 base_file 为 "system.dss"，则新文件为 "system_<worker_idx>.dss"
            
                for line in fin:
                    # 遍历原始 DSS 文件的每一行
                    if line.strip() == 'redirect loadshape.dss':
                        # 如果当前行是重定向到负载曲线文件的指令
                        fout.write('redirect loadshape_' + str(ranks) + '.dss\n')
                        # 修改指令，将其重定向到工作者专属的负载曲线文件
                        # 例如: "redirect loadshape.dss" 改为 "redirect loadshape_0.dss"（假设 worker_idx 为 0）
                    elif line.strip() == 'redirect irrad_up_down.dss':
                        fout.write('redirect irrad_up_down_' + str(ranks) + '.dss\n')
                    else:
                        fout.write(line)
                        # 其他行保持不变，直接写入新的 DSS 文件
                        
        dss_file_copy=self.dss_file
        dss_file_copy = dss_file_copy[:-4] + '_' + str(ranks) + '.dss'
        self.load_profile = LoadProfile(\
                 info['max_episode_steps'],
                 self.dss_folder_path,
                 dss_file_copy,
                 worker_idx = ranks)

        

        self.num_profiles = self.load_profile.gen_loadprofile()
        # choose a dummy load profile for the initialization of the circuit
        self.load_profile.choose_loadprofile(ranks)#在这里得到loadprofile的路径
        self.circuit=circuit(os.path.join(info['dss_folder_path'],dss_file_copy))
        
        # self.cap_names = list(self.circuit.capacitors.keys())
        # self.reg_names = list(self.circuit.regulators.keys())
        # self.bat_names = list(self.circuit.batteries.keys())
        # self.cap_num = len(self.cap_names)
        # self.reg_num = len(self.reg_names)
        # self.bat_num = len(self.bat_names)
        
        self.pv_names = list(self.circuit.pvs.keys())
        self.pv_num = len(self.pv_names)

        
        
        self.bat_names = list(self.circuit.batteries.keys())
        self.bat_num = len(self.bat_names)
        # create action space and observation space
        self.ActionSpace = ActionSpace(self.pv_num,self.bat_num)
        #定义观察空间，先定义统一的观察空间，包括发电机有功（3）、转速功率（3）、gsp点的频率（1）、联络线功率（1）、节点有功pu（3）、pv出力（4）
        self.genp_num=3
        self.genw_num=3
        self.Tload_num=3
        self.gspfreq_num=1
        self.tielinep=1#联络线功数量
        self.irrad_num=1
        
        self.action_space = self.ActionSpace.space
        
        
        self.long_term_load_disturbance=info['long_term_load_disturbance']
        
        self.use_irrad=info['use_irrad']
        self.use_render=info['use_render']
        self.irrad_style=info['irrad_style']
        self.render_fault=info['render_fault']
        
        
        self.reset_obs_space()#这里调用了reset

        self.state = None#状态空间
        self.env_name = info['env_name']
        self.horizon = info['max_episode_steps']#确定一个episode的最大环境步数
        assert self.horizon>=1, 'invalid horizon'
        self.t = 0
        self.f_set=60#hz
        self.load_lamda=0.01
        self.load1p=100
        self.load1q=35e6
        
        self.load2p=90
        self.load2q=30e6
        
        self.load3p=125
        self.load3q=50e6
        
        self.sample_probability=0
        self.loadp=np.array([self.load1p,self.load2p,self.load3p])
        
        import matlab.engine  # 惰性导入：无 MATLAB 的环境（CI）可正常 import 本模块

        self.eng = matlab.engine.start_matlab()
        # P0-10：原实现在此 os.chdir 进程目录后启动引擎；改为引擎侧 cd，
        # 模型按名加载行为不变，进程工作目录不再被污染。
        self.eng.cd(_SCRIPT_DIR, nargout=0)
        
        # P0-5：worker→Simulink 模型名映射显式化（原 elif 链缺 else，非法 rank 直接 NameError）
        # rank 0/4（4=评估 worker）用基础模型名；1/2 用 _1/_2 副本；其它 rank 直接报错
        _rank_suffix = {0: "", 1: "_1", 2: "_2", 4: ""}
        assert ranks in _rank_suffix, f"不支持的 worker rank: {ranks}（可用: {sorted(_rank_suffix)}）"
        simulink_name = info['env_name'] + _rank_suffix[ranks]
            
        self.no_rl=False
        self.env_name=simulink_name
        self.eng.load_system(self.env_name)
        #获取记录reward_parameters的参数
        # self.f_deviation=float(info["reward_parameters"]["f_deviation"])
        # self.total_violation=float(info["reward_parameters"]["total_violation"])

        # self.allocate_err=float(info["reward_parameters"]["allocate_err"])
        # self.pv_output_smooth=float(info["reward_parameters"]["pv_output_smooth"])
        # self.bat_output_smooth=float(info["reward_parameters"]["bat_output_smooth"])
        
        self.f1=float(info["reward_parameters"]["f1"])
        self.f2=float(info["reward_parameters"]["f2"])
        self.f3=float(info["reward_parameters"]["f3"])
        self.f4=float(info["reward_parameters"]["f4"])
        self.f5=float(info["reward_parameters"]["f5"])
        self.f6=float(info["reward_parameters"]["f6"])
        self.f7=float(info["reward_parameters"]["f7"])
        
        #self.dis_errs=float(info["reward_parameters"]["dis_errs"]) _optional
        
        #self.f_deviation:1000,用来评价gsp频率接近1pu的程度
        #self.total_violation:1000,用来描述gsp节点的频率越限程度
        #self.allocate_err:0.000001,用来限制使得光伏出力保持在其出力能力的中间，确保对每个光伏设备的最大利用
        #self.pv_output_smooth：0.001，用来平滑光伏输出
        #self.bat_output_smooth：0.00001，用来平滑电池输出
        
        
        #todo:20241224,把是否使用辐照度，以及辐照度的类型放到config文件中，应该把故障的类型也弄出来
        
        if self.long_term_load_disturbance:
            self.dis_load=[]
            df = pd.read_csv(os.path.join(_SCRIPT_DIR, "sampled_180s_data.csv"))  # P0-10
            self.dis_load = (df.iloc[:, 1]).tolist()
        
        
        

    def opendss_reset(self, load_profile_idx=0,use_irrad=False,irrad_style=None,use_render=False):
        """Reset state of enviroment for new episode
        
        Args:
            load_profile_idx (int, optional): ID number for load profile
        
        Returns:
            numpy array: wrapped observation
        """
        ###reset time
        self.t = 0
 
        ### choose load profile
        self.load_profile.choose_loadprofile(load_profile_idx)
        self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)
        
        #TODO: reset时会选择是否使用可变辐照度，以及可变辐照度的类型，应该把这个参数放在lfc文件的config之中。
        ##chose irrad profile
        if use_irrad:
            options = ['Irrad_down', 'Irrad_up_down', 'Irrad_up']
            self.style=random.choice(options)
        else:
            self.style='Irrad_constant'
        if use_render:
            self.style=irrad_style
        self.load_profile.choose_irrad_profile(self.style)
        
        
        
        
        ### re-compile dss and reset batteries
        self.circuit.reset()

        
        ### status of pvs
        pv_statuses = {name:[pv.pmpp, pv.kW/pv.pmpp] for name, pv in self.circuit.pvs.items()}
        self.obs['pv_statuses'] = pv_statuses
        
        ### status of battery
        bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        self.obs['bat_statuses'] = bat_statuses
        
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t].to_dict()

    
    
    
    def reset_obs_space(self, wrap_observation=True, observe_load=False):#定义观察空间，先定义统一的观察空间，包括发电机有功（3）、机械功率（3）、节点有功pu（3）、gsp点的频率（1）pv出力（4）
        '''
        reset the observation space based on the option of wrapping and load.
        
        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        
        '''
        self.wrap_observation = wrap_observation
        self.observe_load = observe_load
        
        self.opendss_reset(load_profile_idx=0,use_irrad=self.use_irrad,irrad_style=self.irrad_style,use_render=self.use_render)
        #nnode = len(self.obs['bus_voltages'])
        # nnode = len(np.hstack( list(self.obs['bus_voltages'].values()) ))
        # if observe_load: nload = len(self.obs['load_profile_t'])
        
        if self.wrap_observation:#此处封装状态空间，包括发电机

            low, high = [-2.0]*self.genp_num, [2.0]*self.genp_num  # add 发电机有功输出的边界pu--3，固定的常数
            low, high = low+[-2.0]*self.genw_num, high+[2]*self.genw_num # add 发电机转速输出的边界pu--3，固定的常数
            low, high = low+[-2]*self.Tload_num, high+[2]*self.Tload_num # add 输电网负荷有功功率大小--3，固定的常数
            #low, high = low+[0.8]*self.gspfreq_num, high+[1.2]*self.gspfreq_num # add 频率观测点--1，固定的常数
            low, high = low+[-3]*self.gspfreq_num, high+[3]*self.gspfreq_num # add 频率观测点--1，固定的常数
            #20250219修改f的观察为delta f
            #13bus省略了tieline的观测，因为我认为它对系统频率是无意义的
            #low, high = low+[0]*self.tielinep, high+[2]*self.tielinep # add tieline传输线功率--1，区间在6mW左右
            
            #low, high = low+[-10]*self.pv_num, high+[0]*self.pv_num #添加pv的实际有功输出，单位为10mw--4
            low, high = low+[-1]*self.pv_num, high+[0]*self.pv_num #2025-1-3我想把光伏的实际有功输出替换为光伏出力占最大的比例
            #if self.use_irrad:
            low, high = low+[-1]*self.pv_num, high+[0]*self.pv_num #添加pv的最大输出，单位为10mw--4，当不用irrad时可以去掉
            
            low, high = low+[-1]*self.bat_num, high+[1]*self.bat_num #添加储能电池的实际有功输出
            low, high = low+[0]*self.bat_num, high+[1]*self.bat_num #添加储能电池的soc
            # if self.use_irrad:
            #     low, high = low+[0]*self.irrad_num, high+[1]*self.irrad_num #添加irrad
            low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low, high)
            print("self.observation_space:",self.observation_space) 
        # else:
        #     bat_dict = {bat: gym.spaces.Box(np.array([0,-1]), np.array([1,1]), dtype=np.float32) 
        #                 for bat in self.obs['bat_statuses'].keys()}
        #     obs_dict = {
        #         'bus_voltages': gym.spaces.Box(0.8, 1.2, shape=(nnode,)),
        #         'cap_statuses': gym.spaces.MultiDiscrete([2]*self.cap_num),
        #         'reg_statuses': gym.spaces.MultiDiscrete([self.reg_act_num]*self.cap_num),
        #         'bat_statuses': gym.spaces.Dict(bat_dict)
        #     }
        #     if observe_load: obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
        #     self.observation_space = gym.spaces.Dict(obs_dict)

    
    
    def load_genrator(self):
        self.loadp=np.array([100,90,125])
        mean1 = 100  # 均值
        mean2=90
        mean3=125
        std_dev = 5  # 标准差
        self.loadp[0]=np.random.normal(loc=mean1, scale=std_dev, size=1)[0]
        self.loadp[1]=np.random.normal(loc=mean2, scale=std_dev, size=1)[0]
        self.loadp[2]=np.random.normal(loc=mean3, scale=std_dev, size=1)[0]
        return self.loadp
    
    
    def _wait_until_paused(self, timeout=600):
        """P0-11：等待 Simulink 到达暂停点。

        原实现为无休眠无超时的忙等循环，模型一旦卡住/报错，Python 侧即 100% CPU 死循环挂死。
        现加入 10ms 轮询间隔与超时异常；轮询语义与原实现一致（状态仍由 get_param 读取）。
        """
        t0 = time.time()
        while True:
            if self.eng.get_param(self.env_name, 'SimulationStatus') == "paused":
                return
            if time.time() - t0 > timeout:
                status = self.eng.get_param(self.env_name, 'SimulationStatus')
                raise RuntimeError(
                    f"等待 Simulink 暂停超时（{timeout}s），当前状态: {status}"
                )
            time.sleep(0.01)

    def step(self, action):#在这一部分，传入的action是各类设备的控制动作
        """Steps through one step of enviroment and calls DSS solver after one control action
        Args:
            action [array]: Integer array of actions for capacitors, regulators and batteries and loads
        
        Returns:
            self.wrap_obs(self.obs), reward, done, all_reward, all_diff
            next wrapped observation (array), reward (float), Done (bool), all rewards (dict), all state errors (dict)
        """
        #整个动作流程分为两步：一是把动作送入opendss，获得单步的与主网连接点处的有功功率和无功功率
        #目前的action仅为所有光伏的pmpp，4
        #step第一步，把action的动作分解到每一个设备上（
        # 1.获取当前负荷的有功无功大小
        # 2.使用动作调整当前的负荷值
        # 3.传输值调用simulink进行仿真
        pv_err=[]
        bat_err=[]
        pvs_ob=[]
        pvs_actual_power=[]
        
        pvs_max_ob=[]
        pvs_reward=[]
        
        bat_ob=[]
        soc_ob=[]
        
        bat_actual_power=[]
        info={}
        
        action_idx = 0
        self.str_action = '' # the action string to be printed at self.plot_graph()
        action1=action.reshape(-1)  
        ### pvs control
        if self.pv_num>0:
            action2 = action1[action_idx:action_idx+self.pv_num]
            #此处就暂时不设置光伏逆变器pmpp的动作误差了
            pctPmpps=np.clip(action2, 0, 1) 
            self.circuit.set_all_pvs_before_solve(pctPmpps)
            pv_pctPmpps = {pv:pct for pv, pct in \
                            zip(self.circuit.pvs.keys(), pctPmpps)}
            action_idx += self.pv_num
            self.str_action += 'pv pctpmpp:'+str(pctPmpps)
        else: pv_pctPmpps = dict()  # P0-13：原为 [], dict() 生成元组
        
          ### battery control
        if self.bat_num>0:
            states = action1[action_idx:]
            self.circuit.set_all_batteries_before_solve(states)
            self.str_action += 'Bat Status:'+str(states)

        
        
        self.circuit.dss.ActiveCircuit.Solution.Solve()
        
        
        ### update battery kWh. record soc_err and discharge_err
        if self.bat_num>0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        else: soc_errs, dis_errs, bat_statuses = [], [], dict()
        
        for name, bat in self.circuit.batteries.items():
            batpower=-1*bat.actual_power()/bat.max_kw#放电此项观察为正
            soc=bat.soc
            bat_ob.append(batpower)#单位为10mw
            soc_ob.append(soc)
            bat_actual_power.append(bat.actual_power()/10000)#是个正值，单位是10mw,actual_power小于0意味着放电
        
        bat_average_output=np.mean(bat_ob)
        bat_allocate_reward = np.mean(np.abs(bat_ob - bat_average_output))*2
        
        #update the gsp node load demand
        Vsource_power=self.circuit.dss.ActiveCircuit.CktElements('Vsource.SOURCE').TotalPowers
        
        #光伏的实验先缓缓，光伏做的时候就考虑一种故障类型，负载增加之类，只考虑光伏的波动性，9.26就先把所有的render跑一跑，出出图
        irradnow=self.circuit.get_all_pvs_irrad_after_solve()#此时我需要的是什么？光伏的实际输出和最大输出，最大输出也作为obs的输入
        
        for name in self.pv_names:
            pv_ob=self.circuit.dss.ActiveCircuit.CktElements(name).TotalPowers[0]
            #pv_act_reward=abs(pv_ob/(-self.circuit.pvs[name].max_output)-0.5)
            #pvs_reward.append(pv_act_reward)
            #pvs_ob.append(pv_ob/10000)#单位为10mw
            pvs_ob.append(pv_ob/self.circuit.pvs[name].max_output)#目前是和最大输出的比值
            pvs_max_ob.append(-self.circuit.pvs[name].max_output/100000)#单位为100mw
            
            pvs_actual_power.append(pv_ob/10000)#单位为10mw
        
        pv_average_output=np.mean(pvs_ob)
        pvs_reward = np.mean(np.abs(pvs_ob - pv_average_output))*2
        
        
        self._wait_until_paused()  # P0-11（原为无超时忙等）
        if self.t==self.horizon-1:
            done=True
        else:
            done=False

        
        if self.long_term_load_disturbance:
            if self.t%10==0:
                dis_load=self.dis_load[self.t//10]
                self.eng.set_param(self.env_name+'/loadbus6p', 'value',f"{90e6*dis_load}", nargout=0)
                self.eng.set_param(self.env_name+'/loadbus6q', 'value',f"{30e6*dis_load}", nargout=0)
        
        
        if self.t==10:
            sample=None
            if self.sample[0]==3:#load bus6p change
                
                sample = np.random.uniform(low=0.5, high=1.5, size=1)
                self.sample_probability=sample[0]
                if self.use_render:
                    sample[0]=1.4
                self.eng.set_param(self.env_name+'/loadbus6p', 'value',f"{90e6*sample[0]}", nargout=0)
                self.eng.set_param(self.env_name+'/loadbus6q', 'value',f"{30e6*sample[0]}", nargout=0)
            elif self.sample[0]==4:#load bus5p change
                sample = np.random.uniform(low=-0.5, high=0.5, size=1)
                self.sample_probability=sample[0]
                if self.use_render:
                    sample[0]=-0.1
                self.eng.set_param(self.env_name+'/loadbus5p1', 'value',f"{125e6*sample[0]}", nargout=0)
                self.eng.set_param(self.env_name+'/loadbus5q1', 'value',f"{50e6*sample[0]}", nargout=0)
            elif self.sample[0]==1:#节点4和节点6之间的线路断开
                self.eng.set_param(self.env_name+'/Step46', 'value',"1", nargout=0)
            elif self.sample[0]==2:#节点8和节点9之间的线路断开
                self.eng.set_param(self.env_name+'/Step89', 'value',"1", nargout=0)
            else:pass#loadbus8change暂时没选，因为和光伏接在一起，不好直接影响
        
        
        self.eng.set_param(self.env_name+'/load1p', 'value',f"{-Vsource_power[0]*1000}", nargout=0)
        self.eng.set_param(self.env_name+'/load1q', 'value',f"{-Vsource_power[1]*1000}", nargout=0)
            
        self.eng.set_param(self.env_name+'/pause_time', 'value', f"{1+self.t}", nargout=0) # 设置下次暂停时间,4s为一个暂停的间隔
        self.eng.set_param(self.env_name, 'SimulationCommand', 'continue', nargout=0)#继续运行
        self._wait_until_paused()  # P0-11（原为无超时忙等）
        #读取当前状态下的st+1，以及奖励r
        gen_obtemp=np.array(self.eng.workspace['gen_ob'])#读取后转为ndarray数组
        load_ob_temp=np.array(self.eng.workspace['load_ob'])
        gsp_obtemp=np.array(self.eng.workspace['out_f_load1'])
        #tieline_ob_temp=np.array(self.eng.workspace['tieline_ob'])
        
        gen_temp=gen_obtemp[-1]
        gen1_ob=gen_temp[0]/69400000
        gen2_ob=gen_temp[1]/163000000
        gen3_ob=gen_temp[2]/85000000
        
        genp_ob=np.array([gen1_ob,gen2_ob,gen3_ob]).reshape(3,1)
        genw_ob=np.array(gen_temp[3:]).reshape(3,1)
        gen_ob=np.concatenate((genp_ob,genw_ob))
        
        
        
        load_temp=load_ob_temp[-1]#laodp=0,2,4
        load1_ob=load_temp[0]/100000000+1
        load2_ob=load_temp[1]/90000000
        load3_ob=load_temp[2]/125000000
        load_ob=np.array([load1_ob,load2_ob,load3_ob]).reshape(3,1)
        gsp_ob=np.array((gsp_obtemp[-1]-60)*100).reshape(1,1)
        ###可以被忽略
        #tieline_ob=np.array((tieline_ob_temp[-1][0]+tieline_ob_temp[-1][1])/6000000).reshape(1,1)
        
        pvs_ob=np.array(pvs_ob).reshape(self.pv_num,1)
        
        pvs_max_ob=np.array(pvs_max_ob).reshape(self.pv_num,1)
        
        bat_ob=np.array(bat_ob).reshape(self.bat_num,1)
        soc_ob=np.array(soc_ob).reshape(self.bat_num,1)
        bat_actual_power=np.array(bat_actual_power).reshape(self.bat_num,1)
        # if self.use_irrad:
        #     irradnow_ob=np.array(irradnow).reshape(1,1)
        # #observation=np.concatenate((gen_ob,load_ob,gsp_ob,tieline_ob,pvs_ob,pvs_max_ob,bat_ob,soc_ob))#15维的一手变量,尽量把这些值缩放为[0-1]
        #     observation=np.concatenate((gen_ob,load_ob,gsp_ob,pvs_ob,pvs_max_ob,bat_ob,soc_ob,irradnow_ob))#15维的一手变量,尽量把这些值缩放为[0-1]
        # else:
        observation=np.concatenate((gen_ob,load_ob,gsp_ob,pvs_ob,pvs_max_ob,bat_ob,soc_ob))#15维的一手变量,尽量把这些值缩放为[0-1]
        self.state= observation.squeeze()#把形状变成(15,)的1维数组
        
        for bat in bat_ob:
            bat_mid=abs(bat-0.5)
            bat_err.append(bat_mid)
            
        # for pct in pctPmpps:
        #     pct_mid=abs(pct-0.5)
        #     pv_err.append(pct_mid)
        #计算当前步数的reward
        #TODO:待编写reward，只计算当前负载接入点的频率变化
        max_penalty = min(0, 2 - gsp_ob) #penalty is negative if above max
        min_penalty = min(0, gsp_ob + 2) #penalty is negative if below min
        total_violation = (max_penalty + min_penalty)
        # #reward=total_violation+self.load_lamda*(action[0]*loadp[0]+action[1]*loadp[1]+action[2]*loadp[2])
        # reward=-abs(f[-1]-60)+self.load_lamda*(action[0]*loadp[0]+action[1]*loadp[1]+action[2]*loadp[2])
        if self.t==0:
           self.gsp_before=gsp_ob
           self.pvs_ob_before=pvs_ob
           self.bat_actual_power_before=bat_actual_power
           self.pvs_actual_power_before=pvs_actual_power
        test2=sum(abs(pvs_ob-self.pvs_ob_before))     
        #reward=-abs(gsp_ob-1)*1000-0.0001*sum(dis_errs)+1000*total_violation-0.0001*(sum(bat_err)+sum(pvs_reward))+0.000001*sum(abs(pvs_ob-self.pvs_ob_before))#先稳定频率，tielineob先不控制
        #reward=-abs(gsp_ob-1)*1000#先稳定频率，tielineob先不控制
        #reward=-abs(gsp_ob-1)*1000-0.0001*sum(dis_errs)+1000*total_violation-0.000001*(sum(bat_err)+sum(pvs_reward))+0.000001*sum(abs(pvs_ob-self.pvs_ob_before))#先稳定频率，tielineob先不控制
        #reward=-abs(gsp_ob-1)*1000-0.0001*sum(dis_errs)+1000*total_violation-0.000001*(sum(pvs_reward))-0.00001*sum(bat_err)-0.000001*sum(abs(pvs_ob-self.pvs_ob_before))#先稳定频率，tielineob先不控制
        #reward=-abs(gsp_ob-1)*1000-0.0001*sum(dis_errs)+1000*total_violation-0.000001*(sum(pvs_reward))-0.00001*abs(abs(sum(pvs_ob)+sum(bat_actual_power))-abs(sum(self.pvs_ob_before)+sum(self.bat_actual_power_before)))#先稳定频率，tielineob先不控制
        #reward=-abs(gsp_ob-1)*1000-0.0001*sum(dis_errs)+1000*total_violation-0.000001*(sum(pvs_reward))-0.00001*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))-0.00001*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))
        #reward=-abs(gsp_ob-1)*1000+1000*total_violation-0.000001*(sum(pvs_reward))-0.00001*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))-0.00001*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))
        #reward=-abs(gsp_ob-1)*1000+1000*total_violation-0.000001*(sum(pvs_reward))-0.0001*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))-0.00001*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))
        #2025-1-3-reward=-abs(gsp_ob-1)*self.f_deviation+self.total_violation*total_violation-self.allocate_err*(sum(pvs_reward))-self.pv_output_smooth*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))-self.bat_output_smooth*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))
        # reward=-((gsp_ob-1)**2)*self.f_deviation+self.total_violation*total_violation-self.allocate_err*(sum(pvs_reward))-self.pv_output_smooth*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))-self.bat_output_smooth*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))
        
        # f_deviation_reward=-abs(gsp_ob-1)*self.f_deviation
        # total_violation_reward=self.total_violation*total_violation
        # allocate_err_reward=-self.allocate_err*(sum(pvs_reward))
        # pv_output_smooth_reward=-self.pv_output_smooth*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))
        # bat_output_smooth_reward=-self.bat_output_smooth*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))
        #此处目前奖励函数也有多个值
        #对于奖励函数，使用基于目标值的相对归一化
        
        #f_deviation_reward=(gsp_ob-1)**2*100000/1.111#这里的最大偏差是+-0.2，所以值域是[0,0.00001111],乘以10w后值域为[0,1.111]，这样就把值域限制在了0，1之间
        #total_violation_reward=total_violation*1000/3.333#电压违规最大的值为+-0.2，所以值域为[0,0.00333],把值域缩放为[0,1]
        
        f_deviation_reward=(gsp_ob/10)**2 #更改一下大小
        
        
        total_violation_reward=total_violation*10#电压违规最大的值为+-0.2，所以值域为[0,0.00333],把值域缩放为[0,1]
        #allocate_err_reward=sum(pvs_reward)/(0.5*self.pv_num)#把其最大值归一化为[0,1]
        allocate_err_reward=pvs_reward

        # pv_output_smooth_reward=-self.pv_output_smooth*abs(abs(sum(pvs_ob))-abs(sum(self.pvs_ob_before)))#光伏和储能互补的平滑项不好引导。能否改为频率的平滑？
        # bat_output_smooth_reward=-self.bat_output_smooth*abs(abs(sum(bat_actual_power))-abs(sum(self.bat_actual_power_before)))


        
        # if (gsp_ob-1)>0:#鼓励储能这一项要加，放电时，bat_ob为正值；充电时，bat_ob为负值
        #     bat_output_reward=-sum(bat_ob)/self.bat_num#此项目放电为正，充电为负
        # elif (gsp_ob-1)<0:#希望储能放电
        #     bat_output_reward=sum(bat_ob)/self.bat_num#值还是0-1之间，只要放电，就正
            
        if gsp_ob/10 > 0.01:  # 系统频率高于1.0 pu，期望储能电池给自己充电
            if sum(bat_actual_power) > 0.1:  # 储能正在充电
                bat_output_reward = 0  # 充电时奖励自己
            else:  # 储能不充电或放电
                bat_output_reward = -1  # 惩罚放电行为(可以将这个也换成放电的大小np.mean),充电才不是-1
        elif gsp_ob/10 < -0.01:  # 系统频率低于1.0 pu，期望放电，意味着光伏出力不足，因此期望储能放电，
            if sum(bat_actual_power) < -0.1:  # 储能正在放电
                bat_output_reward = 0
            else:  # 储能不放电或充电
                bat_output_reward = -1
        else:
            bat_output_reward = 0  # 频率在正常范围内，不给予奖励
        
        bat_discharge_reward = 0
        if sum(bat_actual_power) <-0.01:  # 储能放电,储能会自己想办法，使得加和大于0
            if gsp_ob/10 > -0.001:#-1
                bat_discharge_reward = -np.mean(bat_ob)#万一这时候充电怎么办？
            else:
                bat_discharge_reward = 0
        
        power_change = abs(sum(pvs_actual_power) + sum(bat_actual_power) - (sum(self.pvs_actual_power_before) + sum(self.bat_actual_power_before)))

        # 计算功率变化的相对变化（避免绝对变化量对不同规模的影响）
        total_power_before = abs(sum(self.pvs_actual_power_before) + sum(self.bat_actual_power_before)) + 1e-6  # 加上小值避免除零错误
        relative_change = power_change / total_power_before  # 相对变化
        
        pv_bat_smooth_reward=relative_change#缩放到0-1之间
        
        f_bat_discharge_reward=bat_discharge_reward
        
        bat_allocate_single=bat_allocate_reward
        
        reward=-self.f1*f_deviation_reward+self.f2*total_violation_reward-self.f3*allocate_err_reward-self.f4*bat_allocate_single+self.f5*bat_output_reward+self.f6*f_bat_discharge_reward-self.f7*pv_bat_smooth_reward

        self.gsp_before=gsp_ob
        self.pvs_ob_before=pvs_ob
        self.bat_actual_power_before=bat_actual_power
        self.pvs_actual_power_before=pvs_actual_power
        self.t += 1
        #把三个reward的大小都记录在info里面去
        # info["f_deviation_reward"]=f_deviation_reward
        # info["total_violation_reward"]=total_violation_reward
        # info["allocate_err_reward"]=allocate_err_reward
        # info["pv_output_smooth_reward"]=pv_output_smooth_reward
        # info["bat_output_smooth_reward"]=bat_output_smooth_reward
        
        info["f_deviation_reward"]=f_deviation_reward
        info["total_violation_reward"]=total_violation_reward
        info["allocate_err_reward"]=allocate_err_reward
        info["bat_allocate_reward"]=bat_allocate_reward
        info["bat_output_reward"]=bat_output_reward
        info["pv_bat_smooth_reward"]=pv_bat_smooth_reward
        info["bat_discharge_reward"]=bat_discharge_reward
        info["active_output"]=-Vsource_power[0]
        if done:
            self.eng.set_param(self.env_name, 'SimulationCommand', 'stop', nargout=0)
        
        return self.state, reward,done,False,info
    
    def reset(self,seed=None):#启动matlab
        #问题1.环境初始化就启动matlab吗，还是在reset的时候启动，按理来说应该放在reset
        pvs_ob=[]
        pvs_max_ob=[]
        
        bat_ob=[]
        soc_ob=[]
        observation=None
        self.sample=None
        
        self.gsp_before=0
        self.pvs_ob_before=[]
        self.bat_actual_power_before=[]
        self.pvs_actual_power_before=[]
        
        self.circuit.reset()#首先进行opendss电路参数的初始化
        
        irradnow=self.circuit.get_all_pvs_irrad_after_solve()#此时我需要的是什么？光伏的实际输出和最大输出，最大输出也作为obs的输入
        for name in self.pv_names:
            pv_ob=self.circuit.dss.ActiveCircuit.CktElements(name).TotalPowers[0]
            #pvs_ob.append(pv_ob/10000)#单位为10mw
            pvs_ob.append(pv_ob/self.circuit.pvs[name].max_output)#单位为10mw
            pvs_max_ob.append(-self.circuit.pvs[name].max_output/100000)#单位为100mw
            
        if self.bat_num>0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        else: soc_errs, dis_errs, bat_statuses = [], [], dict()
        
        for name, bat in self.circuit.batteries.items():
            batpower=-1*bat.actual_power()/bat.max_kw
            soc=bat.soc
            bat_ob.append(batpower)#单位为10mw
            soc_ob.append(soc)
        
        
        
        Vsource_power=self.circuit.dss.ActiveCircuit.CktElements('Vsource.SOURCE').TotalPowers#配网向主网吸收功率时，vsource_power的数值为负。
        
        
        self.loadp=np.array([100,90,125])
        self.t = 0
        #TODO:restart
        #self.eng.set_param(self.env_name, 'SimulationCommand', 'stop', nargout=0)
        
        faults=[1,2,3,5]
        self.sample = np.random.choice(faults, size=1)
        #self.sample=5
        if self.use_render:
            self.sample=[int(self.render_fault)]
        # if self.use_irrad:
        #     self.sample=[5]
        # #设置故障类型这里智能
        print('sample=',self.sample)
        
        self.eng.set_param(self.env_name+'/load1p', 'value',f"{-Vsource_power[0]*1000}", nargout=0)#这里是先从opendss处读取了功率，在仿真未开始前，这样所有实验的初始状态都相同，都是已经介入了系统
        self.eng.set_param(self.env_name+'/load1q', 'value',f"{-Vsource_power[1]*1000}", nargout=0)
        # self.eng.set_param('self.env_name/load1p', 'value','100e6', nargout=0)
        # self.eng.set_param('self.env_name/load1q', 'value','35e6', nargout=0)
        self.eng.set_param(self.env_name+'/loadbus6p', 'value','90e6', nargout=0)
        self.eng.set_param(self.env_name+'/loadbus6q', 'value','30e6', nargout=0)
        self.eng.set_param(self.env_name+'/loadbus5p1', 'value','1e-5', nargout=0)
        self.eng.set_param(self.env_name+'/loadbus5q1', 'value','1e-5', nargout=0)
        
        self.eng.set_param(self.env_name+'/Step46', 'value',"0", nargout=0)
        self.eng.set_param(self.env_name+'/Step89', 'value',"0", nargout=0)
        
        self.eng.set_param(self.env_name+'/pause_time', 'value', str(1), nargout=0) # 初始化第一个内部暂停时间为1s
        self.eng.set_param(self.env_name, 'StopTime', str(self.horizon+1), nargout=0) # 设定仿真截止时间
        self.eng.set_param(self.env_name, 'SimulationCommand', 'start', nargout=0) # 开启控制，会在0.01s自动pause
        self._wait_until_paused()  # P0-11（原为无超时忙等）
        # 获取当前负荷的有功无功大小,即读取当前的matlab工作区
        gen_obtemp=np.array(self.eng.workspace['gen_ob'])#读取后转为ndarray数组
        load_ob_temp=np.array(self.eng.workspace['load_ob'])
        gsp_obtemp=np.array(self.eng.workspace['out_f_load1'])
        #tieline_ob_temp=np.array(self.eng.workspace['tieline_ob'])
        
        gen_temp=gen_obtemp[-1]
        gen1_ob=gen_temp[0]/69400000
        gen2_ob=gen_temp[1]/163000000
        gen3_ob=gen_temp[2]/85000000
        
        genp_ob=np.array([gen1_ob,gen2_ob,gen3_ob]).reshape(3,1)
        genw_ob=np.array(gen_temp[3:]).reshape(3,1)
        gen_ob=np.concatenate((genp_ob,genw_ob))
        load_temp=load_ob_temp[-1]#laodp=0,2,4
        load1_ob=load_temp[0]/100000000+1
        load2_ob=load_temp[1]/90000000
        load3_ob=load_temp[2]/125000000
        load_ob=np.array([load1_ob,load2_ob,load3_ob]).reshape(3,1)
        gsp_ob=np.array((gsp_obtemp[-1]-60)*100).reshape(1,1)
        
        #tieline_ob=np.array((tieline_ob_temp[-1][0]+tieline_ob_temp[-1][1])/6000000).reshape(1,1)
        
        
        pvs_ob=np.array(pvs_ob).reshape(self.pv_num,1)
        pvs_max_ob=np.array(pvs_max_ob).reshape(self.pv_num,1)
        bat_ob=np.array(bat_ob).reshape(self.bat_num,1)
        soc_ob=np.array(soc_ob).reshape(self.bat_num,1)
        
        # if self.use_irrad:
        #     irradnow_ob=np.array(irradnow).reshape(1,1)
        #     observation=np.concatenate((gen_ob,load_ob,gsp_ob,pvs_ob,pvs_max_ob,bat_ob,soc_ob,irradnow_ob))#15维的一手变量
        # else:
        #observation=np.concatenate((gen_ob,load_ob,gsp_ob,tieline_ob,pvs_ob,pvs_max_ob,bat_ob,soc_ob))#15维的一手变量
        observation=np.concatenate((gen_ob,load_ob,gsp_ob,pvs_ob,pvs_max_ob,bat_ob,soc_ob))#14维的一手变量，少了tieline_ob
        self.state= observation.squeeze()#把形状变成(15,)的1维数组
        #return self.state,{}#for SB3
        return self.state
        
    def render(self, mode='human'):
        t = np.array(self.eng.workspace['out_t']).reshape(-1)
        f = np.array(self.eng.workspace['out_f_load1']).reshape(-1)
        
        #loadt=np.array(self.eng.workspace['load_ob1t']).reshape(-1)/1000
        
        matplotlib.use('TkAgg')
        # plt.subplot(2, 1, 1)
        fig = plt.figure(figsize=(30, 60))
        plt.plot(t, f, 'or-')
        
        # plt.subplot(2, 1, 2)
        #plt.plot(t, loadt, 'or-')
        #plt.savefig('/home/yushixuan/HARL/results/lfc/good_result/no_control_0.45_plot'+'/plot_sample=5.png')
        
        plt.show()
        return t,f
    
    def seed(self, seed):
        self.ActionSpace.seed(seed)
        
    def close(self):
        self.eng.quit()
        return None
        
        
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
    def __init__(self,pv_num,bat_num):
        
        low, high = [0.05]*pv_num, [0.8]*pv_num  
        low, high = low+[-1]*bat_num, high+[1]*bat_num 
        low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
        self.space = gym.spaces.Box(low, high, shape=(pv_num+bat_num,))

    def sample(self):#联合采样
        ss = self.space.sample()
        # if self.bat_act_num == np.inf:
        #     return np.concatenate(ss)#如果电池是连续动作，就把他们拼接在一起
        return ss#都是连续动作不用拼接

    def seed(self, seed):
        self.space.seed(seed)



class PIDController: 
    def __init__(self, Kp, Ki, Kd):
        """
        初始化增量式PID控制器
        :param Kp: 比例增益
        :param Ki: 积分增益
        :param Kd: 微分增益
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.previous_error = 0
        self.integral = 0
        self.previous_output = 0.42
        self.prev_prev_error = 0  # e(k-2)

    def calculate(self, error):
        """
        根据误差计算增量式PID输出
        :param error: 当前误差
        :return: PID输出
        """
        # 误差死区
        dead_zone = 0.01
        if abs(error) < dead_zone:
            error = 0
        
        # 计算比例项的增量
        delta_P = self.Kp * (error - self.previous_error)
        
        # # 计算积分项的增量，防止积分饱和
        # if error >=0:#频率低于系统频率，积分项为0
        #     self.integral = 0
        # else:
        #     self.integral += error
        # self.integral = np.clip(self.integral, -2, 2)
        # delta_I = self.Ki * self.integral
        
        delta_I = self.Ki * error
        # 计算微分项的增量
        delta_D = self.Kd * (error - 2*self.previous_error+ self.prev_prev_error)
        
        # 计算总增量
        pid_output = self.previous_output +(delta_P + delta_I + delta_D)
        
        # 更新前一个误差
        self.prev_prev_error = self.previous_error
        self.previous_error = error
        # 更新前一个输出
        self.previous_output = pid_output
        
        return pid_output

def pid_control_for_devices(frequency_deviations, pid_controllers, device_type, device_num):
    """
    根据每个设备的频率偏差计算光伏和储能电池的功率输出
    :param frequency_deviations: 每个设备的频率偏差值列表
    :param pid_controllers: 每个设备的PID控制器实例列表
    :param device_type: 设备类型(光伏:pvs,储能电池:bess)
    :param device_num: 设备数量
    :return: 每个设备的功率输出(0到1之间的值)
    """
    # 存储每个设备的功率输出
    power_outputs = []
    
    for i in range(device_num):
        # 根据频率偏差值和PID控制器计算功率输出
        pid_output = pid_controllers[i].calculate(frequency_deviations)  # 这里需要用frequency_deviations[i]而不是frequency_deviations
        
        if device_type == "pvs":
            # 若是光伏的输出，将PID输出限制在[0, 1]范围内
            power_output = np.clip(pid_output, 0, 1)
            #power_output = np.clip(pid_output, 0, 1)
        else:
            # 若是储能电池的输出，将PID输出限制在[-1, 1]范围内
            power_output = np.clip(pid_output, -1, 1)
        
        power_outputs.append(power_output)
    
    return power_outputs

def pid_control_output(pid_controllers, frequency_deviations):
    """
    计算每个设备的功率输出
    :param pid_controllers: PID控制器实例列表
    :param frequency_deviations: 每个设备的频率偏差
    :return: 所有设备的功率输出
    """
    # 计算光伏和储能电池的功率输出
    power_outputs_pvs = pid_control_for_devices(frequency_deviations, pid_controllers, "pvs", 16)#对于13节点，前7个为光伏，后3个为储能电池
    power_outputs_bess = pid_control_for_devices(frequency_deviations, pid_controllers, "bess", 4)
    
    # 合并光伏和储能电池的功率输出
    pid_actions = power_outputs_pvs + power_outputs_bess
    
    return pid_actions
