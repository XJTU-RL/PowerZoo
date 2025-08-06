import os
import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from envs.power_envs.powerzoo.powerzoo.env import Env, ActionSpace
import logging

logger = logging.getLogger(__name__)

class SingleAgentPowerZooEnv(Env):
    """单智能体PowerZoo环境
    
    基于原有的多智能体PowerZoo环境，创建单智能体版本。
    在全离散动作情况下使用MultiDiscrete动作空间，更加自然和高效。
    
    主要特点:
    - 直接使用MultiDiscrete动作空间，无需分解为多个Discrete
    - 保持原有的观测空间和奖励函数
    - 简化动作处理逻辑
    - 适合传统单智能体RL算法（DQN、PPO、SAC等）
    """
    
    def __init__(self, folder_path: str = None, info: Dict[str, Any] = None, 
                 dss_act: bool = False, config=None, action_space_type: str = "discrete"):
        """初始化单智能体PowerZoo环境
        
        Args:
            folder_path: DSS文件夹路径（可选，如果提供config则自动设置）
            info: 环境配置信息（可选，如果提供config则自动设置）
            dss_act: 是否使用DSS动作
            config: SingleAgentConfig配置对象（推荐使用）
            action_space_type: 动作空间类型 ("discrete" 或 "continuous")
        """
        # 如果提供了config，则从config中提取参数
        if config is not None:
            from .single_agent_config import SingleAgentConfig
            if not isinstance(config, SingleAgentConfig):
                raise TypeError("config must be an instance of SingleAgentConfig")
            
            # 从config生成folder_path和info
            folder_path = config.get_folder_path()
            info = config.to_env_info()
            
            # 设置日志级别
            if config.log_level:
                logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        
        # 检查必要参数
        if folder_path is None or info is None:
            raise ValueError("必须提供folder_path和info，或者提供config参数")
        
        # 保存动作空间类型
        self.action_space_type = action_space_type
        
        # 调用父类初始化
        super().__init__(folder_path, info, dss_act)
        
        # 添加PV相关属性（父类中没有但单智能体环境需要）
        self.pv_names = list(self.circuit.pvs.keys()) if hasattr(self.circuit, 'pvs') else []
        self.pv_num = len(self.pv_names)
        self.pv_control_enabled = info.get('pv_control', False)
        self.pv_act_num = info.get('pv_act_num', 5)  # 从配置中读取光伏动作数量，默认为5
        
        # 重新设置动作空间为单智能体版本
        self._setup_single_agent_action_space()
        
        # 重新设置观测空间以兼容stable-baselines3
        self._setup_single_agent_observation_space()
        
        logger.info(f"单智能体PowerZoo环境初始化完成")
        logger.info(f"设备数量 - 电容器: {self.cap_num}, 调压器: {self.reg_num}, 电池: {self.bat_num}, 光伏: {self.pv_num}")
        logger.info(f"动作空间: {self.action_space}")
        logger.info(f"观测空间: {self.observation_space}")
    
    def _setup_single_agent_action_space(self):
        """设置单智能体动作空间
        
        根据action_space_type参数创建合适的动作空间：
        - discrete: 将MultiDiscrete转换为单个Discrete动作空间
        - continuous: 创建Box连续动作空间
        """
        original_space = self.ActionSpace.space
        logger.info(f"原始动作空间类型: {type(original_space)}")
        logger.info(f"原始动作空间: {original_space}")
        logger.info(f"请求的动作空间类型: {self.action_space_type}")
        
        if self.action_space_type == "continuous":
            # 创建连续动作空间
            self._setup_continuous_action_space()
        else:
            # 创建离散动作空间（默认）
            self._setup_discrete_action_space(original_space)
    
    def _setup_discrete_action_space(self, original_space):
        """设置离散动作空间
        
        创建包含电容器、调压器、电池和光伏的完整MultiDiscrete动作空间
        """
        # 构建完整的离散动作空间向量
        nvec = []
        
        # 电容器动作（每个电容器2个状态：开/关）
        if self.cap_num > 0:
            nvec.extend([2] * self.cap_num)
        
        # 调压器动作
        if self.reg_num > 0:
            nvec.extend([self.reg_act_num] * self.reg_num)
        
        # 电池动作（如果是离散的）
        if self.bat_num > 0 and isinstance(self.bat_act_num, (int, float)) and self.bat_act_num < float('inf'):
            nvec.extend([int(self.bat_act_num)] * self.bat_num)
        
        # 光伏动作（如果启用且是离散的）
        if (self.pv_control_enabled and self.pv_num > 0 and 
            isinstance(self.pv_act_num, (int, float)) and self.pv_act_num < float('inf')):
            nvec.extend([int(self.pv_act_num)] * self.pv_num)
        
        if nvec:
            # 创建MultiDiscrete动作空间
            multi_discrete_space = gym.spaces.MultiDiscrete(nvec)
            
            # 计算所有可能的动作组合数量
            total_actions = int(np.prod(nvec))
            self.action_space = gym.spaces.Discrete(total_actions)
            self._multi_discrete_nvec = np.array(nvec)
            
            logger.info(f"创建离散动作空间: MultiDiscrete{nvec} -> Discrete({total_actions})")
            logger.info(f"动作维度分配 - 电容器: {self.cap_num}, 调压器: {self.reg_num}, 电池: {self.bat_num}, 光伏: {self.pv_num}")
        else:
            # 如果没有任何离散动作，创建一个虚拟动作空间
            self.action_space = gym.spaces.Discrete(1)
            self._multi_discrete_nvec = np.array([1])
            logger.warning("没有找到离散动作，创建虚拟动作空间")
    
    def _setup_continuous_action_space(self):
        """设置连续动作空间
        
        为DDPG、TD3等连续控制算法创建Box动作空间
        """
        # 计算连续动作维度
        action_dim = 0
        
        # 电容器动作（每个电容器1个连续值，范围[0,1]表示开关状态概率）
        if self.cap_num > 0:
            action_dim += self.cap_num
        
        # 调压器动作（每个调压器1个连续值，范围[-1,1]表示调节方向和强度）
        if self.reg_num > 0:
            action_dim += self.reg_num
        
        # 电池动作（每个电池1个连续值，范围[-1,1]表示充放电功率）
        if self.bat_num > 0:
            action_dim += self.bat_num
        
        # 光伏动作（每个光伏1个连续值：功率输出档位）
        if self.pv_num > 0:
            action_dim += self.pv_num
        
        # 创建Box动作空间
        low = np.full(action_dim, -1.0, dtype=np.float32)
        high = np.full(action_dim, 1.0, dtype=np.float32)
        
        # 电容器动作范围调整为[0,1]
        if self.cap_num > 0:
            low[:self.cap_num] = 0.0
        
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._multi_discrete_nvec = None
        self._continuous_action_dim = action_dim
        
        logger.info(f"创建连续动作空间: Box({action_dim},) 范围[{low.min():.1f}, {high.max():.1f}]")
        logger.info(f"动作维度分配 - 电容器: {self.cap_num}, 调压器: {self.reg_num}, 电池: {self.bat_num}, 光伏: {self.pv_num}")
    
    def _setup_single_agent_observation_space(self):
        """设置单智能体观测空间
        
        修复观测空间定义以兼容stable-baselines3。
        """
        # 确保环境已经重置过，获取观测维度
        if not hasattr(self, 'obs') or self.obs is None:
            self.reset(load_profile_idx=0)
        
        # 计算观测维度
        nnode = len(np.hstack(list(self.obs['bus_voltages'].values())))
        nload = len(self.obs['load_profile_t']) if self.observe_load else 0
        
        # 构建观测空间边界
        low = [0.8] * nnode  # 电压下界
        high = [1.2] * nnode  # 电压上界
        
        # 添加电容器状态边界
        low.extend([0] * self.cap_num)
        high.extend([1] * self.cap_num)
        
        # 添加调压器状态边界
        low.extend([0] * self.reg_num)
        high.extend([self.reg_act_num] * self.reg_num)
        
        # 添加电池状态边界（SOC + 功率）
        low.extend([0, -1] * self.bat_num)
        high.extend([1, 1] * self.bat_num)
        
        # 添加负荷观测边界（如果启用）
        if self.observe_load:
            low.extend([0.0] * nload)
            high.extend([1.0] * nload)
        
        # 转换为numpy数组
        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        
        # 创建Box观测空间，明确指定shape和dtype
        self.observation_space = gym.spaces.Box(
            low=low, 
            high=high, 
            shape=(len(low),), 
            dtype=np.float32
        )
        
        logger.info(f"观测空间维度: {self.observation_space.shape}")
        logger.info(f"观测空间范围: [{low.min():.1f}, {high.max():.1f}]")
    
    def _convert_discrete_to_multi_discrete(self, action: int) -> np.ndarray:
        """将单个离散动作转换为MultiDiscrete格式
        
        Args:
            action: 单个离散动作值
            
        Returns:
            MultiDiscrete格式的动作数组
        """
        if self._multi_discrete_nvec is None:
            return action
            
        # 将单个动作值转换为多维动作（使用正确的基数转换）
        multi_action = []
        remaining = action
        
        # 从右到左处理每个维度
        for i in range(len(self._multi_discrete_nvec) - 1, -1, -1):
            nvec_i = self._multi_discrete_nvec[i]
            action_i = remaining % nvec_i
            remaining = remaining // nvec_i
            multi_action.insert(0, action_i)
            
        logger.debug(f"转换动作: {action} -> {multi_action} (nvec: {self._multi_discrete_nvec})")
        return np.array(multi_action, dtype=np.int32)
    
    def _convert_continuous_to_discrete(self, action: np.ndarray) -> np.ndarray:
        """将连续动作转换为离散动作
        
        Args:
            action: 连续动作数组，范围在[-1, 1]或[0, 1]
            
        Returns:
            MultiDiscrete格式的动作数组
        """
        action = np.array(action, dtype=np.float32)
        discrete_actions = []
        action_idx = 0
        
        # 电容器动作转换（连续值[0,1] -> 离散开关状态）
        for i in range(self.cap_num):
            # 使用阈值0.5来决定开关状态
            cap_action = 1 if action[action_idx] > 0.5 else 0
            discrete_actions.append(cap_action)
            action_idx += 1
        
        # 调压器动作转换（连续值[-1,1] -> 离散调节档位）
        for i in range(self.reg_num):
            # 将连续值映射到调压器档位
            continuous_val = action[action_idx]
            # 映射到[0, reg_act_num-1]范围
            reg_action = int((continuous_val + 1) / 2 * (self.reg_act_num - 1))
            reg_action = np.clip(reg_action, 0, self.reg_act_num - 1)
            discrete_actions.append(reg_action)
            action_idx += 1
        
        # 电池动作转换（连续值[-1,1] -> 离散功率档位）
        for i in range(self.bat_num):
            # 将连续值映射到电池功率档位
            continuous_val = action[action_idx]
            # 映射到[0, bat_act_num-1]范围
            bat_action = int((continuous_val + 1) / 2 * (self.bat_act_num - 1))
            bat_action = np.clip(bat_action, 0, self.bat_act_num - 1)
            discrete_actions.append(bat_action)
            action_idx += 1
        
        # 光伏动作转换（1个连续值 -> 1个离散值：功率输出档位）
        for i in range(self.pv_num):
            # 光伏功率输出档位
            pv_continuous = action[action_idx]
            pv_action = int((pv_continuous + 1) / 2 * (self.pv_act_num - 1))
            pv_action = np.clip(pv_action, 0, self.pv_act_num - 1)
            discrete_actions.append(pv_action)
            action_idx += 1
        
        result = np.array(discrete_actions, dtype=np.int32)
        logger.debug(f"连续动作转换: {action} -> {result}")
        return result
    
    def step(self, action):
        """执行一步环境交互
        
        Args:
            action: 单智能体动作，可能是单个离散值、MultiDiscrete数组或连续动作数组
        
        Returns:
            observation: 环境观测
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 根据动作空间类型处理动作
        if self.action_space_type == "continuous":
            # 连续动作转换为离散动作
            action = self._convert_continuous_to_discrete(action)
        elif isinstance(action, (int, np.integer)):
            # 单个离散动作转换为MultiDiscrete格式
            action = self._convert_discrete_to_multi_discrete(action)
        
        # 解析动作
        cap_actions, reg_actions, bat_actions, pv_actions = self._parse_single_agent_action(action)
        
        # 调试信息
        logger.debug(f"解析后的动作 - 电容器: {cap_actions}, 调压器: {reg_actions}, 电池: {bat_actions}, 光伏: {pv_actions}")
        
        # 执行动作并返回结果
        return self._execute_actions(cap_actions, reg_actions, bat_actions, pv_actions)
    
    def _parse_single_agent_action(self, action):
        """解析单智能体动作为各设备的动作
        
        Args:
            action: 单智能体动作
            
        Returns:
            cap_actions: 电容器动作列表
            reg_actions: 调压器动作列表 
            bat_actions: 电池动作列表
            pv_actions: 光伏动作列表
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            # 单个离散动作，需要转换为MultiDiscrete格式
            action = self._convert_discrete_to_multi_discrete(action)
            
        if isinstance(self.action_space, (gym.spaces.MultiDiscrete, gym.spaces.Discrete)):
            # 纯离散动作空间
            # 确保action是1维数组
            action = np.array(action).flatten()
            
            # 按设备类型分割动作
            idx = 0
            cap_actions = action[idx:idx+self.cap_num].tolist() if self.cap_num > 0 else []
            idx += self.cap_num
            
            reg_actions = action[idx:idx+self.reg_num].tolist() if self.reg_num > 0 else []
            idx += self.reg_num
            
            # 处理电池动作（可能是离散或连续）
            if self.bat_num > 0:
                if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num < float('inf'):
                    # 离散电池动作
                    bat_actions = action[idx:idx+self.bat_num].tolist()
                    idx += self.bat_num
                else:
                    # 连续电池动作（不应该在MultiDiscrete中出现）
                    bat_actions = []
            else:
                bat_actions = []
            
            # 处理光伏动作（可能是离散或连续）
            if self.pv_control_enabled and self.pv_num > 0:
                if isinstance(self.pv_act_num, (int, float)) and self.pv_act_num < float('inf'):
                    # 离散光伏动作
                    pv_actions = action[idx:idx+self.pv_num].tolist()
                else:
                    # 连续光伏动作（不应该在MultiDiscrete中出现）
                    pv_actions = []
            else:
                pv_actions = []
                
        elif isinstance(self.action_space, gym.spaces.Box):
            # 连续动作空间
            continuous_action = np.array(action)
            
            # 使用之前实现的连续动作转换方法
            cap_actions, reg_actions, bat_actions, pv_actions = self._convert_continuous_to_discrete(continuous_action)
            
        elif isinstance(self.action_space, gym.spaces.Tuple):
            # 混合动作空间（离散 + 连续）
            discrete_action, continuous_action = action[0], action[1]
            
            # 解析离散动作
            discrete_action = np.array(discrete_action)
            idx = 0
            cap_actions = discrete_action[idx:idx+self.cap_num].tolist() if self.cap_num > 0 else []
            idx += self.cap_num
            
            reg_actions = discrete_action[idx:idx+self.reg_num].tolist() if self.reg_num > 0 else []
            idx += self.reg_num
            
            # 处理离散电池动作
            if self.bat_num > 0 and isinstance(self.bat_act_num, (int, float)) and self.bat_act_num < float('inf'):
                bat_discrete = discrete_action[idx:idx+self.bat_num].tolist()
                idx += self.bat_num
            else:
                bat_discrete = []
            
            # 处理离散光伏动作
            if (self.pv_control_enabled and self.pv_num > 0 and 
                isinstance(self.pv_act_num, (int, float)) and self.pv_act_num < float('inf')):
                pv_discrete = discrete_action[idx:idx+self.pv_num].tolist()
            else:
                pv_discrete = []
            
            # 解析连续动作
            continuous_action = np.array(continuous_action)
            cont_idx = 0
            
            # 处理连续电池动作
            if self.bat_num > 0 and (not isinstance(self.bat_act_num, (int, float)) or self.bat_act_num == float('inf')):
                bat_continuous = continuous_action[cont_idx:cont_idx+self.bat_num].tolist()
                cont_idx += self.bat_num
                bat_actions = bat_continuous
            else:
                bat_actions = bat_discrete
            
            # 处理连续光伏动作
            if (self.pv_control_enabled and self.pv_num > 0 and 
                (not isinstance(self.pv_act_num, (int, float)) or self.pv_act_num == float('inf'))):
                pv_continuous = continuous_action[cont_idx:cont_idx+self.pv_num*2].tolist()  # 有功功率 + 功率因数
                pv_actions = pv_continuous
            else:
                pv_actions = pv_discrete
                
        else:
            raise ValueError(f"不支持的动作空间类型: {type(self.action_space)}")
        
        return cap_actions, reg_actions, bat_actions, pv_actions
    
    def _execute_actions(self, cap_actions, reg_actions, bat_actions, pv_actions):
        """执行解析后的动作（复用父类逻辑）
        
        Args:
            cap_actions: 电容器动作列表
            reg_actions: 调压器动作列表
            bat_actions: 电池动作列表
            pv_actions: 光伏动作列表
            
        Returns:
            observation, reward, done, info
        """
        # 记录动作执行前的状态
        prev_obs = self.obs.copy() if hasattr(self, 'obs') else {}
        
        # 执行电容器动作
        if cap_actions:
            self.circuit.set_all_capacitor_statuses(cap_actions)
        
        # 执行调压器动作
        if reg_actions:
            self.circuit.set_all_regulator_tappings(reg_actions)
        
        # 执行电池动作
        if bat_actions:
            if isinstance(self.bat_act_num, (int, float)) and self.bat_act_num < float('inf'):
                # 离散电池动作
                self.circuit.set_all_batteries_before_solve(bat_actions)
            else:
                # 连续电池动作
                self.circuit.set_all_batteries_before_solve(bat_actions)
        
        # 执行光伏动作（如果启用）
        if pv_actions and self.pv_control_enabled:
            if isinstance(self.pv_act_num, (int, float)) and self.pv_act_num < float('inf'):
                # 离散光伏动作
                self.circuit.set_all_pv_statuses(pv_actions)
            else:
                # 连续光伏动作
                self.circuit.set_all_pv_powers(pv_actions)
        
        # 运行电路仿真
        self.circuit.dss.ActiveCircuit.Solution.Solve()
        
        # 更新电池状态（在求解后）
        if self.bat_num > 0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
        
        # 更新观测
        self._update_observation()
        
        # 计算奖励
        reward, reward_info = self._calculate_reward(prev_obs, cap_actions, reg_actions, bat_actions, pv_actions)
        
        # 更新时间步
        self.t += 1
        done = self.t >= self.horizon
        truncated = False  # 添加truncated标志
        
        # 构建info字典
        info = {
            'reward_components': reward_info,
            'timestep': self.t,
            'horizon': self.horizon,
            'actions': {
                'capacitors': cap_actions,
                'regulators': reg_actions,
                'batteries': bat_actions,
                'pv_systems': pv_actions if self.pv_control_enabled else []
            }
        }
        
        # 返回观测（根据wrap_observation设置）
        if self.wrap_observation:
            observation = self._flatten_observation()
        else:
            observation = self.obs.copy()
        
        return observation, reward, done, truncated, info
    
    def _update_observation(self):
        """更新环境观测（复用父类逻辑）"""
        # 更新观测状态 - 获取所有母线电压
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0]
        self.obs['bus_voltages'] = bus_voltages
        
        # 获取设备状态 - 使用字典格式以保持与父类一致
        self.obs['cap_statuses'] = {cap: self.circuit.capacitors[cap].status for cap in self.cap_names}
        self.obs['reg_statuses'] = {reg: self.circuit.regulators[reg].tap for reg in self.reg_names}
        
        # 获取电池状态
        bat_statuses = {}
        for bat in self.bat_names:
            battery = self.circuit.batteries[bat]
            bat_statuses[bat] = [battery.soc, battery.actual_power()]
        self.obs['bat_statuses'] = bat_statuses
        
        # 获取光伏状态（如果启用）
        if self.pv_control_enabled and self.pv_num > 0:
            pv_statuses = {}
            for pv in self.pv_names:
                pv_system = self.circuit.pvs[pv]
                pv_statuses[pv] = [pv_system.power_output, pv_system.power_factor]
            self.obs['pv_statuses'] = pv_statuses
        
        # 获取系统指标 - 计算功率损耗比值（与父类格式保持一致）
        total_loss = self.circuit.total_loss()[0]  # 取第一个元素（有功损耗）
        total_power = self.circuit.total_power()[0]  # 取第一个元素（有功功率）
        self.obs['power_loss'] = -total_loss / total_power  # 计算损耗比值
        
        # 时间步信息
        self.obs['time'] = self.t
        
        # 获取负载曲线（如果需要）
        if self.observe_load:
            self.obs['load_profile_t'] = self.load_profile.get_current_load()
    
    def _flatten_observation(self):
        """将观测字典展平为数组（复用父类逻辑）"""
        obs_list = []
        
        # 添加母线电压
        for voltages in self.obs['bus_voltages'].values():
            obs_list.extend(voltages)
        
        # 添加电容器状态（从字典中按顺序提取值）
        for cap_name in self.cap_names:
            obs_list.append(self.obs['cap_statuses'].get(cap_name, 0))
        
        # 添加调压器状态（从字典中按顺序提取值）
        for reg_name in self.reg_names:
            obs_list.append(self.obs['reg_statuses'].get(reg_name, 0))
        
        # 添加电池状态
        for bat_status in self.obs['bat_statuses'].values():
            obs_list.extend(bat_status)
        
        # 添加光伏状态（如果启用）
        if self.pv_control_enabled and 'pv_statuses' in self.obs:
            for pv_status in self.obs['pv_statuses'].values():
                obs_list.extend(pv_status)
        
        # 添加负载曲线（如果需要）
        if self.observe_load:
            obs_list.extend(self.obs['load_profile_t'])
        
        return np.array(obs_list, dtype=np.float32)
    
    def _calculate_reward(self, prev_obs, cap_actions, reg_actions, bat_actions, pv_actions):
        """计算奖励（复用父类的奖励函数）"""
        # 计算动作差异
        prev_cap_dict = prev_obs.get('cap_statuses', {})
        prev_reg_dict = prev_obs.get('reg_statuses', {})
        
        # 将字典转换为列表以便与动作列表进行比较
        prev_cap = [prev_cap_dict.get(cap_name, 0) for cap_name in self.cap_names] if prev_cap_dict else [0] * self.cap_num
        prev_reg = [prev_reg_dict.get(reg_name, 0) for reg_name in self.reg_names] if prev_reg_dict else [0] * self.reg_num
        
        cap_diff = [abs(int(a) - int(b)) for a, b in zip(cap_actions, prev_cap)] if cap_actions else []
        reg_diff = [abs(int(a) - int(b)) for a, b in zip(reg_actions, prev_reg)] if reg_actions else []
        
        # 计算电池相关指标
        soc_errors = []
        discharge_errors = []
        if bat_actions:
            for i, bat_name in enumerate(self.bat_names):
                if i < len(bat_actions):
                    battery = self.circuit.batteries[bat_name]
                    soc_errors.append(abs(battery.soc - 0.5))  # 目标SOC为50%
                    discharge_errors.append(max(0, -battery.actual_power()))  # 放电惩罚
        
        # 计算光伏相关指标
        pv_diff = []
        if pv_actions and self.pv_control_enabled:
            # 简单的光伏控制成本（可根据需要调整）
            pv_diff = [abs(action) for action in pv_actions]
        
        # 使用父类的奖励函数
        reward, reward_info = self.reward_func.composite_reward(
            cap_diff, reg_diff, soc_errors, discharge_errors, record_node=False
        )
        
        return reward, reward_info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, load_profile_idx: Optional[int] = None):
        """重置环境
        
        Args:
            seed: 随机种子（gymnasium标准参数）
            options: 额外选项（gymnasium标准参数）
            load_profile_idx: 负载曲线索引，None表示使用默认值0
            
        Returns:
            observation: 初始观测
            info: 信息字典
        """
        # 处理seed
        if seed is not None:
            self.seed(seed)
            
        # 如果load_profile_idx为None，使用默认值0
        if load_profile_idx is None:
            load_profile_idx = 0
            
        # 调用父类reset方法
        obs = super().reset(load_profile_idx=load_profile_idx)
        
        # 返回gymnasium标准格式
        return obs, {}
    
    def render(self, mode='human'):
        """渲染环境（复用父类方法）"""
        return super().render(mode)
    
    def close(self):
        """关闭环境（复用父类方法）"""
        return super().close()
    
    def get_action_meanings(self):
        """获取动作含义说明
        
        Returns:
            dict: 动作含义字典
        """
        meanings = {
            'action_space_type': type(self.action_space).__name__,
            'total_actions': self.action_space.n if hasattr(self.action_space, 'n') else 'variable',
            'devices': {
                'capacitors': {
                    'count': self.cap_num,
                    'actions': 'Binary (0=Off, 1=On)',
                    'range': '[0, 1]'
                },
                'regulators': {
                    'count': self.reg_num,
                    'actions': f'Discrete tap positions',
                    'range': f'[0, {self.reg_act_num-1}]'
                },
                'batteries': {
                    'count': self.bat_num,
                    'actions': 'Discrete' if isinstance(self.bat_act_num, int) else 'Continuous',
                    'range': f'[0, {self.bat_act_num-1}]' if isinstance(self.bat_act_num, int) else '[-1, 1]'
                }
            }
        }
        
        if self.pv_control_enabled and self.pv_num > 0:
            meanings['devices']['pv_systems'] = {
                'count': self.pv_num,
                'actions': 'Discrete' if isinstance(self.pv_act_num, int) else 'Continuous',
                'range': f'[0, {self.pv_act_num-1}]' if isinstance(self.pv_act_num, int) else '[-1, 1] (Power, PF)'
            }
        
        return meanings


def create_single_agent_powerzoo_env(folder_path: str, info: Dict[str, Any], **kwargs):
    """创建单智能体PowerZoo环境的便捷函数
    
    Args:
        folder_path: DSS文件夹路径
        info: 环境配置信息
        **kwargs: 其他参数
        
    Returns:
        SingleAgentPowerZooEnv: 单智能体环境实例
    """
    return SingleAgentPowerZooEnv(folder_path, info, **kwargs)


# 使用示例
if __name__ == "__main__":
    # 示例配置
    config = {
        'system_name': '34Bus_PV',
        'dss_file': 'ieee34Mod1.dss',
        'max_episode_steps': 100,
        'reg_act_num': 33,
        'bat_act_num': 5,  # 离散电池动作
        'pv_control': True,
        'pv_act_num': float('inf'),  # 连续PV控制
        'worker_idx': 0
    }
    
    # 创建环境
    env = create_single_agent_powerzoo_env('/path/to/node_systems', config)
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作含义: {env.get_action_meanings()}")
    
    # 测试环境
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print(f"奖励: {reward}")
    print(f"完成: {done}")
    print(f"信息: {info}")