# -*- coding: utf-8 -*-
"""
DSR Core Environment
基于OpenDSS的配电网恢复核心环境
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
import random
import os

try:
    from envs.powerzoo.powerzoo.circuit import Circuits
    from envs.powerzoo.powerzoo.loadprofile import LoadProfile
    POWERZOO_AVAILABLE = True
except ImportError as e:
    POWERZOO_AVAILABLE = False
    print(f"PowerZoo components not available: {e}")
from envs.dsr.core.config import DSRConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSRCoreEnv:
    """配电网恢复核心环境，基于OpenDSS"""
    
    def __init__(self, config: DSRConfig, worker_idx: Optional[int] = None):
        self.config = config
        self.worker_idx = worker_idx
        
        # 验证配置
        self.config.validate()
        
        # 检查PowerZoo可用性
        if not POWERZOO_AVAILABLE:
            raise ImportError("PowerZoo components (OpenDSS/cupy) not available. Please install required dependencies.")
        
        # 初始化OpenDSS电路
        self._init_opendss_circuit()
        
        # 初始化智能体
        self._init_agents()
        
        # 初始化故障管理
        self._init_fault_management()
        
        # 环境状态
        self.current_step = 0
        self.done = False
        self.fault_lines = []  # 当前故障线路
        self.energized_buses = set()  # 已通电母线
        
        logger.info(f"DSR核心环境初始化完成: {self.n_agents}个智能体, 基于{self.config.system_name}")
    
    def _init_opendss_circuit(self):
        """初始化OpenDSS电路"""
        # 获取PowerZoo配置
        powerzoo_config = self.config.to_powerzoo_config()
        
        # 构建DSS文件路径
        dss_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'powerzoo', 'systems', self.config.system_name
        )
        dss_file_path = os.path.join(dss_folder, self.config.dss_file)
        
        # 创建Circuits对象
        self.circuit = Circuits(
            dss_file_path,
            RB_act_num=(powerzoo_config['reg_act_num'], powerzoo_config['bat_act_num']),
            dss_act=False
        )
        
        # 获取系统信息
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.n_bus = len(self.all_bus_names)
        
        # 获取线路信息
        self._get_line_info()
        
        # 获取负荷信息
        self._get_load_info()
        
        # 初始化负荷配置文件
        self.load_profile = LoadProfile(
            self.config.max_episode_steps,
            dss_folder,
            self.config.dss_file,
            self.config.load_noise,
            worker_idx=self.worker_idx
        )
        self.num_profiles = self.load_profile.gen_loadprofile(
            use_noise=self.config.load_noise,
            scale=self.config.scale
        )
        
        logger.info(f"OpenDSS电路初始化完成: {self.n_bus}个母线, {len(self.line_info)}条线路")
    
    def _get_line_info(self):
        """获取线路信息"""
        self.line_info = {}
        self.circuit.dss.ActiveCircuit.Lines.First
        while True:
            line_name = self.circuit.dss.ActiveCircuit.Lines.Name
            bus1 = self.circuit.dss.ActiveCircuit.Lines.Bus1.split('.')[0].lower()
            bus2 = self.circuit.dss.ActiveCircuit.Lines.Bus2.split('.')[0].lower()
            enabled = self.circuit.dss.ActiveCircuit.Lines.Enabled
            
            self.line_info[line_name] = {
                'bus1': bus1,
                'bus2': bus2,
                'enabled': enabled,
                'original_enabled': enabled  # 保存原始状态
            }
            
            if self.circuit.dss.ActiveCircuit.Lines.Next == 0:
                break
    
    def _get_load_info(self):
        """获取负荷信息"""
        self.load_info = {}
        load_names = self.circuit.dss.ActiveCircuit.Loads.AllNames
        
        for load_name in load_names:
            self.circuit.dss.ActiveCircuit.Loads.Name = load_name
            bus = self.circuit.dss.ActiveCircuit.Loads.Bus1.split('.')[0].lower()
            kw = self.circuit.dss.ActiveCircuit.Loads.kW
            enabled = self.circuit.dss.ActiveCircuit.Loads.Enabled
            
            # 随机分配负荷优先级
            priority = random.choice(list(range(1, self.config.max_priority_level + 1)))
            priority_weight = self.config.priority_weights.get(priority, 1.0)
            
            self.load_info[load_name] = {
                'bus': bus,
                'kw': kw,
                'priority': priority,
                'priority_weight': priority_weight,
                'enabled': enabled,
                'original_enabled': enabled
            }
        
        logger.info(f"负荷信息获取完成: {len(self.load_info)}个负荷")
    
    def _init_agents(self):
        """初始化智能体"""
        agent_config = self.config.get_agent_config()
        
        self.n_agents = agent_config['total_agents']
        self.n_switch_agents = agent_config['n_switch_agents']
        self.n_pv_agents = agent_config['n_pv_agents']  
        self.n_load_agents = agent_config['n_load_agents']
        self.n_switches = agent_config['n_switches']  # 实际可控开关数量
        
        # 智能体类型和索引映射
        self.agent_types = []
        self.agent_indices = []
        self.agent_bus_mapping = {}
        
        # 开关智能体（1个全局开关控制者）
        self.agent_types.append('switch')
        self.agent_indices.append(0)
        self.agent_bus_mapping[0] = 'global'  # 全局开关控制
        
        # PV智能体
        self.pv_agents = []
        available_buses = list(self.all_bus_names[:self.config.n_pv])  # 简化：选择前N个母线
        for i in range(self.n_pv_agents):
            self.agent_types.append('pv')
            self.agent_indices.append(i)
            bus = available_buses[i] if i < len(available_buses) else f"pv_bus_{i}"
            self.agent_bus_mapping[1 + i] = bus
            self.pv_agents.append({
                'agent_id': 1 + i,
                'bus': bus,
                'max_power': self.config.pv_max_power,  # kW
                'current_power': 0.0
            })
        
        # 负荷智能体
        load_names = list(self.load_info.keys())
        self.load_agents = []
        for i, load_name in enumerate(load_names[:self.n_load_agents]):
            agent_id = 1 + self.n_pv_agents + i
            self.agent_types.append('load')
            self.agent_indices.append(i)
            self.agent_bus_mapping[agent_id] = self.load_info[load_name]['bus']
            self.load_agents.append({
                'agent_id': agent_id,
                'load_name': load_name,
                'bus': self.load_info[load_name]['bus'],
                'priority': self.load_info[load_name]['priority'],
                'kw': self.load_info[load_name]['kw']
            })
        
        logger.info(f"智能体初始化完成: 1个开关 + {self.n_pv_agents}个PV + {self.n_load_agents}个负荷")
    
    def _init_fault_management(self):
        """初始化故障管理"""
        # 可故障的线路（排除重要的主干线路）
        all_lines = list(self.line_info.keys())
        # 简化：随机选择可能故障的线路
        self.faultable_lines = random.sample(all_lines, min(len(all_lines), self.config.max_faultable_lines))
        
        # 分布式发电机（黑启动电源）位置
        self.dg_buses = random.sample(self.all_bus_names, min(len(self.all_bus_names), self.config.n_dg))
        
        logger.info(f"故障管理初始化: {len(self.faultable_lines)}条可故障线路, {len(self.dg_buses)}个DG")
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        self.current_step = 0
        self.done = False
        
        # 选择负荷配置文件
        profile_idx = self.worker_idx if self.worker_idx is not None else 0
        self.load_profile.choose_loadprofile(profile_idx, self.config.load_noise)
        
        # 重置电路
        self.circuit.reset()
        
        # 生成随机故障
        self._generate_random_faults()
        
        # 初始化所有设备状态
        self._reset_device_states()
        
        # 更新通电状态
        self._update_energized_buses()
        
        # 获取初始观测和状态
        #TODO 这里暂且将二者等同
        obs = self._get_observations() 
        state = self._get_global_state()
        
        return obs, state
    
    def _generate_random_faults(self):
        """生成随机故障"""
        # 随机选择故障数量
        n_faults = random.randint(self.config.min_faults, self.config.max_faults)
        
        # 随机选择故障线路
        self.fault_lines = random.sample(self.faultable_lines, n_faults)
        
        # 在OpenDSS中设置线路故障
        for line_name in self.fault_lines:
            self.circuit.dss.ActiveCircuit.Lines.Name = line_name
            self.circuit.dss.ActiveCircuit.Lines.Enabled = False
            self.line_info[line_name]['enabled'] = False
        
        logger.info(f"生成{len(self.fault_lines)}个故障: {self.fault_lines}")
    
    def _reset_device_states(self):
        """重置设备状态"""
        # 断开所有开关（除了基本的主干线路）
        for line_name, line_data in self.line_info.items():
            if line_name not in self.fault_lines:
                # 保持一些基本连接，断开其他
                if random.random() < self.config.line_disconnect_prob:  # 配置的概率断开
                    self.circuit.dss.ActiveCircuit.Lines.Name = line_name
                    self.circuit.dss.ActiveCircuit.Lines.Enabled = False
                    line_data['enabled'] = False
        
        # 重置PV输出
        for pv_agent in self.pv_agents:
            pv_agent['current_power'] = 0.0
        
        # 断开所有负荷
        for load_name in self.load_info:
            self.circuit.dss.ActiveCircuit.Loads.Name = load_name
            self.circuit.dss.ActiveCircuit.Loads.Enabled = False
            self.load_info[load_name]['enabled'] = False
    
    def _update_energized_buses(self):
        """更新通电母线状态"""
        # 构建当前网络拓扑
        G = nx.Graph()
        G.add_nodes_from(self.all_bus_names)
        
        # 添加启用的线路
        for line_name, line_data in self.line_info.items():
            if line_data['enabled']:
                G.add_edge(line_data['bus1'], line_data['bus2'])
        
        # 找到与DG连通的母线
        self.energized_buses = set()
        for dg_bus in self.dg_buses:
            if dg_bus in G:
                connected_buses = nx.node_connected_component(G, dg_bus)
                self.energized_buses.update(connected_buses)
    
    def step(self, actions: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any], 
                                                List[float], bool, Dict[str, Any]]:
        """执行一步动作"""
        self.current_step += 1
        
        # 执行智能体动作
        self._execute_actions(actions)
        
        # 运行潮流计算
        try:
            self.circuit.dss.ActiveCircuit.Solution.Solve()
            converged = self.circuit.dss.ActiveCircuit.Solution.Converged
        except Exception as e:
            logger.warning(f"潮流计算失败: {e}")
            converged = False
        
        # 更新通电状态
        self._update_energized_buses()
        
        # 计算奖励
        rewards = self._calculate_rewards(converged)
        
        # 检查终止条件
        self.done = (not converged or 
                    self.current_step >= self.config.max_episode_steps or
                    self._check_restoration_complete())
        
        # 获取观测和状态
        obs = self._get_observations()
        state = self._get_global_state()
        
        # 生成信息字典
        info = {
            'converged': converged,
            'restored_load_ratio': self._get_restored_load_ratio(),
            'energized_buses': len(self.energized_buses),
            'total_buses': self.n_bus,
            'current_step': self.current_step,
            'fault_lines': self.fault_lines,
        }
        
        return obs, state, rewards, self.done, info
    
    def _execute_actions(self, actions: List[int]):
        """执行智能体动作"""
        # 开关智能体动作
        switch_action = actions[0]
        if switch_action > 0 and switch_action <= len(self.faultable_lines):
            # 选择操作的线路（排除故障线路）
            available_lines = [line for line in self.faultable_lines 
                             if line not in self.fault_lines]
            if available_lines and switch_action <= len(available_lines):
                line_name = available_lines[switch_action - 1]
                # 切换线路状态
                current_state = self.line_info[line_name]['enabled']
                new_state = not current_state
                
                self.circuit.dss.ActiveCircuit.Lines.Name = line_name
                self.circuit.dss.ActiveCircuit.Lines.Enabled = new_state
                self.line_info[line_name]['enabled'] = new_state
        
        # PV智能体动作
        for i, pv_agent in enumerate(self.pv_agents):
            if 1 + i < len(actions):
                pv_action = actions[1 + i]
                # 设置PV输出功率（配置的等级范围）
                max_level = self.config.pv_power_levels - 1
                if 0 <= pv_action <= max_level:
                    power_ratio = pv_action / max_level
                    pv_agent['current_power'] = power_ratio * pv_agent['max_power']
        
        # 负荷智能体动作
        for i, load_agent in enumerate(self.load_agents):
            action_idx = 1 + self.n_pv_agents + i
            if action_idx < len(actions):
                load_action = actions[action_idx]
                load_name = load_agent['load_name']
                
                if load_action == 1:  # 尝试恢复负荷
                    # 检查负荷所在母线是否通电
                    if load_agent['bus'] in self.energized_buses:
                        self.circuit.dss.ActiveCircuit.Loads.Name = load_name
                        self.circuit.dss.ActiveCircuit.Loads.Enabled = True
                        self.load_info[load_name]['enabled'] = True
                elif load_action == 0:  # 断开负荷
                    self.circuit.dss.ActiveCircuit.Loads.Name = load_name
                    self.circuit.dss.ActiveCircuit.Loads.Enabled = False
                    self.load_info[load_name]['enabled'] = False
    
    def _calculate_rewards(self, converged: bool) -> List[float]:
        """计算奖励"""
        if not converged:
            # 潮流不收敛，给予惩罚
            return [self.config.reward_done] * self.n_agents
        
        #NOTE 负荷恢复奖励
        restored_ratio = self._get_restored_load_ratio()
        restore_reward = self.config.reward_restore * restored_ratio
        
        #NOTE 电压越限惩罚 
        voltage_violations = self._get_voltage_violations()
        voltage_penalty = -self.config.reward_voltage * voltage_violations
        
        #NOTE 线路过载惩罚
        line_overloads = self._get_line_overloads()
        overload_penalty = -self.config.reward_overload * line_overloads
        
        #NOTE 总奖励
        total_reward = restore_reward + voltage_penalty + overload_penalty
        
        # 为每个智能体分配奖励（可以根据贡献度调整）
        rewards = [total_reward] * self.n_agents
        
        return rewards
    
    def _get_restored_load_ratio(self) -> float:
        """计算加权负荷恢复率"""
        total_weighted_load = 0.0
        restored_weighted_load = 0.0
        
        for load_name, load_data in self.load_info.items():
            weight = load_data['priority_weight']
            power = load_data['kw']
            total_weighted_load += weight * power
            
            if load_data['enabled']:
                restored_weighted_load += weight * power
        
        if total_weighted_load == 0:
            return 0.0
        
        return restored_weighted_load / total_weighted_load
    
    def _get_voltage_violations(self) -> int:
        """获取电压越限数量"""
        violations = 0
        for bus_name in self.all_bus_names:
            try:
                voltages = self.circuit.bus_voltage(bus_name)
                # 只检查奇数索引（相电压）
                phase_voltages = [voltages[i] for i in range(len(voltages)) if i % 2 == 0]
                
                for voltage in phase_voltages:
                    if voltage < self.config.v_min or voltage > self.config.v_max:
                        violations += 1
            except:
                pass  # 忽略无效母线
        
        return violations
    
    def _get_line_overloads(self) -> int:
        """获取线路过载数量"""
        overloads = 0
        overload_details = []  # 用于调试和日志记录
        
        try:
            # 遍历所有启用的线路
            for line_name, line_data in self.line_info.items():
                if not line_data['enabled']:
                    continue
                    
                # 获取线路电流
                try:
                    currents = self.circuit.edge_current(f"Line.{line_name}")
                    # currents是复数数组，包含实部和虚部
                    # 计算电流幅值
                    current_magnitudes = []
                    for i in range(0, len(currents), 2):
                        real = currents[i]
                        imag = currents[i + 1] if i + 1 < len(currents) else 0
                        magnitude = (real**2 + imag**2)**0.5
                        current_magnitudes.append(magnitude)
                    
                    # 获取线路额定电流
                    self.circuit.dss.ActiveCircuit.Lines.Name = line_name
                    norm_amps = self.circuit.dss.ActiveCircuit.Lines.NormAmps 
                    emerg_amps = self.circuit.dss.ActiveCircuit.Lines.EmergAmps
                    
                    # 如果没有设置额定电流，跳过
                    if norm_amps <= 0:
                        continue
                    
                    # 获取线路电流
                    currents = self.circuit.edge_current(f"Line.{line_name}")
                    
                    # 计算电流幅值
                    current_magnitudes = []
                    for i in range(0, len(currents), 2):
                        real = currents[i]
                        imag = currents[i + 1] if i + 1 < len(currents) else 0
                        magnitude = (real**2 + imag**2)**0.5
                        current_magnitudes.append(magnitude)
                    
                    # 检查过载情况
                    max_current = max(current_magnitudes) if current_magnitudes else 0
                    
                    # 使用正常额定电流作为过载判断标准
                    if max_current > norm_amps:
                        overloads += 1
                        overload_ratio = max_current / norm_amps
                        overload_details.append({
                            'line': line_name,
                            'current': max_current,
                            'rating': norm_amps,
                            'ratio': overload_ratio
                        })
                        
                        # 记录严重过载（超过紧急额定值）
                        if max_current > emerg_amps and emerg_amps > 0:
                            logger.warning(f"线路{line_name}严重过载: {max_current:.2f}A > {emerg_amps:.2f}A")
                            
                except Exception as e:
                    logger.debug(f"无法获取线路{line_name}的电流信息: {e}")
                    continue
                
        except Exception as e:
            logger.warning(f"线路过载检测失败: {e}")
            
        # 记录过载详情（用于调试）
        if overload_details and logger.isEnabledFor(logging.DEBUG):
            for detail in overload_details:
                logger.debug(f"过载线路: {detail['line']}, 电流: {detail['current']:.2f}A, "
                            f"额定: {detail['rating']:.2f}A, 比例: {detail['ratio']:.2f}")
        
        return overloads
    
    def _check_restoration_complete(self) -> bool:
        """检查恢复是否完成"""
        # 当恢复率达到配置阈值以上时认为完成
        return self._get_restored_load_ratio() >= self.config.restoration_threshold 
    
    def _get_observations(self) -> Dict[str, Any]:
        """获取智能体观测"""
        obs = {}
        
        # 获取母线电压
        bus_voltages = {}
        for bus_name in self.all_bus_names:
            try:
                voltages = self.circuit.bus_voltage(bus_name)
                bus_voltages[bus_name] = [voltages[i] for i in range(len(voltages)) if i % 2 == 0]
            except:
                bus_voltages[bus_name] = [self.config.default_voltage]  # 配置的默认电压
        
        obs['bus_voltages'] = bus_voltages
        obs['energized_buses'] = list(self.energized_buses)
        obs['load_states'] = {name: data['enabled'] for name, data in self.load_info.items()}
        obs['line_states'] = {name: data['enabled'] for name, data in self.line_info.items()}
        obs['current_step'] = self.current_step
        obs['max_steps'] = self.config.max_episode_steps
        
        return obs
    
    def _get_global_state(self) -> Dict[str, Any]:
        """获取全局状态"""
        # 简化：全局状态与观测相同
        #TODO 未来可以加入一个转换函数
        return self._get_observations()
    
    def get_available_actions(self) -> List[List[int]]:
        """获取可用动作"""
        avail_actions = []
        
        for i in range(self.n_agents):
            if self.agent_types[i] == 'switch':
                # 开关智能体：可以操作的线路数量+1（不操作）
                n_actions = len([line for line in self.faultable_lines 
                               if line not in self.fault_lines]) + 1
                avail_actions.append([1] * n_actions)
            
            elif self.agent_types[i] == 'pv':
                # PV智能体：配置的功率等级数量
                avail_actions.append([1] * self.config.pv_power_levels)
            
            elif self.agent_types[i] == 'load':
                # 负荷智能体：断开(0)或恢复(1)
                agent_idx = self.agent_indices[i]
                load_agent = self.load_agents[agent_idx]
                
                avail = [1, 1]  # 默认都可用
                
                # 如果母线未通电，不能恢复
                if load_agent['bus'] not in self.energized_buses:
                    avail[1] = 0
                
                avail_actions.append(avail)
        
        return avail_actions
    
    def seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        random.seed(seed)
    
    def close(self):
        """关闭环境"""
        # 清理OpenDSS资源
        pass