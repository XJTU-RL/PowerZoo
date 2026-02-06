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
    from envs.vvc.vvc.circuit import Circuits
    from envs.dsr.core.loadprofile import LoadProfile
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
        from utils.path_utils import get_system_folder

        # 获取PowerZoo配置
        powerzoo_config = self.config.to_powerzoo_config()

        # 构建DSS文件路径（使用统一路径工具）
        dss_folder = str(get_system_folder(self.config.system_name))
        dss_file_path = os.path.join(dss_folder, powerzoo_config['dss_file'])
        
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
            powerzoo_config['dss_file'],
            worker_idx=self.worker_idx
        )
        # DSR LoadProfile 不生成动态负荷曲线，使用静态文件数
        self.num_profiles = max(len(self.load_profile.get_loadshape_files()), 1)
        
        logger.info(f"OpenDSS电路初始化完成: {self.n_bus}个母线, {len(self.line_info)}条线路")
        
        # 执行电路健康检查
        self._validate_circuit_health()
    
    def _validate_circuit_health(self):
        """验证电路健康状态"""
        try:
            # 执行初始求解
            self.circuit.dss.ActiveCircuit.Solution.Solve()
            converged = self.circuit.dss.ActiveCircuit.Solution.Converged
            
            if not converged:
                logger.warning("电路初始化后未收敛，可能存在配置问题")
            
            # 检查母线电压获取能力
            try:
                all_voltages = self.circuit.dss.ActiveCircuit.AllBusVmag
                if len(all_voltages) < len(self.all_bus_names):
                    logger.warning(f"母线电压数量不匹配: 期望{len(self.all_bus_names)}，实际{len(all_voltages)}")
            except Exception as e:
                logger.warning(f"无法获取AllBusVmag，将使用逐个获取方式: {e}")
            
            logger.info("电路健康检查完成")
            
        except Exception as e:
            logger.error(f"电路健康检查失败: {e}")
    
    def _get_line_info(self):
        """获取线路信息"""
        self.line_info = {}
        self.circuit.dss.ActiveCircuit.Lines.First
        while True:
            line_name = self.circuit.dss.ActiveCircuit.Lines.Name
            # 使用正确的DSS Python API方式获取Bus1和Bus2信息
            bus_names = self.circuit.dss.ActiveCircuit.ActiveCktElement.BusNames
            bus1 = bus_names[0].split('.')[0].lower() if len(bus_names) > 0 else ''
            bus2 = bus_names[1].split('.')[0].lower() if len(bus_names) > 1 else ''
            # 使用正确的DSS Python API方式获取Enabled状态
            enabled = self.circuit.dss.ActiveCircuit.ActiveCktElement.Enabled
            
            self.line_info[line_name] = {
                'bus1': bus1,
                'bus2': bus2,
                'enabled': enabled,
                'original_enabled': enabled  # 保存原始状态
            }
            # 检查
            if self.circuit.dss.ActiveCircuit.Lines.Next == 0:
                break
    
    def _get_load_info(self):
        """获取负荷信息"""
        self.load_info = {}
        load_names = self.circuit.dss.ActiveCircuit.Loads.AllNames
        
        for load_name in load_names:
            self.circuit.dss.ActiveCircuit.Loads.Name = load_name
            # 使用正确的DSS Python API方式获取Bus1信息
            bus_names = self.circuit.dss.ActiveCircuit.ActiveCktElement.BusNames
            bus = bus_names[0].split('.')[0].lower() if bus_names else ''
            kw = self.circuit.dss.ActiveCircuit.Loads.kW
            # 使用正确的DSS Python API方式获取Enabled状态
            enabled = self.circuit.dss.ActiveCircuit.ActiveCktElement.Enabled
            
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
        self.use_aggregation = agent_config.get('use_aggregation', False)
        self.aggregation_ratio = agent_config.get('aggregation_ratio', 1.0)
        
        load_names = list(self.load_info.keys())
        self.load_agents = []
        self.load_agent_mapping = {}  # 智能体到负荷的映射
        
        if self.use_aggregation and self.aggregation_ratio > 1:
            # 使用聚合模式
            self._init_aggregated_load_agents(load_names)
        else:
            # 一对一模式，一个智能体对应一个负荷，适用于小网络
            self._init_individual_load_agents(load_names)
        
        logger.info(f"智能体初始化完成: 1个开关 + {self.n_pv_agents}个PV + {self.n_load_agents}个负荷智能体")
        if self.use_aggregation and self.aggregation_ratio > 1:
            logger.info(f"负荷聚合模式: {len(load_names)}个负荷聚合到{self.n_load_agents}个智能体")
    
    def _init_individual_load_agents(self, load_names: List[str]):
        """初始化一对一负荷智能体"""
        for i, load_name in enumerate(load_names):
            if i >= self.n_load_agents:
                break  # 限制智能体数量
                
            load_data = self.load_info[load_name]
            self.agent_types.append('load')
            self.agent_indices.append(i)
            
            agent_id = 1 + self.n_pv_agents + i
            self.agent_bus_mapping[agent_id] = load_data['bus']
            
            self.load_agents.append({
                'agent_id': agent_id,
                'load_name': load_name,
                'bus': load_data['bus'],
                'priority': load_data['priority'],
                'kw': load_data['kw'],
                'managed_loads': [load_name]  # 管理的负荷列表
            })
            
            self.load_agent_mapping[agent_id] = [load_name]
    
    def _init_aggregated_load_agents(self, load_names: List[str]):
        """初始化聚合负荷智能体"""
        # 根据聚合方法对负荷进行分组
        load_groups = self._group_loads_for_aggregation(load_names)
        
        for i, (group_key, group_loads) in enumerate(load_groups.items()):
            if i >= self.n_load_agents:
                break
                
            self.agent_types.append('load')
            self.agent_indices.append(i)
            
            agent_id = 1 + self.n_pv_agents + i
            
            # 选择代表性母线（优先级最高或功率最大的负荷所在母线）
            representative_load = self._select_representative_load(group_loads)
            representative_bus = self.load_info[representative_load]['bus']
            
            self.agent_bus_mapping[agent_id] = representative_bus
            
            # 计算聚合信息
            total_kw = sum(self.load_info[load]['kw'] for load in group_loads)
            avg_priority = np.mean([self.load_info[load]['priority'] for load in group_loads])
            
            self.load_agents.append({
                'agent_id': agent_id,
                'load_name': f"aggregated_load_{i}",  # 聚合负荷的虚拟名称
                'bus': representative_bus,
                'priority': round(avg_priority),
                'kw': total_kw,
                'managed_loads': group_loads  # 管理的负荷列表
            })
            
            self.load_agent_mapping[agent_id] = group_loads
    
    def _group_loads_for_aggregation(self, load_names: List[str]) -> Dict[str, List[str]]:
        """根据聚合方法对负荷进行分组"""
        if self.config.load_aggregation_method == "zone":
            # 基于区域（母线）的聚合
            return self._group_loads_by_zone(load_names)
        elif self.config.load_aggregation_method == "priority":
            # 基于优先级的聚合
            return self._group_loads_by_priority(load_names)
        elif self.config.load_aggregation_method == "random":
            # 随机聚合
            return self._group_loads_randomly(load_names)
        else:
            # 默认使用区域聚合
            logger.warning(f"未知的聚合方法: {self.config.load_aggregation_method}, 使用区域聚合")
            return self._group_loads_by_zone(load_names)
    
    def _group_loads_by_zone(self, load_names: List[str]) -> Dict[str, List[str]]:
        """基于区域（母线）的负荷分组"""
        # 构建网络拓扑图
        G = nx.Graph()
        for line_name, line_data in self.line_info.items():
            if line_data['original_enabled']:  # 使用原始状态
                G.add_edge(line_data['bus1'], line_data['bus2'])
        
        # 按母线对负荷进行初步分组
        bus_loads = {}
        for load_name in load_names:
            bus = self.load_info[load_name]['bus']
            if bus not in bus_loads:
                bus_loads[bus] = []
            bus_loads[bus].append(load_name)
        
        # 使用社区检测或简单的距离聚类
        groups = {}
        loads_per_group = max(1, len(load_names) // self.n_load_agents)
        
        # 简单实现：将相邻母线的负荷聚合在一起
        #TODO 可以实现更复杂的负荷聚类
        assigned_loads = set()
        group_id = 0
        
        for bus, loads in bus_loads.items():
            if any(load in assigned_loads for load in loads):
                continue
                
            group_key = f"zone_{group_id}"
            groups[group_key] = []
            
            # 添加当前母线的负荷
            for load in loads:
                if load not in assigned_loads:
                    groups[group_key].append(load)
                    assigned_loads.add(load)
            
            # 如果组还不够大，添加相邻母线的负荷
            if bus in G:
                neighbors = list(G.neighbors(bus))
                for neighbor in neighbors:
                    if len(groups[group_key]) >= loads_per_group:
                        break
                    if neighbor in bus_loads:
                        for load in bus_loads[neighbor]:
                            if load not in assigned_loads and len(groups[group_key]) < loads_per_group:
                                groups[group_key].append(load)
                                assigned_loads.add(load)
            
            if groups[group_key]:  # 只保留非空组
                group_id += 1
            else:
                del groups[group_key]
        
        # 处理剩余未分配的负荷
        unassigned = [load for load in load_names if load not in assigned_loads]
        if unassigned:
            # 分配到现有组或创建新组
            for load in unassigned:
                # 找到最小的组
                min_group = min(groups.keys(), key=lambda k: len(groups[k]))
                groups[min_group].append(load)
        
        return groups
    
    def _group_loads_by_priority(self, load_names: List[str]) -> Dict[str, List[str]]:
        """基于优先级的负荷分组"""
        # 按优先级排序负荷
        sorted_loads = sorted(load_names, key=lambda x: self.load_info[x]['priority'])
        
        groups = {}
        loads_per_group = max(1, len(load_names) // self.n_load_agents)
        
        for i in range(self.n_load_agents):
            group_key = f"priority_group_{i}"
            start_idx = i * loads_per_group
            end_idx = start_idx + loads_per_group if i < self.n_load_agents - 1 else len(sorted_loads)
            
            groups[group_key] = sorted_loads[start_idx:end_idx]
            
            if not groups[group_key]:  # 删除空组
                del groups[group_key]
        
        return groups
    
    def _group_loads_randomly(self, load_names: List[str]) -> Dict[str, List[str]]:
        """随机负荷分组"""
        # 随机打乱负荷顺序
        shuffled_loads = load_names.copy()
        random.shuffle(shuffled_loads)
        
        groups = {}
        loads_per_group = max(1, len(load_names) // self.n_load_agents)
        
        for i in range(self.n_load_agents):
            group_key = f"random_group_{i}"
            start_idx = i * loads_per_group
            end_idx = start_idx + loads_per_group if i < self.n_load_agents - 1 else len(shuffled_loads)
            
            groups[group_key] = shuffled_loads[start_idx:end_idx]
            
            if not groups[group_key]:  # 删除空组
                del groups[group_key]
        
        return groups
    
    def _select_representative_load(self, load_names: List[str]) -> str:
        """选择代表性负荷（优先级最高或功率最大）"""
        if not load_names:
            return None
            
        # 首先按优先级选择（优先级数字越小越重要）
        min_priority = min(self.load_info[load]['priority'] for load in load_names)
        high_priority_loads = [load for load in load_names 
                              if self.load_info[load]['priority'] == min_priority]
        
        # 在同优先级中选择功率最大的
        return max(high_priority_loads, key=lambda x: self.load_info[x]['kw'])
    
    def _init_fault_management(self):
        """初始化故障管理"""
        # 可故障的线路（排除重要的主干线路）
        all_lines = list(self.line_info.keys())
        # TODO 简化：随机选择可能故障的线路
        # TODO 后面可以按照场景来分配故障线路
        
        self.faultable_lines = random.sample(all_lines, min(len(all_lines), self.config.max_faultable_lines))
        
        # 分布式发电机（黑启动电源）位置
        self.dg_buses = random.sample(self.all_bus_names, min(len(self.all_bus_names), self.config.n_dg))
        
        logger.info(f"故障管理初始化: {len(self.faultable_lines)}条可故障线路, {len(self.dg_buses)}个DG")
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        self.current_step = 0
        self.done = False
        
        # DSR 使用静态负荷，不切换 loadprofile
        
        # 重置电路
        self.circuit.reset()
        
        # 生成随机故障
        self._generate_random_faults()
        
        # 初始化所有设备状态
        self._reset_device_states()
        
        # 更新通电状态
        self._update_energized_buses()
        
        # 获取初始观测和状态
        #TODO 这里暂且将状态空间等同于观测空间，后面可以深化实现
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
            # 使用DSS文本命令设置线路禁用状态
            self.circuit.dss.Text.Command = f'line.{line_name}.enabled=no'
            self.line_info[line_name]['enabled'] = False
        
        logger.info(f"生成{len(self.fault_lines)}个故障: {self.fault_lines}")
    
    def _reset_device_states(self):
        """重置设备状态"""
        # 断开所有开关（除了基本的主干线路）
        for line_name, line_data in self.line_info.items():
            if line_name not in self.fault_lines:
                # 保持一些基本连接，断开其他
                if random.random() < self.config.line_disconnect_prob:  # 配置的概率断开
                    # 使用DSS文本命令设置线路禁用状态
                    self.circuit.dss.Text.Command = f'line.{line_name}.enabled=no'
                    line_data['enabled'] = False
        
        # 重置PV输出
        for pv_agent in self.pv_agents:
            pv_agent['current_power'] = 0.0
        
        # 断开所有负荷
        for load_name in self.load_info:
            # 使用DSS文本命令设置负荷禁用状态
            self.circuit.dss.Text.Command = f'load.{load_name}.enabled=no'
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
        
        # 记录动作前的系统状态（用于对比分析）
        pre_action_state = self._capture_system_state()
        
        # 执行智能体动作
        action_results = self._execute_actions(actions)
        
        # 运行潮流计算
        try:
            # 确保电路状态正确
            self.circuit.dss.ActiveCircuit.Solution.Solve()
            converged = self.circuit.dss.ActiveCircuit.Solution.Converged
            
            # 验证求解质量
            if not converged:
                logger.warning(f"步骤 {self.current_step}: 潮流计算未收敛")
                # 尝试重新求解
                self.circuit.dss.ActiveCircuit.Solution.Solve()
                converged = self.circuit.dss.ActiveCircuit.Solution.Converged
                
        except Exception as e:
            logger.warning(f"潮流计算失败: {e}")
            converged = False
        
        # 更新通电状态
        self._update_energized_buses()
        
        # 记录动作后的系统状态
        post_action_state = self._capture_system_state()
        
        # 计算奖励
        rewards = self._calculate_rewards(converged)
        
        # 检查终止条件
        self.done = (not converged or 
                    self.current_step >= self.config.max_episode_steps or
                    self._check_restoration_complete())
        
        # 获取观测和状态
        obs = self._get_observations()
        state = self._get_global_state()
        
        # 生成增强的信息字典
        info = self._generate_enhanced_info(
            converged, actions, action_results, 
            pre_action_state, post_action_state, rewards
        )
        
        return obs, state, rewards, self.done, info
    
    def _execute_actions(self, actions: List[int]):
        """执行智能体动作"""
        # 开关智能体动作
        switch_action = actions[0]
        # 确保switch_action是标量值
        if hasattr(switch_action, 'item'):
            switch_action = switch_action.item()
        switch_action = int(switch_action)
        
        if switch_action > 0 and switch_action <= len(self.faultable_lines):
            action_results['switch']['attempted'] = 1
            # 选择操作的线路（排除故障线路）
            available_lines = [line for line in self.faultable_lines 
                             if line not in self.fault_lines]
            if available_lines and switch_action <= len(available_lines):
                line_name = available_lines[switch_action - 1]
                # 切换线路状态
                current_state = self.line_info[line_name]['enabled']
                new_state = not current_state
                
                try:
                    # 使用DSS文本命令设置线路启用/禁用状态
                    enabled_str = 'yes' if new_state else 'no'
                    self.circuit.dss.Text.Command = f'line.{line_name}.enabled={enabled_str}'
                    self.line_info[line_name]['enabled'] = new_state
                    action_results['switch']['successful'] = 1
                    action_results['switch']['line_name'] = line_name
                    action_results['switch']['new_state'] = new_state
                except:
                    action_results['switch']['failed'] = 1
        
        # PV智能体动作
        for i, pv_agent in enumerate(self.pv_agents):
            if 1 + i < len(actions):
                pv_action = actions[1 + i]
                # 确保pv_action是标量值
                if hasattr(pv_action, 'item'):
                    pv_action = pv_action.item()
                pv_action = int(pv_action)
                
                # 设置PV输出功率（配置的等级范围）
                max_level = self.config.pv_power_levels - 1
                if 0 <= pv_action <= max_level:
                    power_ratio = pv_action / max_level
                    old_power = pv_agent['current_power']
                    new_power = power_ratio * pv_agent['max_power']
                    pv_agent['current_power'] = new_power
                    
                    power_change = new_power - old_power
                    action_results['pv']['adjustments'].append({
                        'agent_id': 1 + i,
                        'old_power': old_power,
                        'new_power': new_power,
                        'change': power_change
                    })
                    action_results['pv']['total_power_change'] += power_change
        
        # 负荷智能体动作
        for i, load_agent in enumerate(self.load_agents):
            action_idx = 1 + self.n_pv_agents + i
            if action_idx < len(actions):
                load_action = actions[action_idx]
                # 确保load_action是标量值
                if hasattr(load_action, 'item'):
                    load_action = load_action.item()
                load_action = int(load_action)
                
                # 处理聚合负荷智能体管理的所有负荷
                managed_loads = load_agent.get('managed_loads', [load_agent['load_name']])
                
                for load_name in managed_loads:
                    if load_name not in self.load_info:
                        continue
                        
                    if load_action == 1:  # 尝试恢复负荷
                        action_results['load']['attempted_restore'] += 1
                        # 检查负荷所在母线是否通电
                        load_bus = self.load_info[load_name]['bus']
                        if load_bus in self.energized_buses:
                            try:
                                # 使用DSS文本命令设置负荷启用状态
                                self.circuit.dss.Text.Command = f'load.{load_name}.enabled=yes'
                                self.load_info[load_name]['enabled'] = True
                                action_results['load']['successful_restore'] += 1
                            except:
                                action_results['load']['failed_restore'] += 1
                        else:
                            action_results['load']['failed_restore'] += 1
                    elif load_action == 0:  # 断开负荷
                        # 使用DSS文本命令设置负荷禁用状态
                        self.circuit.dss.Text.Command = f'load.{load_name}.enabled=no'
                        self.load_info[load_name]['enabled'] = False
        
        return action_results
    
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
        self.overload_details = []  # 保存为实例变量，供其他方法使用
        
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
                        self.overload_details.append({
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
        if self.overload_details and logger.isEnabledFor(logging.DEBUG):
            for detail in self.overload_details:
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
        bus_voltages = {}
        
        # 首先验证电路状态
        try:
            circuit_solved = self.circuit.dss.ActiveCircuit.Solution.Converged
            if not circuit_solved:
                logger.debug("电路未收敛，使用默认电压值")
        except:
            circuit_solved = False
        
        # 获取母线电压
        if circuit_solved:
            try:
                # 方法1：使用AllBusVmag获取所有母线电压（推荐）
                all_voltages = self.circuit.dss.ActiveCircuit.AllBusVmag
                if len(all_voltages) >= len(self.all_bus_names):
                    for i, bus_name in enumerate(self.all_bus_names):
                        voltage_magnitude = all_voltages[i]
                        if np.isnan(voltage_magnitude) or np.isinf(voltage_magnitude):
                            voltage_magnitude = self.config.default_voltage
                        else:
                            voltage_magnitude = np.clip(voltage_magnitude, 0.0, 2.0)
                        bus_voltages[bus_name] = [voltage_magnitude]
                else:
                    # 回退到逐个获取
                    self._get_bus_voltages_individually(bus_voltages)
            except Exception as e:
                logger.debug(f"批量获取母线电压失败: {e}，回退到逐个获取")
                self._get_bus_voltages_individually(bus_voltages)
        else:
            # 电路未收敛，使用默认电压
            for bus_name in self.all_bus_names:
                bus_voltages[bus_name] = [self.config.default_voltage]
        
        obs['bus_voltages'] = bus_voltages
        obs['energized_buses'] = list(self.energized_buses)
        obs['load_states'] = {name: data['enabled'] for name, data in self.load_info.items()}
        obs['line_states'] = {name: data['enabled'] for name, data in self.line_info.items()}
        obs['current_step'] = self.current_step
        obs['max_steps'] = self.config.max_episode_steps
        
        return obs
    
    def _get_bus_voltages_individually(self, bus_voltages: Dict[str, List[float]]):
        """逐个获取母线电压的备用方法"""
        for bus_name in self.all_bus_names:
            try:
                voltages = self.circuit.bus_voltage(bus_name)
                # 处理numpy数组的情况，避免歧义错误
                if voltages is not None and len(voltages) > 0:
                    # 提取电压幅值（每隔一个元素）
                    voltage_mags = [voltages[i] for i in range(len(voltages)) if i % 2 == 0]
                    
                    # 检查并处理无效值
                    cleaned_voltages = []
                    for v in voltage_mags:
                        if np.isnan(v) or np.isinf(v):
                            v = self.config.default_voltage
                        else:
                            # 限制电压在合理范围内
                            v = np.clip(v, 0.0, 2.0)
                        cleaned_voltages.append(v)
                    
                    bus_voltages[bus_name] = cleaned_voltages if cleaned_voltages else [self.config.default_voltage]
                else:
                    bus_voltages[bus_name] = [self.config.default_voltage]
            except Exception as e:
                # 只在调试模式下打印详细错误
                if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
                    logger.debug(f"获取母线 {bus_name} 电压失败: {e}")
                bus_voltages[bus_name] = [self.config.default_voltage]
    
    def _get_global_state(self) -> Dict[str, Any]:
        """获取全局状态"""
        # 简化：全局状态与观测相同
        #TODO 未来可以加入一个转换函数
        return self._get_observations()
    
    def get_available_actions(self) -> List[List[int]]:
        """获取可用动作"""
        avail_actions = []
        
        # 计算统一的动作空间大小
        max_switch_actions = len(self.faultable_lines) + 1
        max_pv_actions = self.config.pv_power_levels
        max_load_actions = self.config.load_action_levels
        max_actions = max(max_switch_actions, max_pv_actions, max_load_actions)
        
        for i in range(self.n_agents):
            # 初始化为统一长度的可用动作列表
            avail = [0] * max_actions
            
            if self.agent_types[i] == 'switch':
                # 开关智能体：不操作动作总是可用
                avail[0] = 1
                
                # 检查每条可故障线路是否可操作
                for j, line in enumerate(self.faultable_lines):
                    if line not in self.fault_lines:
                        avail[j + 1] = 1
            
            elif self.agent_types[i] == 'pv':
                # PV智能体：所有功率等级都可用
                for j in range(self.config.pv_power_levels):
                    avail[j] = 1
            
            elif self.agent_types[i] == 'load':
                # 负荷智能体：断开动作总是可用
                avail[0] = 1
                
                # 检查恢复动作是否可用
                if self.config.load_action_levels > 1:
                    agent_idx = self.agent_indices[i]
                    load_agent = self.load_agents[agent_idx]
                    
                    # 对于聚合负荷智能体，检查所有管理的负荷
                    managed_loads = load_agent.get('managed_loads', [load_agent['load_name']])
                    
                    # 如果有任何管理的负荷所在母线通电，就可以恢复
                    can_restore = False
                    for load_name in managed_loads:
                        if load_name in self.load_info:
                            load_bus = self.load_info[load_name]['bus']
                            if load_bus in self.energized_buses:
                                can_restore = True
                                break
                    
                    if can_restore:
                        avail[1] = 1
            
            avail_actions.append(avail)
        
        return avail_actions
    
    def seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        random.seed(seed)
    
    def _get_total_restored_load(self) -> float:
        """获取总恢复负荷（kW）"""
        total = 0.0
        for load_name, load_data in self.load_info.items():
            if load_data['enabled']:
                total += load_data['kw']
        return total
    
    def _get_priority_restoration_status(self) -> Dict[int, Dict[str, float]]:
        """获取按优先级的恢复状态"""
        priority_status = {}
        
        for priority in range(1, self.config.max_priority_level + 1):
            total_kw = 0.0
            restored_kw = 0.0
            count_total = 0
            count_restored = 0
            
            for load_name, load_data in self.load_info.items():
                if load_data['priority'] == priority:
                    total_kw += load_data['kw']
                    count_total += 1
                    if load_data['enabled']:
                        restored_kw += load_data['kw']
                        count_restored += 1
            
            priority_status[priority] = {
                'total_kw': total_kw,
                'restored_kw': restored_kw,
                'restoration_ratio': restored_kw / total_kw if total_kw > 0 else 0,
                'total_count': count_total,
                'restored_count': count_restored,
            }
        
        return priority_status
    
    def _calculate_restoration_rate(self, pre_state: Dict, post_state: Dict) -> float:
        """计算恢复速率（本步恢复的负荷数量）"""
        loads_restored = post_state['restored_loads'] - pre_state['restored_loads']
        return loads_restored
    
    def _calculate_action_effectiveness(self, pre_state: Dict, post_state: Dict) -> float:
        """计算动作有效性（0-1之间）"""
        # 基于系统改善程度计算
        improvements = 0
        total_metrics = 0
        
        # 通电母线增加
        if post_state['energized_buses'] > pre_state['energized_buses']:
            improvements += 1
        total_metrics += 1
        
        # 负荷恢复增加
        if post_state['restored_loads'] > pre_state['restored_loads']:
            improvements += 1
        total_metrics += 1
        
        # 线路连接增加（如果不是减少）
        if post_state['active_lines'] >= pre_state['active_lines']:
            improvements += 0.5
        total_metrics += 1
        
        return improvements / total_metrics if total_metrics > 0 else 0
    
    def _get_line_loading_details(self) -> Dict[str, Dict[str, float]]:
        """获取详细的线路负载信息"""
        line_loading = {}
        
        for line_name, line_data in self.line_info.items():
            if not line_data['enabled']:
                line_loading[line_name] = {'status': 'disabled', 'loading': 0}
                continue
            
            try:
                self.circuit.dss.ActiveCircuit.Lines.Name = line_name
                norm_amps = self.circuit.dss.ActiveCircuit.Lines.NormAmps
                
                if norm_amps > 0:
                    currents = self.circuit.edge_current(f"Line.{line_name}")
                    # 计算最大相电流
                    max_current = 0
                    for i in range(0, len(currents), 2):
                        real = currents[i]
                        imag = currents[i + 1] if i + 1 < len(currents) else 0
                        magnitude = (real**2 + imag**2)**0.5
                        max_current = max(max_current, magnitude)
                    
                    loading_percent = (max_current / norm_amps) * 100
                    
                    line_loading[line_name] = {
                        'status': 'active',
                        'current': max_current,
                        'rating': norm_amps,
                        'loading': loading_percent,
                        'overloaded': loading_percent > 100
                    }
                else:
                    line_loading[line_name] = {'status': 'no_rating', 'loading': 0}
                    
            except:
                line_loading[line_name] = {'status': 'error', 'loading': 0}
        
        return line_loading
    
    def _get_voltage_profile(self) -> Dict[str, Dict[str, Any]]:
        """获取电压分布信息"""
        voltage_profile = {}
        
        for bus_name in self.all_bus_names:
            try:
                voltages = self.circuit.bus_voltage(bus_name)
                phase_voltages = [voltages[i] for i in range(len(voltages)) if i % 2 == 0]
                
                min_v = min(phase_voltages) if phase_voltages else 0
                max_v = max(phase_voltages) if phase_voltages else 0
                avg_v = np.mean(phase_voltages) if phase_voltages else 0
                
                voltage_profile[bus_name] = {
                    'min': min_v,
                    'max': max_v,
                    'avg': avg_v,
                    'phases': len(phase_voltages),
                    'violation': min_v < self.config.v_min or max_v > self.config.v_max,
                    'energized': bus_name in self.energized_buses
                }
            except:
                voltage_profile[bus_name] = {
                    'min': 0,
                    'max': 0,
                    'avg': 0,
                    'phases': 0,
                    'violation': False,
                    'energized': False
                }
        
        return voltage_profile
    
    def _calculate_topology_metrics(self) -> Dict[str, Any]:
        """计算网络拓扑指标"""
        # 构建当前网络图
        G = nx.Graph()
        G.add_nodes_from(self.all_bus_names)
        
        for line_name, line_data in self.line_info.items():
            if line_data['enabled']:
                G.add_edge(line_data['bus1'], line_data['bus2'])
        
        # 计算连通分量
        components = list(nx.connected_components(G))
        
        # 找出有电源的分量
        powered_components = []
        for component in components:
            if any(bus in self.dg_buses for bus in component):
                powered_components.append(component)
        
        # 计算指标
        metrics = {
            'total_components': len(components),
            'powered_components': len(powered_components),
            'largest_component_size': len(max(components, key=len)) if components else 0,
            'isolated_buses': len([n for n in G.nodes() if G.degree(n) == 0]),
            'average_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            'graph_density': nx.density(G) if G.number_of_nodes() > 1 else 0,
        }
        
        return metrics
    
    def close(self):
        """关闭环境"""
        # 清理OpenDSS资源
        pass
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """捕获当前系统状态快照"""
        state = {
            'timestamp': self.current_step,
            'energized_buses': len(self.energized_buses),
            'restored_loads': sum(1 for load in self.load_info.values() if load['enabled']),
            'active_lines': sum(1 for line in self.line_info.values() if line['enabled']),
        }
        
        # 获取系统级指标
        if hasattr(self.circuit.dss.ActiveCircuit, 'Losses'):
            losses = self.circuit.dss.ActiveCircuit.Losses
            state['power_losses'] = {
                'active': losses[0] if len(losses) > 0 else 0,
                'reactive': losses[1] if len(losses) > 1 else 0
            }
        
        return state
    
    def _generate_enhanced_info(self, converged: bool, actions: List[int], 
                                action_results: Dict[str, Any],
                                pre_state: Dict[str, Any], 
                                post_state: Dict[str, Any],
                                rewards: List[float]) -> Dict[str, Any]:
        """生成增强的信息字典"""
        # 基础信息
        info = {
            'converged': converged,
            'restored_load_ratio': self._get_restored_load_ratio(),
            'energized_buses': len(self.energized_buses),
            'total_buses': self.n_bus,
            'current_step': self.current_step,
            'fault_lines': self.fault_lines,
        }
        
        # 系统状态信息
        info['system_state'] = {
            'voltage_violations': self._get_voltage_violations(),
            'line_overloads': len(self.overload_details) if hasattr(self, 'overload_details') else 0,
            'overload_details': self.overload_details if hasattr(self, 'overload_details') else [],
            'power_losses': post_state.get('power_losses', {}),
            'topology_changes': {
                'lines_switched': post_state['active_lines'] - pre_state['active_lines'],
                'loads_restored': post_state['restored_loads'] - pre_state['restored_loads'],
                'buses_energized': post_state['energized_buses'] - pre_state['energized_buses'],
            }
        }
        
        # 智能体行为信息
        info['agent_behavior'] = {
            'actions': actions,
            'action_results': action_results,
            'rewards': rewards,
            'action_effectiveness': self._calculate_action_effectiveness(pre_state, post_state),
        }
        
        # 恢复过程信息
        info['restoration_progress'] = {
            'total_load_restored': self._get_total_restored_load(),
            'priority_restoration': self._get_priority_restoration_status(),
            'restoration_rate': self._calculate_restoration_rate(pre_state, post_state),
            'completion_ratio': self._get_restored_load_ratio(),
        }
        
        # 详细的线路负载信息
        info['line_loading'] = self._get_line_loading_details()
        
        # 详细的电压分布信息
        info['voltage_profile'] = self._get_voltage_profile()
        
        # 网络拓扑指标
        info['topology_metrics'] = self._calculate_topology_metrics()
        
        return info
    
    def _execute_actions(self, actions: List[int]) -> Dict[str, Any]:
        """执行智能体动作"""
        action_results = {
            'switch': {'attempted': 0, 'successful': 0, 'failed': 0},
            'pv': {'adjustments': [], 'total_power_change': 0},
            'load': {'attempted_restore': 0, 'successful_restore': 0, 'failed_restore': 0},
        }
        
        # 开关智能体动作
        switch_action = actions[0]
        # 确保 switch_action 是标量整数
        if hasattr(switch_action, 'item'):
            switch_action = switch_action.item()
        switch_action = int(switch_action)
        
        if switch_action > 0 and switch_action <= len(self.faultable_lines):
            action_results['switch']['attempted'] = 1
            # 选择操作的线路（排除故障线路）
            available_lines = [line for line in self.faultable_lines 
                             if line not in self.fault_lines]
            if available_lines and switch_action <= len(available_lines):
                line_name = available_lines[switch_action - 1]
                # 切换线路状态
                current_state = self.line_info[line_name]['enabled']
                new_state = not current_state
                
                try:
                    # 使用DSS文本命令设置线路启用/禁用状态
                    enabled_str = 'yes' if new_state else 'no'
                    self.circuit.dss.Text.Command = f'line.{line_name}.enabled={enabled_str}'
                    self.line_info[line_name]['enabled'] = new_state
                    action_results['switch']['successful'] = 1
                    action_results['switch']['line_name'] = line_name
                    action_results['switch']['new_state'] = new_state
                except:
                    action_results['switch']['failed'] = 1
        
        # PV智能体动作
        for i, pv_agent in enumerate(self.pv_agents):
            if 1 + i < len(actions):
                pv_action = actions[1 + i]
                # 设置PV输出功率（配置的等级范围）
                max_level = self.config.pv_power_levels - 1
                if 0 <= pv_action <= max_level:
                    power_ratio = pv_action / max_level
                    old_power = pv_agent['current_power']
                    new_power = power_ratio * pv_agent['max_power']
                    pv_agent['current_power'] = new_power
                    
                    power_change = new_power - old_power
                    action_results['pv']['adjustments'].append({
                        'agent_id': 1 + i,
                        'old_power': old_power,
                        'new_power': new_power,
                        'change': power_change
                    })
                    action_results['pv']['total_power_change'] += power_change
        
        # 负荷智能体动作
        for i, load_agent in enumerate(self.load_agents):
            action_idx = 1 + self.n_pv_agents + i
            if action_idx < len(actions):
                load_action = actions[action_idx]
                
                # 处理聚合负荷智能体管理的所有负荷
                managed_loads = load_agent.get('managed_loads', [load_agent['load_name']])
                
                for load_name in managed_loads:
                    if load_name not in self.load_info:
                        continue
                        
                    if load_action == 1:  # 尝试恢复负荷
                        action_results['load']['attempted_restore'] += 1
                        # 检查负荷所在母线是否通电
                        load_bus = self.load_info[load_name]['bus']
                        if load_bus in self.energized_buses:
                            try:
                                # 使用DSS文本命令设置负荷启用状态
                                self.circuit.dss.Text.Command = f'load.{load_name}.enabled=yes'
                                self.load_info[load_name]['enabled'] = True
                                action_results['load']['successful_restore'] += 1
                            except:
                                action_results['load']['failed_restore'] += 1
                        else:
                            action_results['load']['failed_restore'] += 1
                    elif load_action == 0:  # 断开负荷
                        # 使用DSS文本命令设置负荷禁用状态
                        self.circuit.dss.Text.Command = f'load.{load_name}.enabled=no'
                        self.load_info[load_name]['enabled'] = False
        
        return action_results