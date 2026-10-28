# -*- coding: utf-8 -*-
"""
DSR Circuit Module
配电网恢复环境专用电路模块
基于OpenDSS的轻量化电路管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import os

try:
    import opendssdirect as dss
    DSS_AVAILABLE = True
except ImportError:
    DSS_AVAILABLE = False
    print("OpenDSS not available. Please install opendssdirect.")

# 设置日志
logger = logging.getLogger(__name__)


class DSRCircuit:
    """
    DSR环境专用电路类
    专注于配电网恢复场景的核心功能
    """
    
    def __init__(self, dss_file_path: str, validate_on_init: bool = True):
        """
        初始化DSR电路
        
        Args:
            dss_file_path: DSS文件路径
            validate_on_init: 是否在初始化时验证电路
        """
        if not DSS_AVAILABLE:
            raise ImportError("OpenDSS not available. Please install opendssdirect.")
        
        self.dss_file_path = dss_file_path
        self.dss = dss
        
        # 电路基础信息
        self.circuit_name = None
        self.source_bus = None
        self.all_bus_names = []
        self.n_buses = 0
        
        # 设备信息
        self.lines = {}  # 线路信息
        self.loads = {}  # 负荷信息
        self.generators = {}  # 发电机信息（包括PV和DG）
        self.transformers = {}  # 变压器信息
        self.switches = {}  # 开关信息
        
        # 状态信息
        self.is_solved = False
        self.converged = False
        
        # 初始化电路
        self._initialize_circuit()
        
        if validate_on_init:
            self._validate_circuit()
    
    def _initialize_circuit(self):
        """初始化OpenDSS电路"""
        try:
            # 清除现有电路
            self.dss.run_command("Clear")
            
            # 编译DSS文件
            if not os.path.exists(self.dss_file_path):
                raise FileNotFoundError(f"DSS file not found: {self.dss_file_path}")
            
            self.dss.run_command(f"Compile {self.dss_file_path}")
            
            # 获取电路基础信息
            self.circuit_name = self.dss.Circuit.Name()
            self.all_bus_names = self.dss.Circuit.AllBusNames()
            self.n_buses = len(self.all_bus_names)
            
            # 获取源母线
            self._identify_source_bus()
            
            # 加载设备信息
            self._load_device_info()
            
            logger.info(f"DSR电路初始化完成: {self.circuit_name}, {self.n_buses}个母线")
            
        except Exception as e:
            logger.error(f"电路初始化失败: {e}")
            raise
    
    def _identify_source_bus(self):
        """识别源母线"""
        # 常见的源母线名称
        common_source_names = ['sourcebus', 'source', '150', 'e192860']
        
        for bus_name in self.all_bus_names:
            if bus_name.lower() in [name.lower() for name in common_source_names]:
                self.source_bus = bus_name
                break
        
        if not self.source_bus:
            # 如果没有找到，使用第一个母线作为源母线
            self.source_bus = self.all_bus_names[0] if self.all_bus_names else None
            logger.warning(f"未找到标准源母线，使用 {self.source_bus} 作为源母线")
    
    def _load_device_info(self):
        """加载设备信息"""
        self._load_lines()
        self._load_loads()
        self._load_generators()
        self._load_transformers()
        self._load_switches()
    
    def _load_lines(self):
        """加载线路信息"""
        self.lines = {}
        line_names = self.dss.Lines.AllNames()
        
        for line_name in line_names:
            self.dss.Lines.Name(line_name)
            
            # 获取线路连接的母线
            bus_names = self.dss.CktElement.BusNames()
            bus1 = bus_names[0].split('.')[0] if len(bus_names) > 0 else ''
            bus2 = bus_names[1].split('.')[0] if len(bus_names) > 1 else ''
            
            self.lines[line_name] = {
                'bus1': bus1,
                'bus2': bus2,
                'enabled': self.dss.CktElement.Enabled(),
                'normal_amps': self.dss.Lines.NormAmps(),
                'emergency_amps': self.dss.Lines.EmergAmps(),
                'length': self.dss.Lines.Length(),
                'r1': self.dss.Lines.R1(),
                'x1': self.dss.Lines.X1(),
                'original_enabled': self.dss.CktElement.Enabled()  # 保存原始状态
            }
    
    def _load_loads(self):
        """加载负荷信息"""
        self.loads = {}
        load_names = self.dss.Loads.AllNames()
        
        for load_name in load_names:
            self.dss.Loads.Name(load_name)
            
            # 获取负荷连接的母线
            bus_names = self.dss.CktElement.BusNames()
            bus = bus_names[0].split('.')[0] if bus_names else ''
            
            self.loads[load_name] = {
                'bus': bus,
                'kw': self.dss.Loads.kW(),
                'kvar': self.dss.Loads.kvar(),
                'kv': self.dss.Loads.kV(),
                'enabled': self.dss.CktElement.Enabled(),
                'original_enabled': self.dss.CktElement.Enabled(),
                'original_kw': self.dss.Loads.kW(),
                'original_kvar': self.dss.Loads.kvar()
            }
    
    def _load_generators(self):
        """加载发电机信息（包括PV和DG）"""
        self.generators = {}
        gen_names = self.dss.Generators.AllNames()
        
        for gen_name in gen_names:
            self.dss.Generators.Name(gen_name)
            
            # 获取发电机连接的母线
            bus_names = self.dss.CktElement.BusNames()
            bus = bus_names[0].split('.')[0] if bus_names else ''
            
            self.generators[gen_name] = {
                'bus': bus,
                'kw': self.dss.Generators.kW(),
                'kvar': self.dss.Generators.kvar(),
                'kv': self.dss.Generators.kV(),
                'pf': self.dss.Generators.PF(),
                'enabled': self.dss.CktElement.Enabled(),
                'original_enabled': self.dss.CktElement.Enabled(),
                'original_kw': self.dss.Generators.kW()
            }
    
    def _load_transformers(self):
        """加载变压器信息"""
        self.transformers = {}
        trans_names = self.dss.Transformers.AllNames()
        
        for trans_name in trans_names:
            self.dss.Transformers.Name(trans_name)
            
            # 获取变压器连接的母线
            bus_names = self.dss.CktElement.BusNames()
            
            self.transformers[trans_name] = {
                'buses': [bus.split('.')[0] for bus in bus_names],
                'enabled': self.dss.CktElement.Enabled(),
                'kva': self.dss.Transformers.kVA(),
                'tap': self.dss.Transformers.Tap(),
                'original_enabled': self.dss.CktElement.Enabled()
            }
    
    def _load_switches(self):
        """加载开关信息（从线路中识别）"""
        self.switches = {}
        
        # 在DSR环境中，开关通常是特殊的线路或SwtControl元件
        # 这里简化处理，将短线路视为开关
        for line_name, line_data in self.lines.items():
            # 简单判断：长度很短的线路可能是开关
            if line_data['length'] < 0.01:  # 小于0.01英里的线路视为开关
                self.switches[line_name] = {
                    'bus1': line_data['bus1'],
                    'bus2': line_data['bus2'],
                    'enabled': line_data['enabled'],
                    'original_enabled': line_data['original_enabled'],
                    'type': 'line_switch'
                }
    
    def solve_circuit(self, mode: str = 'snapshot') -> bool:
        """
        求解电路
        
        Args:
            mode: 求解模式 ('snapshot', 'daily', 'yearly')
            
        Returns:
            是否收敛
        """
        try:
            if mode == 'snapshot':
                self.dss.Solution.Solve()
            elif mode == 'daily':
                self.dss.Solution.Mode(1)  # Daily mode
                self.dss.Solution.Solve()
            elif mode == 'yearly':
                self.dss.Solution.Mode(2)  # Yearly mode
                self.dss.Solution.Solve()
            
            self.converged = self.dss.Solution.Converged()
            self.is_solved = True
            
            if not self.converged:
                logger.warning("电路求解未收敛")
            
            return self.converged
            
        except Exception as e:
            logger.error(f"电路求解失败: {e}")
            self.is_solved = False
            self.converged = False
            return False
    
    def get_bus_voltages(self) -> Dict[str, float]:
        """
        获取所有母线电压（标幺值）
        
        Returns:
            母线电压字典 {bus_name: voltage_pu}
        """
        if not self.is_solved:
            logger.warning("电路未求解，返回默认电压")
            return {bus: 1.0 for bus in self.all_bus_names}
        
        try:
            voltages = {}
            for bus_name in self.all_bus_names:
                self.dss.Circuit.SetActiveBus(bus_name)
                # 获取母线电压幅值（kV）
                bus_voltage = self.dss.Bus.kVBase()
                if bus_voltage > 0:
                    actual_voltage = self.dss.Bus.VMagAngle()[0]  # 第一个相的电压幅值
                    voltage_pu = actual_voltage / bus_voltage
                else:
                    voltage_pu = 0.0
                voltages[bus_name] = voltage_pu
            
            return voltages
            
        except Exception as e:
            logger.error(f"获取母线电压失败: {e}")
            return {bus: 1.0 for bus in self.all_bus_names}
    
    def get_line_currents(self) -> Dict[str, float]:
        """
        获取线路电流（标幺值）
        
        Returns:
            线路电流字典 {line_name: current_pu}
        """
        if not self.is_solved:
            return {line: 0.0 for line in self.lines.keys()}
        
        try:
            currents = {}
            for line_name in self.lines.keys():
                self.dss.Lines.Name(line_name)
                if self.dss.CktElement.Enabled():
                    # 获取线路电流
                    line_currents = self.dss.CktElement.CurrentsMagAng()
                    if line_currents:
                        max_current = max(line_currents[::2])  # 取最大相电流
                        normal_amps = self.lines[line_name]['normal_amps']
                        current_pu = max_current / normal_amps if normal_amps > 0 else 0.0
                    else:
                        current_pu = 0.0
                else:
                    current_pu = 0.0
                
                currents[line_name] = current_pu
            
            return currents
            
        except Exception as e:
            logger.error(f"获取线路电流失败: {e}")
            return {line: 0.0 for line in self.lines.keys()}
    
    def set_line_status(self, line_name: str, enabled: bool) -> bool:
        """
        设置线路状态
        
        Args:
            line_name: 线路名称
            enabled: 是否启用
            
        Returns:
            操作是否成功
        """
        try:
            if line_name not in self.lines:
                logger.warning(f"线路 {line_name} 不存在")
                return False
            
            self.dss.Lines.Name(line_name)
            self.dss.CktElement.Enabled(enabled)
            
            # 更新内部状态
            self.lines[line_name]['enabled'] = enabled
            
            # 标记需要重新求解
            self.is_solved = False
            
            return True
            
        except Exception as e:
            logger.error(f"设置线路状态失败: {e}")
            return False
    
    def set_load_status(self, load_name: str, enabled: bool, kw: Optional[float] = None) -> bool:
        """
        设置负荷状态
        
        Args:
            load_name: 负荷名称
            enabled: 是否启用
            kw: 负荷功率（可选）
            
        Returns:
            操作是否成功
        """
        try:
            if load_name not in self.loads:
                logger.warning(f"负荷 {load_name} 不存在")
                return False
            
            self.dss.Loads.Name(load_name)
            self.dss.CktElement.Enabled(enabled)
            
            if kw is not None:
                self.dss.Loads.kW(kw)
                self.loads[load_name]['kw'] = kw
            
            # 更新内部状态
            self.loads[load_name]['enabled'] = enabled
            
            # 标记需要重新求解
            self.is_solved = False
            
            return True
            
        except Exception as e:
            logger.error(f"设置负荷状态失败: {e}")
            return False
    
    def set_generator_power(self, gen_name: str, kw: float) -> bool:
        """
        设置发电机功率
        
        Args:
            gen_name: 发电机名称
            kw: 功率值
            
        Returns:
            操作是否成功
        """
        try:
            if gen_name not in self.generators:
                logger.warning(f"发电机 {gen_name} 不存在")
                return False
            
            self.dss.Generators.Name(gen_name)
            self.dss.Generators.kW(kw)
            
            # 更新内部状态
            self.generators[gen_name]['kw'] = kw
            
            # 标记需要重新求解
            self.is_solved = False
            
            return True
            
        except Exception as e:
            logger.error(f"设置发电机功率失败: {e}")
            return False
    
    def get_energized_buses(self) -> Set[str]:
        """
        获取通电母线集合
        
        Returns:
            通电母线名称集合
        """
        if not self.is_solved:
            return set()
        
        try:
            energized_buses = set()
            voltages = self.get_bus_voltages()
            
            for bus_name, voltage in voltages.items():
                if voltage > 0.1:  # 电压大于0.1标幺值认为是通电
                    energized_buses.add(bus_name)
            
            return energized_buses
            
        except Exception as e:
            logger.error(f"获取通电母线失败: {e}")
            return set()
    
    def get_total_load_served(self) -> float:
        """
        获取总服务负荷
        
        Returns:
            总服务负荷（kW）
        """
        if not self.is_solved:
            return 0.0
        
        try:
            total_served = 0.0
            energized_buses = self.get_energized_buses()
            
            for load_name, load_data in self.loads.items():
                if load_data['enabled'] and load_data['bus'] in energized_buses:
                    total_served += load_data['kw']
            
            return total_served
            
        except Exception as e:
            logger.error(f"获取总服务负荷失败: {e}")
            return 0.0
    
    def get_total_load_capacity(self) -> float:
        """
        获取总负荷容量
        
        Returns:
            总负荷容量（kW）
        """
        return sum(load_data['original_kw'] for load_data in self.loads.values())
    
    def reset_to_original_state(self):
        """
        重置电路到原始状态
        """
        try:
            # 重置线路状态
            for line_name, line_data in self.lines.items():
                self.set_line_status(line_name, line_data['original_enabled'])
            
            # 重置负荷状态
            for load_name, load_data in self.loads.items():
                self.set_load_status(load_name, load_data['original_enabled'], load_data['original_kw'])
            
            # 重置发电机状态
            for gen_name, gen_data in self.generators.items():
                self.set_generator_power(gen_name, gen_data['original_kw'])
            
            # 标记需要重新求解
            self.is_solved = False
            
            logger.info("电路已重置到原始状态")
            
        except Exception as e:
            logger.error(f"重置电路状态失败: {e}")
    
    def _validate_circuit(self):
        """
        验证电路配置
        """
        try:
            # 执行初始求解
            success = self.solve_circuit()
            
            if not success:
                logger.warning("电路初始验证未通过")
            
            # 检查基本信息
            if not self.all_bus_names:
                raise ValueError("未找到母线信息")
            
            if not self.source_bus:
                raise ValueError("未找到源母线")
            
            logger.info(f"电路验证完成: {len(self.lines)}条线路, {len(self.loads)}个负荷, {len(self.generators)}个发电机")
            
        except Exception as e:
            logger.error(f"电路验证失败: {e}")
            raise
    
    def get_circuit_summary(self) -> Dict[str, Any]:
        """
        获取电路摘要信息
        
        Returns:
            电路摘要字典
        """
        return {
            'circuit_name': self.circuit_name,
            'n_buses': self.n_buses,
            'n_lines': len(self.lines),
            'n_loads': len(self.loads),
            'n_generators': len(self.generators),
            'n_transformers': len(self.transformers),
            'n_switches': len(self.switches),
            'source_bus': self.source_bus,
            'total_load_capacity': self.get_total_load_capacity(),
            'is_solved': self.is_solved,
            'converged': self.converged
        }


# 为了保持向后兼容性，提供原始Circuits类的别名
Circuits = DSRCircuit


# 辅助函数
def create_dsr_circuit(dss_file_path: str, **kwargs) -> DSRCircuit:
    """
    创建DSR电路实例的便捷函数
    
    Args:
        dss_file_path: DSS文件路径
        **kwargs: 其他参数
        
    Returns:
        DSRCircuit实例
    """
    return DSRCircuit(dss_file_path, **kwargs)

