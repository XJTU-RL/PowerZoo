# -*- coding: utf-8 -*-
"""
单智能体PowerZoo环境配置管理

本模块定义了单智能体环境的配置参数，包括:
- 环境基础配置
- 奖励函数参数
- 动作空间配置
- 观测空间配置
- 训练相关参数
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class SingleAgentConfig:
    """单智能体PowerZoo环境配置类"""
    
    # 基础环境配置
    circuit_name: str = "13Bus"  # 电路名称
    max_episode_steps: int = 24  # 最大步数
    seed: int = 42  # 随机种子
    
    # 奖励函数配置
    voltage_penalty_weight: float = 1.0  # 电压偏差惩罚权重
    power_loss_weight: float = 0.1  # 功率损耗权重
    discharge_penalty_weight: float = 0.5  # 放电惩罚权重
    voltage_target: float = 1.0  # 目标电压(标幺值)
    voltage_tolerance: float = 0.05  # 电压容差
    
    # 动作空间配置
    enable_capacitors: bool = True  # 启用电容器控制
    enable_regulators: bool = True  # 启用调压器控制
    enable_batteries: bool = True  # 启用电池控制
    enable_pv_systems: bool = True  # 启用光伏系统控制
    
    # 观测空间配置
    include_bus_voltages: bool = True  # 包含母线电压
    include_power_flows: bool = True  # 包含功率流
    include_device_states: bool = True  # 包含设备状态
    normalize_observations: bool = True  # 标准化观测
    
    # 训练配置
    log_level: str = "INFO"  # 日志级别
    save_episode_data: bool = False  # 保存回合数据
    render_mode: Optional[str] = None  # 渲染模式
    
    # 高级配置
    custom_reward_weights: Optional[Dict[str, float]] = None  # 自定义奖励权重
    action_constraints: Optional[Dict[str, Any]] = None  # 动作约束
    observation_filters: Optional[List[str]] = None  # 观测过滤器
    
    def __post_init__(self):
        """初始化后处理"""
        # 设置默认的自定义奖励权重
        if self.custom_reward_weights is None:
            self.custom_reward_weights = {
                'voltage_penalty': self.voltage_penalty_weight,
                'power_loss': self.power_loss_weight,
                'discharge_penalty': self.discharge_penalty_weight
            }
        
        # 设置默认的动作约束
        if self.action_constraints is None:
            self.action_constraints = {
                'capacitor_steps': [-1, 0, 1],  # 电容器步数
                'regulator_taps': list(range(-16, 17)),  # 调压器抽头
                'battery_power_range': (-1.0, 1.0),  # 电池功率范围
                'pv_power_factor_range': (0.8, 1.0)  # 光伏功率因数范围
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'circuit_name': self.circuit_name,
            'max_episode_steps': self.max_episode_steps,
            'seed': self.seed,
            'voltage_penalty_weight': self.voltage_penalty_weight,
            'power_loss_weight': self.power_loss_weight,
            'discharge_penalty_weight': self.discharge_penalty_weight,
            'voltage_target': self.voltage_target,
            'voltage_tolerance': self.voltage_tolerance,
            'enable_capacitors': self.enable_capacitors,
            'enable_regulators': self.enable_regulators,
            'enable_batteries': self.enable_batteries,
            'enable_pv_systems': self.enable_pv_systems,
            'include_bus_voltages': self.include_bus_voltages,
            'include_power_flows': self.include_power_flows,
            'include_device_states': self.include_device_states,
            'normalize_observations': self.normalize_observations,
            'log_level': self.log_level,
            'save_episode_data': self.save_episode_data,
            'render_mode': self.render_mode,
            'custom_reward_weights': self.custom_reward_weights,
            'action_constraints': self.action_constraints,
            'observation_filters': self.observation_filters
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SingleAgentConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'SingleAgentConfig':
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self
    
    def validate(self) -> bool:
        """验证配置参数的有效性"""
        # 检查基础参数
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        
        if not 0 < self.voltage_tolerance < 1:
            raise ValueError("voltage_tolerance must be between 0 and 1")
        
        if self.voltage_target <= 0:
            raise ValueError("voltage_target must be positive")
        
        # 检查权重参数
        weights = [self.voltage_penalty_weight, self.power_loss_weight, self.discharge_penalty_weight]
        if any(w < 0 for w in weights):
            raise ValueError("All weight parameters must be non-negative")
        
        # 检查至少启用一种控制设备
        if not any([self.enable_capacitors, self.enable_regulators, 
                   self.enable_batteries, self.enable_pv_systems]):
            raise ValueError("At least one control device type must be enabled")
        
        return True
    
    def get_folder_path(self) -> str:
        """获取DSS文件夹路径"""
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        return os.path.join(project_root, "node_systems")
    
    def to_env_info(self) -> Dict[str, Any]:
        """转换为环境info字典"""
        # 根据circuit_name映射到对应的系统配置
        circuit_mapping = {
            "13Bus": {
                'system_name': '13Bus',
                'dss_file': 'IEEE13Nodeckt_daily.dss',
                'reg_act_num': 4,
                'bat_act_num': 3,
                'pv_control': True
            },
            "34Bus": {
                'system_name': '34Bus', 
                'dss_file': 'ieee34Mod1_daily.dss',
                'reg_act_num': 33,
                'bat_act_num': 5,
                'pv_control': True
            },
            "123Bus": {
                'system_name': '123Bus',
                'dss_file': 'IEEE123Master_daily.dss', 
                'reg_act_num': 4,
                'bat_act_num': 8,
                'pv_control': True
            }
        }
        
        # 获取电路配置
        circuit_config = circuit_mapping.get(self.circuit_name, circuit_mapping["13Bus"])
        
        return {
            'system_name': circuit_config['system_name'],
            'dss_file': circuit_config['dss_file'],
            'max_episode_steps': self.max_episode_steps,
            'reg_act_num': circuit_config['reg_act_num'],
            'bat_act_num': circuit_config['bat_act_num'],
            'pv_control': circuit_config['pv_control'] and self.enable_pv_systems,
            'worker_idx': 0,
            'seed': self.seed,
            # 添加自定义配置
            'voltage_penalty_weight': self.voltage_penalty_weight,
            'power_loss_weight': self.power_loss_weight,
            'discharge_penalty_weight': self.discharge_penalty_weight,
            'enable_capacitors': self.enable_capacitors,
            'enable_regulators': self.enable_regulators,
            'enable_batteries': self.enable_batteries,
            'enable_pv_systems': self.enable_pv_systems
        }


# 预定义配置模板
DEFAULT_CONFIG = SingleAgentConfig()

TRAINING_CONFIG = SingleAgentConfig(
    max_episode_steps=24,
    voltage_penalty_weight=2.0,
    power_loss_weight=0.2,
    discharge_penalty_weight=1.0,
    save_episode_data=True,
    log_level="DEBUG"
)

TESTING_CONFIG = SingleAgentConfig(
    max_episode_steps=24,
    voltage_penalty_weight=1.0,
    power_loss_weight=0.1,
    discharge_penalty_weight=0.5,
    save_episode_data=False,
    log_level="INFO"
)

FAST_CONFIG = SingleAgentConfig(
    max_episode_steps=12,
    voltage_penalty_weight=1.0,
    power_loss_weight=0.1,
    discharge_penalty_weight=0.5,
    normalize_observations=False,
    save_episode_data=False
)