# -*- coding: utf-8 -*-
"""
DSR Environment Configuration
配电网恢复环境配置
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DSRConfig:
    """配电网恢复环境配置"""
    
    # 基础配置
    env_name: str = "dsr"
    system_name: str = "123Bus"  # 支持: 13Bus, 34Bus, 123Bus, 8500-Node
    dss_file: Optional[str] = None  # 自动根据system_name选择
    max_episode_steps: int = 15  # 最大恢复步数
    seed: int = 123456
    
    # 智能体配置
    n_dg: int = 7  # 柴油发电机数量（黑启动电源）
    n_pv: int = 9  # 光伏数量
    n_switch: int = 20  # 开关数量
    n_load_levels: int = 3  # 负荷优先级等级
    
    # 聚合智能体配置
    use_load_aggregation: bool = True  # 是否使用负荷聚合智能体
    n_load_agents: Optional[int] = None  # 负荷智能体数量（None表示自动计算）
    load_aggregation_method: str = "zone"  # 聚合方法: zone(区域), priority(优先级), random(随机)
    
    # 物理约束
    v_min: float = 0.95  # 最小电压标幺值
    v_max: float = 1.05  # 最大电压标幺值
    max_load_per_step: float = 500.0  # 每步最大恢复负荷(kW)
    
    # 奖励权重
    reward_restore: float = 20.0  # 负荷恢复奖励权重
    reward_voltage: float = 1.0  # 电压越限惩罚权重
    reward_overload: float = 1.0  # 线路过载惩罚权重
    reward_done: float = -5.0  # 失败惩罚
    
    # 恢复完成判断
    restoration_threshold: float = 0.95  # 恢复完成阈值（负荷恢复率）
    
    # 故障配置
    fault_scenarios: int = 5  # 故障场景数量
    min_faults: int = 3  # 最小故障数量
    max_faults: int = 5  # 最大故障数量
    
    # 高级特性
    use_action_mask: bool = True  # 是否使用动作掩码
    use_dynamic_network: bool = True  # 是否使用动态智能体网络
    use_render: bool = False  # 是否使用渲染
    record_node: bool = True  # 是否记录节点违规信息
    
    # PowerZoo集成配置
    worker_idx: Optional[int] = None  # 并行工作进程索引
    load_noise: bool = False  # 是否使用负荷噪声
    scale: float = 1.0  # 负荷缩放因子

    # 过载检测相关配置
    overload_threshold: float = 1.0  # 过载阈值倍数
    use_emergency_rating: bool = False  # 是否使用紧急额定值
    log_overload_details: bool = False  # 是否记录过载详情
    
    # 动作空间配置
    pv_power_levels: int = 11  # PV功率等级数量（0-10）
    load_action_levels: int = 2  # 负荷动作等级数量（断开/恢复）
    pv_max_power: float = 150.0  # PV最大功率（kW）
    
    # 观测空间配置
    obs_reserved_dim: int = 10  # 智能体特定观测预留维度
    default_voltage: float = 1.0  # 默认电压标幺值
    
    # 设备重置配置
    line_disconnect_prob: float = 0.7  # 线路断开概率（重置时）
    max_faultable_lines: int = 50  # 最大可故障线路数量
    
    # 负荷优先级配置
    priority_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 3.0,  # 重要负荷权重
        2: 2.0,  # 一般负荷权重
        3: 1.0   # 普通负荷权重
    })
    max_priority_level: int = 3  # 最大优先级等级
    
    # 日志记录配置
    log_interval_episodes: int = 10  # 日志记录间隔（episode数）
    success_threshold: float = 0.9  # 成功恢复阈值（90%恢复率）
    recent_episodes_window: int = 10  # 最近episode窗口大小
    
    # IEEE 123节点系统配置
    ieee123_load_count: int = 85  # IEEE 123节点系统的典型负荷数量
    
    def to_powerzoo_config(self) -> Dict[str, Any]:
        """转换为PowerZoo格式的配置"""
        # 根据系统名称自动选择DSS文件
        dss_files = {
            '13Bus': 'IEEE13Nodeckt_daily.dss',
            '34Bus': 'ieee34Mod1_daily.dss', 
            '123Bus': 'IEEE123Master_daily.dss',
            '8500-Node': 'Master_daily.dss'
        }
        
        # 根据系统名称设置源母线
        source_buses = {
            '13Bus': 'sourcebus',
            '34Bus': 'sourcebus',
            '123Bus': '150',
            '8500-Node': 'e192860'
        }
        
        # 根据系统名称设置节点大小和偏移（用于可视化）
        node_configs = {
            '13Bus': {'node_size': 500, 'shift': 10},
            '34Bus': {'node_size': 500, 'shift': 80},
            '123Bus': {'node_size': 400, 'shift': 80},
            '8500-Node': {'node_size': 10, 'shift': 0}
        }
        
        # 使用指定的dss_file，如果没有则根据system_name选择
        dss_file = self.dss_file or dss_files.get(self.system_name, 'IEEE123Master_daily.dss')
        
        return {
            'env_name': self.env_name,
            'system_name': self.system_name,
            'dss_file': dss_file,
            'max_episode_steps': self.max_episode_steps,
            'seed': self.seed,
            'worker_idx': self.worker_idx,
            'load_noise': self.load_noise,
            'scale': self.scale,
            'use_render': self.use_render,
            'record_node': self.record_node,
            'source_bus': source_buses.get(self.system_name, 'sourcebus'),
            'node_size': node_configs.get(self.system_name, {}).get('node_size', 200),
            'shift': node_configs.get(self.system_name, {}).get('shift', 50),
            'show_node_labels': self.system_name != '8500-Node',  # 8500节点太多不显示标签
            'reg_act_num': 33,  # 调压器动作数量
            'bat_act_num': 33,  # 电池动作数量
            'power_w': 1.0,  # 功率损耗权重
            'cap_w': 0.1,   # 电容器切换权重
            'reg_w': 0.1,   # 调压器调节权重
            'soc_w': 0.1,   # SOC权重
            'dis_w': 0.1,   # 放电权重
        }
    
    def get_agent_config(self) -> Dict[str, Any]:
        """获取智能体配置"""
        # 根据系统规模获取实际负荷数量
        load_counts = {
            '13Bus': 15,      # 13节点系统约15个负荷
            '34Bus': 25,      # 34节点系统约25个负荷
            '123Bus': 85,     # 123节点系统约85个负荷
            '8500-Node': 1177 # 8500节点系统约1177个负荷
        }
        
        actual_load_count = load_counts.get(self.system_name, 85)
        
        # 确定负荷智能体数量
        if self.n_load_agents is not None:
            # 用户指定了负荷智能体数量
            n_load_agents = min(self.n_load_agents, actual_load_count)
        elif self.use_load_aggregation:
            # 自动计算合理的负荷智能体数量
            if actual_load_count <= 20:
                n_load_agents = actual_load_count  # 小系统不需要聚合
            elif actual_load_count <= 50:
                n_load_agents = 15  # 中等系统
            elif actual_load_count <= 200:
                n_load_agents = 30  # 大系统
            else:
                n_load_agents = 50  # 超大系统
        else:
            # 不使用聚合，每个负荷一个智能体
            n_load_agents = actual_load_count
        
        # 根据系统规模调整PV和开关数量
        if self.system_name == '13Bus':
            n_pv = min(self.n_pv, 3)  # 小系统限制PV数量
            n_switch = min(self.n_switch, 10)
        elif self.system_name == '34Bus':
            n_pv = min(self.n_pv, 5)
            n_switch = min(self.n_switch, 15)
        elif self.system_name == '8500-Node':
            n_pv = min(self.n_pv, 20)  # 大系统可以有更多PV
            n_switch = min(self.n_switch, 50)
        else:  # 123Bus
            n_pv = self.n_pv
            n_switch = self.n_switch
        
        # 计算聚合比例
        aggregation_ratio = actual_load_count / max(n_load_agents, 1)
        
        total_agents = 1 + n_pv + n_load_agents
        
        return {
            'n_switch_agents': 1,
            'n_pv_agents': n_pv,
            'n_load_agents': n_load_agents,
            'total_agents': total_agents,
            'n_switches': n_switch,
            'actual_load_count': actual_load_count,
            'aggregation_ratio': aggregation_ratio,
            'use_aggregation': self.use_load_aggregation and aggregation_ratio > 1,
        }
    
    def get_reward_config(self) -> Dict[str, float]:
        """获取奖励配置"""
        return {
            'reward_restore': self.reward_restore,
            'reward_voltage': self.reward_voltage,
            'reward_overload': self.reward_overload,
            'reward_done': self.reward_done,
        }
    
    def get_fault_config(self) -> Dict[str, Any]:
        """获取故障配置"""
        return {
            'fault_scenarios': self.fault_scenarios,
            'min_faults': self.min_faults,
            'max_faults': self.max_faults,
        }
    
    def validate(self) -> bool:
        """验证配置合理性"""
        assert self.max_episode_steps > 0, "max_episode_steps must be positive"
        assert 0 < self.v_min < self.v_max < 2.0, "Invalid voltage bounds"
        assert self.n_dg > 0, "Need at least one DG for black start"
        assert self.n_pv >= 0 and self.n_switch >= 0, "Invalid device numbers"
        assert self.min_faults <= self.max_faults, "Invalid fault range"
        return True


# 预定义配置
DEFAULT_DSR_CONFIG = DSRConfig()

# 13节点系统配置
DSR_13BUS_CONFIG = DSRConfig(
    system_name='13Bus',
    max_episode_steps=10,
    n_dg=3,
    n_pv=3,
    n_switch=10,
    min_faults=1,
    max_faults=3,
)

# 34节点系统配置
DSR_34BUS_CONFIG = DSRConfig(
    system_name='34Bus',
    max_episode_steps=12,
    n_dg=5,
    n_pv=5,
    n_switch=15,
    min_faults=2,
    max_faults=4,
)

# 123节点系统配置（默认）
DSR_123BUS_CONFIG = DSRConfig(
    system_name='123Bus',
    max_episode_steps=15,
    n_dg=7,
    n_pv=9,
    n_switch=20,
    min_faults=3,
    max_faults=5,
)

# 8500节点系统配置
DSR_8500NODE_CONFIG = DSRConfig(
    system_name='8500-Node',
    max_episode_steps=20,
    n_dg=15,
    n_pv=20,
    n_switch=50,
    min_faults=5,
    max_faults=10,
)

# 快速测试配置
FAST_DSR_CONFIG = DSRConfig(
    max_episode_steps=5,
    n_pv=3,
    n_switch=5,
    fault_scenarios=2,
)

    
