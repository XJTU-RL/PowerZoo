# -*- coding: utf-8 -*-
"""
DSR Environment Configuration
配电网恢复环境配置
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MonitoringConfig:
    visualization: bool = False
    visualization_interval: int = 50

    def __post_init__(self):
        if self.visualization_interval < 1:
            raise ValueError("监控间隔必须为正")


@dataclass
class DSRConfig:
    """配电网恢复环境配置"""

    # 基础配置
    env_name: str = "dsr"
    state_type: str = "EP"
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    system_name: str = "123Bus"  # 支持: 13Bus, 34Bus, 123Bus, 8500-Node
    dss_folder: Optional[str] = None
    dss_file: Optional[str] = None  # 自动根据system_name选择
    max_episode_steps: int = 15  # 最大恢复步数
    seed: int = 123456

    # 智能体配置
    n_dg: int = 7  # 柴油发电机数量（黑启动电源）
    n_pv: int = 9  # 光伏数量
    n_switch: int = 20  # 开关数量

    # 聚合智能体配置
    use_load_aggregation: bool = True  # 是否使用负荷聚合智能体
    n_load_agents: Optional[int] = None  # 负荷智能体数量（None表示自动计算）
    load_aggregation_method: str = "zone"  # 聚合方法: zone(区域), priority(优先级), random(随机)

    # 物理约束
    v_min: float = 0.95  # 最小电压标幺值
    v_max: float = 1.05  # 最大电压标幺值

    # 奖励权重
    reward_restore: float = 20.0  # 负荷恢复奖励权重
    reward_voltage: float = 1.0  # 电压越限惩罚权重
    reward_overload: float = 1.0  # 线路过载惩罚权重
    reward_done: float = -5.0  # 失败惩罚

    # 恢复完成判断
    restoration_threshold: float = 0.95  # 恢复完成阈值（负荷恢复率）
    success_threshold: float = 0.9  # 成功恢复阈值（用于日志记录）

    # 故障配置
    min_faults: int = 3  # 最小故障数量
    max_faults: int = 5  # 最大故障数量

    # PowerZoo集成配置
    worker_idx: Optional[int] = None  # 并行工作进程索引

    # 动作空间配置
    pv_power_levels: int = 11  # PV功率等级数量（0-10）
    load_action_levels: int = 2  # 负荷动作等级数量（断开/恢复）
    pv_max_power: float = 150.0  # PV最大功率（kW）

    # 观测空间配置
    obs_reserved_dim: int = 10  # 智能体特定观测预留维度
    default_voltage: float = 1.0  # 默认电压标幺值
    debug_mode: bool = False  # 调试模式，控制详细日志输出

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
    recent_episodes_window: int = 10  # 最近episode窗口大小

    @classmethod
    def from_env_args(cls, env_args):
        """旧入口统一转换；正式训练在 ConfigLoader 处严格检查。"""
        from utils.config_validation import legacy_env_config
        return legacy_env_config(cls, env_args)

    def to_vvc_config(self):
        # 核心只消费这三个值；不再输出没有消费者的显示/奖励/负荷开关。
        dss_files = {
            "13Bus": "IEEE13Nodeckt.dss",
            "34Bus": "ieee34Mod1.dss",
            "123Bus": "IEEE123Master.dss",
            "8500-Node": "Master.dss",
        }
        return {
            "dss_file": self.dss_file
            or dss_files.get(self.system_name, "IEEE123Master.dss"),
            "reg_act_num": 33,
            "bat_act_num": 33,
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

    def validate(self) -> bool:
        """验证配置合理性"""
        assert self.max_episode_steps > 0, "max_episode_steps must be positive"
        assert 0 < self.v_min < self.v_max < 2.0, "Invalid voltage bounds"
        assert self.n_dg > 0, "Need at least one DG for black start"
        assert self.n_pv >= 0 and self.n_switch >= 0, "Invalid device numbers"
        assert self.min_faults <= self.max_faults, "Invalid fault range"
        assert self.load_aggregation_method in (
            "zone",
            "priority",
            "random",
        ), "Unknown load aggregation method"
        return True
