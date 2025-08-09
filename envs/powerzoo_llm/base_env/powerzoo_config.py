"""
PowerZoo环境配置系统

针对PowerZoo电力系统环境的特定配置、约束和枚举类型
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 延迟导入以避免循环依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OpenDSSScenario(Enum):
    """OpenDSS场景类型"""
    NORMAL = "normal"  # 正常运行
    OPTIMIZATION = "optimization"  # 系统优化
    EMERGENCY = "emergency"  # 紧急响应
    MAINTENANCE = "maintenance"  # 维护操作
    RENEWABLE = "renewable"  # 可再生能源集成
    DEMAND_RESPONSE = "demand_response"  # 需求响应
    VOLTAGE_CONTROL = "voltage_control"  # 电压控制
    LOSS_MINIMIZATION = "loss_minimization"  # 网损最小化
    LOW_LOAD = "low_load"  # 低负荷场景
    PEAK_LOAD = "peak_load"  # 峰值负荷场景
    FAULT = "fault"  # 故障场景


@dataclass
class OpenDSSConstraints:
    """OpenDSS环境约束"""
    # 电压约束
    min_voltage_pu: float = 0.95  # 最小电压标幺值
    max_voltage_pu: float = 1.05  # 最大电压标幺值
    voltage_deadband: float = 0.01  # 电压死区
    
    # 线路约束
    max_line_loading: float = 0.9  # 最大线路负载率
    line_loading_threshold: float = 0.8  # 线路负载阈值
    thermal_limit_margin: float = 0.1  # 热稳定裕度
    
    # 功率约束
    max_power_imbalance: float = 0.05  # 最大功率不平衡率
    max_losses: float = 100.0  # 最大损耗(kW)
    power_factor_min: float = 0.85  # 最小功率因数
    
    # 控制约束
    max_cap_switches_per_step: int = 2  # 单步最大电容器切换数
    max_reg_changes_per_step: int = 1  # 单步最大调压器调整数
    max_battery_power_change: float = 0.2  # 单步最大电池功率变化率
    
    # 时间约束
    capacitor_cooldown: int = 5  # 电容器冷却时间（步）
    regulator_delay: int = 2  # 调压器延迟（步）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.__dict__


@dataclass
class OpenDSSMetrics:
    """OpenDSS性能指标"""
    # 电能质量指标
    voltage_deviation: float = 0.0  # 电压偏差
    voltage_unbalance: float = 0.0  # 电压不平衡度
    thd_voltage: float = 0.0  # 电压总谐波失真
    
    # 系统效率指标
    power_loss: float = 0.0  # 功率损耗
    loss_rate: float = 0.0  # 网损率
    power_factor: float = 1.0  # 功率因数
    
    # 可靠性指标
    voltage_violations: int = 0  # 电压越限次数
    line_overloads: int = 0  # 线路过载次数
    equipment_utilization: float = 0.0  # 设备利用率
    
    # 经济指标
    energy_cost: float = 0.0  # 能源成本
    peak_demand: float = 0.0  # 峰值需求
    demand_charge: float = 0.0  # 需求费用


@dataclass
class PowerZooEnvConfig:
    """PowerZoo环境配置"""
    # 基础环境配置
    env_name: str = "13Bus"
    scenario: OpenDSSScenario = OpenDSSScenario.NORMAL
    source_bus: str = "sourcebus"
    max_episode_steps: int = 96  # 24小时，15分钟间隔
    seed: int = 123
    useS: bool = False
    
    # 约束配置
    constraints: OpenDSSConstraints = field(default_factory=OpenDSSConstraints)
    
    # 动作空间配置
    action_space_config: Dict[str, Any] = field(default_factory=lambda: {
        "capacitor_control": True,  # 电容器控制
        "regulator_control": True,  # 调压器控制
        "battery_control": False,  # 电池控制（默认关闭）
        "pv_control": False,  # 光伏控制
        "switch_control": False,  # 开关控制
        "reg_act_num": 33,  # 调压器动作数
        "bat_act_num": 33,  # 电池动作数
    })
    
    # 观测空间配置
    obs_attributes: List[str] = field(default_factory=lambda: [
        "bus_voltages",  # 母线电压
        "line_loadings",  # 线路负载
        "cap_statuses",  # 电容器状态
        "reg_statuses",  # 调压器状态
        "power_loss",  # 功率损耗
        "load_profile",  # 负载曲线
    ])
    
    # 奖励配置
    reward_config: Dict[str, float] = field(default_factory=lambda: {
        "voltage_violation": -10.0,  # 电压违规惩罚
        "line_overload": -20.0,  # 线路过载惩罚
        "power_loss": -1.0,  # 功率损耗惩罚
        "voltage_deviation": -0.5,  # 电压偏差惩罚
        "control_cost": -0.1,  # 控制成本
        "power_factor": 1.0,  # 功率因数奖励
        "efficiency": 2.0,  # 效率奖励
    })
    
    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        dim = 0
        if self.action_space_config["capacitor_control"]:
            dim += 10  # 假设10个电容器
        if self.action_space_config["regulator_control"]:
            dim += 4  # 假设4个调压器
        if self.action_space_config["battery_control"]:
            dim += 5  # 假设5个电池
        return dim
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result


@dataclass
class OpenDSSExpertRules:
    """OpenDSS专家规则"""
    # 电压控制规则
    voltage_rules: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("min_voltage < 0.95", "increase_voltage_support"),
        ("max_voltage > 1.05", "reduce_voltage_support"),
        ("voltage_unbalance > 0.02", "balance_voltage"),
    ])
    
    # 无功补偿规则
    reactive_rules: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("power_factor < 0.9", "add_capacitor"),
        ("power_factor > 0.98", "remove_capacitor"),
        ("reactive_demand_high", "optimize_capacitor_placement"),
    ])
    
    # 需求响应规则
    demand_rules: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("peak_period", "reduce_non_critical_load"),
        ("price_spike", "discharge_battery"),
        ("renewable_surplus", "charge_battery"),
    ])


class OpenDSSStateAnalyzer:
    """OpenDSS状态分析器"""
    
    def __init__(self, constraints: OpenDSSConstraints):
        self.constraints = constraints
    
    def analyze(self, obs) -> Dict[str, Any]:
        """全面分析OpenDSS观测"""
        analysis = {
            "severity": self._get_severity(obs),
            "voltage_violations": self._check_voltage_violations(obs),
            "line_overloads": self._check_line_overloads(obs),
            "power_quality": self._assess_power_quality(obs),
            "efficiency": self._calculate_efficiency(obs),
            "risks": self._identify_risks(obs),
            "opportunities": self._find_opportunities(obs),
            "recommendations": self._get_recommendations(obs)
        }
        return analysis
    
    def _get_severity(self, obs) -> str:
        """评估严重程度"""
        issues = []
        
        # 检查电压
        voltage_issues = self._check_voltage_violations(obs)
        if voltage_issues:
            issues.append(("voltage", len(voltage_issues)))
        
        # 检查线路
        line_issues = self._check_line_overloads(obs)
        if line_issues:
            issues.append(("line", len(line_issues)))
        
        # 评估严重性
        if not issues:
            return "low"
        
        total_issues = sum(count for _, count in issues)
        if total_issues > 10:
            return "critical"
        elif total_issues > 5:
            return "high"
        elif total_issues > 2:
            return "medium"
        else:
            return "low"
    
    def _check_voltage_violations(self, obs) -> List[Dict[str, Any]]:
        """检查电压违规"""
        violations = []
        
        bus_voltages = obs.get('bus_voltages', {})
        for bus, voltages in bus_voltages.items():
            for i, v in enumerate(voltages):
                if v < self.constraints.min_voltage_pu:
                    violations.append({
                        "type": "undervoltage",
                        "bus": bus,
                        "phase": i,
                        "voltage": v,
                        "limit": self.constraints.min_voltage_pu,
                        "severity": self.constraints.min_voltage_pu - v
                    })
                elif v > self.constraints.max_voltage_pu:
                    violations.append({
                        "type": "overvoltage",
                        "bus": bus,
                        "phase": i,
                        "voltage": v,
                        "limit": self.constraints.max_voltage_pu,
                        "severity": v - self.constraints.max_voltage_pu
                    })
        
        return violations
    
    def _check_line_overloads(self, obs) -> List[Dict[str, Any]]:
        """检查线路过载"""
        overloads = []
        
        line_loadings = obs.get('line_loadings', {})
        for line, loading in line_loadings.items():
            if loading > self.constraints.line_loading_threshold:
                overloads.append({
                    "type": "overload",
                    "line": line,
                    "loading": loading,
                    "threshold": self.constraints.line_loading_threshold,
                    "severity": loading - self.constraints.line_loading_threshold
                })
        
        return overloads
    
    def _assess_power_quality(self, obs) -> Dict[str, float]:
        """评估电能质量"""
        quality = {}
        
        # 计算电压偏差
        bus_voltages = obs.get('bus_voltages', {})
        all_voltages = []
        for voltages in bus_voltages.values():
            if isinstance(voltages, (list, tuple)):
                all_voltages.extend(voltages)
            else:
                all_voltages.append(voltages)
        
        if all_voltages:
            quality['avg_voltage'] = np.mean(all_voltages)
            quality['voltage_deviation'] = np.std(all_voltages)
            quality['min_voltage'] = min(all_voltages)
            quality['max_voltage'] = max(all_voltages)
        
        # 功率因数
        quality['power_factor'] = obs.get('power_factor', 1.0)
        
        # 功率损耗
        quality['power_loss'] = obs.get('power_loss', 0.0)
        
        return quality
    
    def _calculate_efficiency(self, obs) -> Dict[str, float]:
        """计算系统效率"""
        efficiency = {}
        
        # 网损率
        power_loss = obs.get('power_loss', 0.0)
        efficiency['loss_rate'] = power_loss
        
        # 设备利用率
        cap_statuses = obs.get('cap_statuses', {})
        if cap_statuses:
            active_caps = sum(1 for status in cap_statuses.values() if status)
            efficiency['capacitor_utilization'] = active_caps / len(cap_statuses)
        
        return efficiency
    
    def _identify_risks(self, obs) -> List[Dict[str, Any]]:
        """识别风险"""
        risks = []
        
        # 电压稳定性风险
        voltage_margin = self._calculate_voltage_margin(obs)
        if voltage_margin < 0.05:
            risks.append({
                "type": "voltage_stability",
                "severity": "high",
                "margin": voltage_margin,
                "description": "电压稳定裕度不足"
            })
        
        # 功率损耗风险
        power_loss = obs.get('power_loss', 0.0)
        if power_loss > self.constraints.max_losses:
            risks.append({
                "type": "high_losses",
                "severity": "medium",
                "current_loss": power_loss,
                "limit": self.constraints.max_losses,
                "description": "系统损耗过高"
            })
        
        return risks
    
    def _find_opportunities(self, obs) -> List[Dict[str, Any]]:
        """发现优化机会"""
        opportunities = []
        
        # 无功优化机会
        power_factor = obs.get('power_factor', 1.0)
        if power_factor < 0.95:
            opportunities.append({
                "type": "reactive_optimization",
                "potential": "high",
                "action": "adjust_capacitors",
                "expected_improvement": (0.95 - power_factor) * 100
            })
        
        # 电压优化机会
        voltage_deviation = self._calculate_voltage_deviation(obs)
        if voltage_deviation > 0.02:
            opportunities.append({
                "type": "voltage_optimization",
                "potential": "medium",
                "action": "adjust_regulators",
                "expected_improvement": voltage_deviation * 100
            })
        
        return opportunities
    
    def _get_recommendations(self, obs) -> List[str]:
        """获取操作建议"""
        recommendations = []
        severity = self._get_severity(obs)
        
        if severity == "critical":
            recommendations.append("立即执行紧急电压控制")
            recommendations.append("检查并调整所有可用设备")
        elif severity == "high":
            recommendations.append("执行预防性控制措施")
            recommendations.append("优化无功补偿配置")
        elif severity == "medium":
            recommendations.append("监控系统状态变化")
            recommendations.append("准备调整方案")
        else:
            recommendations.append("保持当前运行状态")
            recommendations.append("定期检查系统指标")
        
        return recommendations
    
    def _calculate_voltage_margin(self, obs) -> float:
        """计算电压裕度"""
        bus_voltages = obs.get('bus_voltages', {})
        all_voltages = []
        for voltages in bus_voltages.values():
            if isinstance(voltages, (list, tuple)):
                all_voltages.extend(voltages)
            else:
                all_voltages.append(voltages)
        
        if all_voltages:
            min_margin = min(v - self.constraints.min_voltage_pu for v in all_voltages)
            max_margin = min(self.constraints.max_voltage_pu - v for v in all_voltages)
            return min(min_margin, max_margin)
        
        return 0.1  # 默认裕度
    
    def _calculate_voltage_deviation(self, obs) -> float:
        """计算电压偏差"""
        bus_voltages = obs.get('bus_voltages', {})
        all_voltages = []
        for voltages in bus_voltages.values():
            if isinstance(voltages, (list, tuple)):
                all_voltages.extend(voltages)
            else:
                all_voltages.append(voltages)
        
        if all_voltages:
            return np.std(all_voltages)
        
        return 0.0


class PowerZooActionSelector:
    """PowerZoo动作选择器"""
    
    def __init__(self, env, config: 'PowerZooEnvConfig'):
        self.env = env
        self.config = config
        self.action_space = env.action_space
        
    def get_available_actions(self, obs) -> List[str]:
        """获取当前可用动作类型"""
        available = []
        
        # 检查电容器动作
        if self.config.action_space_config["capacitor_control"]:
            if self._can_switch_capacitor(obs):
                available.append("capacitor_control")
        
        # 检查调压器动作
        if self.config.action_space_config["regulator_control"]:
            if self._can_adjust_regulator(obs):
                available.append("regulator_control")
        
        # 检查电池动作
        if self.config.action_space_config["battery_control"]:
            if self._can_control_battery(obs):
                available.append("battery_control")
        
        return available
    
    def select_action(self, obs, model=None) -> Optional[np.ndarray]:
        """选择动作"""
        if model is not None:
            # 使用模型选择动作
            return self._model_based_selection(obs, model)
        return None

    def _can_switch_capacitor(self, obs) -> bool:
        """检查是否可以切换电容器"""
        # 检查冷却时间等约束
        return True  # 简化实现
    
    def _can_adjust_regulator(self, obs) -> bool:
        """检查是否可以调整调压器"""
        return True  # 简化实现
    
    def _can_control_battery(self, obs) -> bool:
        """检查是否可以控制电池"""
        return hasattr(self.env, 'bat_num') and self.env.bat_num > 0
    
    def _model_based_selection(self, obs, model) -> Optional[np.ndarray]:
        """基于模型的动作选择"""
        if not TORCH_AVAILABLE:
            return None
            
        try:
            # 将观测转换为模型输入
            obs_array = self.env.wrap_obs(obs)
            
            # 使用模型预测动作
            with torch.no_grad():
                action, _, _ = model.get_actions(
                    torch.FloatTensor(obs_array).unsqueeze(0),
                    None,
                    None,
                    deterministic=True
                )
            
            return action.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Model-based action selection failed: {e}")
            return None
