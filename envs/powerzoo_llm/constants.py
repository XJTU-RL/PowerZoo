# -*- coding: utf-8 -*-
"""
PowerZoo LLM 环境常量定义

集中管理所有魔法数字和配置常量，提高代码可维护性。

@File      : constants.py
@Time      : 2025-11-22
@Author    : Xiaodong Zheng (with Claude Code)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VoltageConstants:
	"""电压相关常量 (per unit)"""
	MIN_PU: float = 0.95          # 最小电压限制
	MAX_PU: float = 1.05          # 最大电压限制
	TARGET_PU: float = 1.0        # 目标电压
	DEADBAND: float = 0.02        # 电压死区 (±2%)
	WARNING_LOW: float = 0.96     # 低电压警告阈值
	WARNING_HIGH: float = 1.04    # 高电压警告阈值


@dataclass(frozen=True)
class PowerFactorConstants:
	"""功率因数相关常量"""
	MIN: float = 0.8              # 最小功率因数
	MAX: float = 1.0              # 最大功率因数
	DEFAULT: float = 0.95         # 默认功率因数
	LAGGING_THRESHOLD: float = 0.95   # 感性运行阈值
	LEADING_THRESHOLD: float = 0.95   # 容性运行阈值


@dataclass(frozen=True)
class SimulationConstants:
	"""仿真相关常量"""
	MAX_ITERATIONS: int = 50       # 最大迭代次数
	MAX_CONTROL_ITER: int = 100    # 最大控制迭代次数
	PERFORMANCE_THRESHOLD_MS: float = 100.0  # 性能警告阈值(毫秒)
	CONVERGENCE_TOLERANCE: float = 1e-6      # 收敛容差


@dataclass(frozen=True)
class RewardConstants:
	"""奖励函数相关常量"""
	POWER_LOSS_BASE: float = 0.02     # 网损基准值 (2%)
	VOLTAGE_THRESHOLD: float = 0.02    # 电压约束死区
	ACTION_SMOOTHING_WEIGHT: float = 0.3  # 动作平滑权重
	POWER_LOSS_WEIGHT: float = 1.0     # 网损权重
	CONTROL_WEIGHT: float = 0.5        # 控制权重
	PV_WEIGHT: float = 0.8             # PV权重


@dataclass(frozen=True)
class DeviceConstants:
	"""设备相关常量"""
	REGULATOR_DEFAULT_TAP: float = 1.0    # 调压器默认抽头
	REGULATOR_MIN_TAP: float = 0.9        # 调压器最小抽头
	REGULATOR_MAX_TAP: float = 1.1        # 调压器最大抽头
	REGULATOR_NUM_TAPS: int = 32          # 调压器抽头数量
	BATTERY_DURATION_HOURS: float = 1.0   # 电池持续时间(小时)


# 创建常量实例供全局使用
VOLTAGE = VoltageConstants()
POWER_FACTOR = PowerFactorConstants()
SIMULATION = SimulationConstants()
REWARD = RewardConstants()
DEVICE = DeviceConstants()


# 便捷访问
VOLTAGE_MIN = VOLTAGE.MIN_PU
VOLTAGE_MAX = VOLTAGE.MAX_PU
VOLTAGE_TARGET = VOLTAGE.TARGET_PU
