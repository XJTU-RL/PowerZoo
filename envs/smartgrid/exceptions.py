# -*- coding: utf-8 -*-
"""
PowerZoo LLM 环境异常类定义

提供专用异常类以提高错误处理精度和可调试性。

@File      : exceptions.py
@Time      : 2025-11-22
@Author    : Xiaodong Zheng (with Claude Code)
"""


class PowerZooError(Exception):
	"""PowerZoo环境基础异常类"""
	pass


class DSSSimulationError(PowerZooError):
	"""DSS仿真相关错误"""
	pass


class DSSConvergenceError(DSSSimulationError):
	"""DSS求解不收敛错误

	当OpenDSS潮流计算不收敛时抛出，通常表示系统配置不当或工况异常。
	"""
	pass


class DSSCompileError(DSSSimulationError):
	"""DSS编译错误

	当DSS文件编译失败时抛出，可能是文件路径错误或DSS脚本语法问题。
	"""
	pass


class ActionValidationError(PowerZooError):
	"""动作验证错误

	当输入动作不符合动作空间要求时抛出。
	"""
	pass


class ConfigurationError(PowerZooError):
	"""配置错误

	当环境配置不合法时抛出。
	"""
	pass


class RewardCalculationError(PowerZooError):
	"""奖励计算错误

	当奖励函数计算过程中出现异常时抛出。
	"""
	pass


class DataProcessingError(PowerZooError):
	"""数据处理错误

	当负载曲线或PV数据处理失败时抛出。
	"""
	pass
