# -*- coding: utf-8 -*-
"""
District Dispatch Constants
区域调度环境常量定义
"""


class VOLTAGE:
	"""电压相关常量"""
	MIN_PU = 0.95			# 最小电压标幺值
	MAX_PU = 1.05			# 最大电压标幺值
	TARGET_PU = 1.0			# 目标电压标幺值
	DEADBAND = 0.02			# 电压死区 (±2%)
	EMERGENCY_MIN = 0.90	# 紧急最低电压
	EMERGENCY_MAX = 1.10	# 紧急最高电压


class STORAGE:
	"""储能相关常量"""
	SOC_MIN = 0.1			# 最低SOC
	SOC_MAX = 0.9			# 最高SOC
	SOC_HEALTHY_MIN = 0.2	# 健康SOC下限
	SOC_HEALTHY_MAX = 0.8	# 健康SOC上限
	SOC_INIT = 0.5			# 初始SOC
	CHARGE_EFFICIENCY = 0.95	# 充电效率
	DISCHARGE_EFFICIENCY = 0.95	# 放电效率


class MARKET:
	"""市场/经济相关常量"""
	BASE_PRICE_YUAN_KWH = 0.5	# 基准电价 (元/kWh)
	CARBON_INTENSITY = 0.6		# 电网碳排放强度 (kgCO2/kWh)
	CARBON_PRICE = 0.05			# 碳价 (元/kgCO2)

	# 分时电价倍率 (24小时)
	TOU_MULTIPLIERS = [
		0.5, 0.5, 0.5, 0.5, 0.5, 0.5,		# 00:00-06:00 谷时
		0.8, 1.0, 1.5, 1.5, 1.2, 1.0,		# 06:00-12:00 平/峰
		1.0, 1.0, 1.2, 1.5, 1.5, 1.5,		# 12:00-18:00 平/峰
		1.2, 1.0, 0.8, 0.5, 0.5, 0.5,		# 18:00-24:00 平/谷
	]


class EPISODE:
	"""回合相关常量"""
	DEFAULT_MAX_STEPS = 96		# 15min分辨率, 24小时
	STEP_INTERVAL_MIN = 15		# 步长(分钟)
	STEPS_PER_HOUR = 4			# 每小时步数


class EXCHANGE:
	"""功率交换相关常量"""
	DEFAULT_CAPACITY_KW = 500.0		# 默认联络线容量(kW)
	MAX_EXCHANGE_RATIO = 1.0		# 最大交换比率
	POWER_FACTOR = 0.95				# 默认功率因数
