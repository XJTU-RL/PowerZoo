# -*- coding: utf-8 -*-
"""
SmartGrid 环境统一配置系统

设计原则：
1. 单一数据源 - 所有配置通过 SmartGridConfig 传递
2. 类型安全 - 使用 dataclass 和类型注解
3. 验证完整 - 在 __post_init__ 中验证所有参数
4. 清晰层次 - DeviceConfig → SmartGridConfig → 环境/电路
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import math


@dataclass
class DeviceConfig:
	"""设备控制配置

	统一描述电容器、调压器、电池、光伏的控制参数
	"""
	action_num: Union[int, float] = 33  # 动作数量，float('inf') 表示连续控制
	control_enabled: bool = True        # 是否启用控制

	@property
	def is_continuous(self) -> bool:
		"""是否为连续控制模式"""
		return self.action_num == float('inf') or self.action_num == math.inf

	def validate(self, device_type: str) -> None:
		"""验证配置有效性"""
		if self.control_enabled:
			if not self.is_continuous and self.action_num < 2:
				raise ValueError(f"{device_type} action_num 必须 >= 2 或为 inf，当前值: {self.action_num}")


@dataclass
class RewardWeights:
	"""奖励权重配置"""
	power_loss: float = 1.0      # 功率损耗权重
	capacitor: float = 0.0303    # 电容器切换权重
	regulator: float = 0.0303    # 调压器调节权重
	battery_soc: float = 0.0     # 电池SOC权重
	battery_discharge: float = 0.303  # 电池放电权重
	pv_control: float = 0.0606   # 光伏控制权重

	def to_dict(self) -> Dict[str, float]:
		"""转换为旧格式字典（兼容性）"""
		return {
			'power_w': self.power_loss,
			'cap_w': self.capacitor,
			'reg_w': self.regulator,
			'soc_w': self.battery_soc,
			'dis_w': self.battery_discharge,
			'pv_w': self.pv_control,
		}


@dataclass
class SmartGridConfig:
	"""SmartGrid 环境统一配置

	这是整个参数传递链的唯一数据源。
	所有组件（Env, Circuits, 节点）都从这个配置获取参数。

	使用示例:
		config = SmartGridConfig.from_env_name('34Bus_pv')
		env = Env(config)
	"""
	# === 基础环境配置 ===
	env_name: str = '34Bus_pv'
	system_name: str = '34Bus_PV'
	dss_file: str = 'ieee34Mod1_duty.dss'
	max_episode_steps: int = 360
	seed: int = 123456

	# === 设备控制配置 ===
	# 使用 default_factory 避免可变默认值问题
	capacitor: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=2))
	regulator: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=33))
	battery: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=33))
	pv: DeviceConfig = field(default_factory=lambda: DeviceConfig(action_num=float('inf'), control_enabled=True))

	# === 奖励配置 ===
	reward_weights: RewardWeights = field(default_factory=RewardWeights)

	# === 约束配置 ===
	voltage_min: float = 0.95
	voltage_max: float = 1.05

	# === 运行时配置 ===
	for_LLM: bool = False
	use_cmdp: bool = True
	dss_act: bool = False  # 是否使用 OpenDSS 自动控制

	# === 显示配置 ===
	source_bus: str = 'sourcebus'
	node_size: int = 300
	shift: int = 50
	show_node_labels: bool = False
	scale: float = 1.0

	# === Worker 配置 ===
	worker_idx: Optional[int] = None

	# === PV 方案 ===
	pv_plan: Optional[str] = None  # PV方案: none/conservative/optimized/aggressive

	def __post_init__(self):
		"""验证并处理配置"""
		# 验证设备配置
		self.capacitor.validate('Capacitor')
		self.regulator.validate('Regulator')
		self.battery.validate('Battery')
		if self.pv.control_enabled:
			self.pv.validate('PV')

		# 验证时间步
		if self.max_episode_steps < 1:
			raise ValueError(f"max_episode_steps 必须 >= 1，当前值: {self.max_episode_steps}")

		# 验证电压约束
		if self.voltage_min >= self.voltage_max:
			raise ValueError(f"voltage_min({self.voltage_min}) 必须小于 voltage_max({self.voltage_max})")

	# === 便捷属性（兼容旧代码）===
	@property
	def reg_act_num(self) -> int:
		"""调压器动作数量"""
		return int(self.regulator.action_num) if not self.regulator.is_continuous else 33

	@property
	def bat_act_num(self) -> Union[int, float]:
		"""电池动作数量"""
		return self.battery.action_num

	@property
	def pv_act_num(self) -> Union[int, float]:
		"""光伏动作数量"""
		return self.pv.action_num

	@property
	def pv_control_enabled(self) -> bool:
		"""是否启用光伏控制"""
		return self.pv.control_enabled

	@property
	def RBP_act_num(self) -> tuple:
		"""调压器、电池、光伏动作数量元组（用于 Circuits）"""
		return (self.reg_act_num, self.bat_act_num, self.pv_act_num)

	def to_info_dict(self) -> Dict[str, Any]:
		"""转换为旧格式 info 字典（向后兼容）

		这个方法用于兼容旧代码，新代码应直接使用 SmartGridConfig
		"""
		info = {
			# 基础配置
			'env_name': self.env_name,
			'system_name': self.system_name,
			'dss_file': self.dss_file,
			'max_episode_steps': self.max_episode_steps,

			# 设备动作配置
			'reg_act_num': self.reg_act_num,
			'bat_act_num': self.bat_act_num,
			'pv_act_num': self.pv_act_num,
			'pv_control': self.pv_control_enabled,

			# 奖励权重
			**self.reward_weights.to_dict(),

			# 约束
			'voltage_min': self.voltage_min,
			'voltage_max': self.voltage_max,

			# 运行时
			'for_LLM': self.for_LLM,
			'use_cmdp': self.use_cmdp,

			# 显示
			'source_bus': self.source_bus,
			'node_size': self.node_size,
			'shift': self.shift,
			'show_node_labels': self.show_node_labels,
			'scale': self.scale,

			# Worker
			'worker_idx': self.worker_idx,

			# PV 方案
			'pv_plan': self.pv_plan,
		}
		return info

	@classmethod
	def from_dict(cls, config_dict: Dict[str, Any]) -> 'SmartGridConfig':
		"""从字典创建配置（用于从旧配置迁移）"""
		# 提取设备配置
		capacitor = DeviceConfig(action_num=2)
		regulator = DeviceConfig(action_num=config_dict.get('reg_act_num', 33))
		battery = DeviceConfig(action_num=config_dict.get('bat_act_num', 33))
		pv = DeviceConfig(
			action_num=config_dict.get('pv_act_num', float('inf')),
			control_enabled=config_dict.get('pv_control', False)
		)

		# 提取奖励权重
		reward_weights = RewardWeights(
			power_loss=config_dict.get('power_w', 1.0),
			capacitor=config_dict.get('cap_w', 0.0303),
			regulator=config_dict.get('reg_w', 0.0303),
			battery_soc=config_dict.get('soc_w', 0.0),
			battery_discharge=config_dict.get('dis_w', 0.303),
			pv_control=config_dict.get('pv_w', 0.0606),
		)

		return cls(
			env_name=config_dict.get('env_name', '34Bus_pv'),
			system_name=config_dict.get('system_name', '34Bus_PV'),
			dss_file=config_dict.get('dss_file', 'ieee34Mod1_duty.dss'),
			max_episode_steps=config_dict.get('max_episode_steps', 360),
			seed=config_dict.get('seed', 123456),
			capacitor=capacitor,
			regulator=regulator,
			battery=battery,
			pv=pv,
			reward_weights=reward_weights,
			voltage_min=config_dict.get('voltage_min', 0.95),
			voltage_max=config_dict.get('voltage_max', 1.05),
			for_LLM=config_dict.get('for_LLM', False),
			use_cmdp=config_dict.get('use_cmdp', True),
			dss_act=config_dict.get('dss_act', False),
			source_bus=config_dict.get('source_bus', 'sourcebus'),
			node_size=config_dict.get('node_size', 300),
			shift=config_dict.get('shift', 50),
			show_node_labels=config_dict.get('show_node_labels', False),
			scale=config_dict.get('scale', 1.0),
			worker_idx=config_dict.get('worker_idx'),
		)

	@classmethod
	def from_env_args(cls, env_args: dict) -> 'SmartGridConfig':
		"""从 unified_config_loader 的 env_args 构建配置

		替代原有的 ConfigLoader.get_config()。
		env_args 结构:
		  - 直接字段: system_name, dss_file, max_episode_steps, seed, pv_plan, ...
		  - env_specific_config.devices.{regulators,batteries,pv_systems}
		  - env_specific_config.reward_weights.{power_loss,capacitor,...}
		  - env_specific_config.constraints.{voltage_min,voltage_max}
		"""
		config = cls()

		# 直接标量字段映射: env_args key -> SmartGridConfig attr
		direct_fields = {
			'system_name': 'system_name',
			'dss_file': 'dss_file',
			'env_name': 'env_name',
			'max_episode_steps': 'max_episode_steps',
			'episode_length': 'max_episode_steps',
			'seed': 'seed',
			'pv_plan': 'pv_plan',
			'dss_act': 'dss_act',
			'voltage_min': 'voltage_min',
			'voltage_max': 'voltage_max',
			'source_bus': 'source_bus',
		}
		for src_key, dst_attr in direct_fields.items():
			if src_key in env_args and env_args[src_key] is not None:
				setattr(config, dst_attr, env_args[src_key])

		# 设备配置 (从 env_specific_config.devices)
		devices = env_args.get('env_specific_config', {}).get('devices', {})
		if 'regulators' in devices:
			reg_cfg = devices['regulators']
			config.regulator.action_num = reg_cfg.get('action_num', config.regulator.action_num)
		if 'batteries' in devices:
			bat_cfg = devices['batteries']
			if bat_cfg.get('action_space') == 'continuous':
				config.battery.action_num = float('inf')
			elif 'action_num' in bat_cfg:
				config.battery.action_num = bat_cfg['action_num']
		if 'pv_systems' in devices:
			pv_cfg = devices['pv_systems']
			config.pv.control_enabled = pv_cfg.get('control_enabled', config.pv.control_enabled)
			if pv_cfg.get('action_space') == 'continuous':
				config.pv.action_num = float('inf')
			elif 'action_num' in pv_cfg:
				config.pv.action_num = pv_cfg['action_num']

		# 奖励权重 (从 env_specific_config.reward_weights)
		weights = env_args.get('env_specific_config', {}).get('reward_weights', {})
		if weights:
			for attr in ['power_loss', 'capacitor', 'regulator', 'battery_soc', 'battery_discharge', 'pv_control']:
				if attr in weights:
					setattr(config.reward_weights, attr, weights[attr])

		# 约束覆盖 (从 env_specific_config.constraints)
		constraints = env_args.get('env_specific_config', {}).get('constraints', {})
		if constraints:
			config.voltage_min = constraints.get('voltage_min', config.voltage_min)
			config.voltage_max = constraints.get('voltage_max', config.voltage_max)

		# 显示配置
		for display_field in ['node_size', 'shift', 'show_node_labels']:
			if display_field in env_args and env_args[display_field] is not None:
				setattr(config, display_field, env_args[display_field])

		return config

	def with_worker_idx(self, worker_idx: int) -> 'SmartGridConfig':
		"""创建带有 worker_idx 的新配置副本"""
		import copy
		new_config = copy.deepcopy(self)
		new_config.worker_idx = worker_idx
		return new_config

	def __repr__(self) -> str:
		return (
			f"SmartGridConfig(\n"
			f"  env_name='{self.env_name}',\n"
			f"  system_name='{self.system_name}',\n"
			f"  max_episode_steps={self.max_episode_steps},\n"
			f"  devices=(\n"
			f"    regulator: action_num={self.reg_act_num},\n"
			f"    battery: action_num={self.bat_act_num},\n"
			f"    pv: action_num={self.pv_act_num}, enabled={self.pv_control_enabled}\n"
			f"  )\n"
			f")"
		)


# === 预定义配置 ===
# 这些配置可以直接使用，无需从文件加载

PRESET_CONFIGS: Dict[str, SmartGridConfig] = {}


def register_preset(name: str, config: SmartGridConfig) -> None:
	"""注册预定义配置"""
	PRESET_CONFIGS[name] = config


def get_preset(name: str) -> SmartGridConfig:
	"""获取预定义配置"""
	if name not in PRESET_CONFIGS:
		raise ValueError(f"未知的预定义配置: {name}，可用配置: {list(PRESET_CONFIGS.keys())}")
	import copy
	return copy.deepcopy(PRESET_CONFIGS[name])


# 注册常用配置
register_preset('34Bus_pv', SmartGridConfig(
	env_name='34Bus_pv',
	system_name='34Bus_PV',
	dss_file='ieee34Mod1_duty.dss',
	max_episode_steps=360,
	regulator=DeviceConfig(action_num=33),
	battery=DeviceConfig(action_num=33),
	pv=DeviceConfig(action_num=float('inf'), control_enabled=True),
	reward_weights=RewardWeights(power_loss=1.0, pv_control=0.0606),
))

register_preset('34Bus', SmartGridConfig(
	env_name='34Bus',
	system_name='34Bus',
	dss_file='ieee34Mod1_duty.dss',
	max_episode_steps=360,
	regulator=DeviceConfig(action_num=33),
	battery=DeviceConfig(action_num=33),
	pv=DeviceConfig(control_enabled=False),
	reward_weights=RewardWeights(power_loss=10.0),
))

register_preset('13Bus', SmartGridConfig(
	env_name='13Bus',
	system_name='13Bus',
	dss_file='IEEE13Nodeckt_daily.dss',
	max_episode_steps=24,
	regulator=DeviceConfig(action_num=33),
	battery=DeviceConfig(action_num=33),
	pv=DeviceConfig(control_enabled=False),
	reward_weights=RewardWeights(power_loss=10.0),
))
