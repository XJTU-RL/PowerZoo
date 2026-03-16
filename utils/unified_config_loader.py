# -*- coding: utf-8 -*-
"""
统一配置加载器 - PowerZoo 配置系统 v2.0

支持功能:
1. system_ref: 引用 configs/systems/{system}.yaml
2. training_ref: 引用 configs/training/{strategy}.yaml
3. 向后兼容旧格式 (environment_specific, power_system)
4. 配置合并和覆盖

使用示例:
	from utils.unified_config_loader import ConfigLoader, load_config

	# 方式1: 使用 ConfigLoader 类
	loader = ConfigLoader()
	config = loader.load('happo', 'smartgrid')

	# 方式2: 使用便捷函数
	algo_args, env_args = load_config('happo', 'smartgrid')
"""
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


# VVC legacy variant mapping
# Maps old env_name variants to {system_ref + parameter overrides}
_VVC_LEGACY_VARIANTS = {
    # 13Bus variants
    '13Bus_cbat':     {'system_ref': '13Bus', 'bat_act_num': float('inf')},
    '13Bus_soc':      {'system_ref': '13Bus', 'soc_w': 20.0 / 33},
    '13Bus_cbat_soc': {'system_ref': '13Bus', 'bat_act_num': float('inf'), 'soc_w': 20.0 / 33},
    # 34Bus variants
    '34Bus_pv':       {'system_ref': '34Bus', 'pv_control_enabled': True, 'irrad_dss': 'irrad_up_down.dss'},
    '34Bus_cbat':     {'system_ref': '34Bus', 'bat_act_num': float('inf'), 'power_w': 1.0},
    '34Bus_soc':      {'system_ref': '34Bus', 'power_w': 1.0, 'soc_w': 500.0 / 33, 'dis_w': 4.0 / 33},
    '34Bus_cbat_soc': {'system_ref': '34Bus', 'bat_act_num': float('inf'), 'power_w': 1.0, 'soc_w': 500.0 / 33, 'dis_w': 4.0 / 33},
    # 123Bus variants
    '123Bus_cbat':     {'system_ref': '123Bus', 'bat_act_num': float('inf')},
    '123Bus_soc':      {'system_ref': '123Bus', 'soc_w': 500.0 / 33, 'dis_w': 5.0 / 33},
    '123Bus_cbat_soc': {'system_ref': '123Bus', 'bat_act_num': float('inf'), 'soc_w': 500.0 / 33, 'dis_w': 5.0 / 33},
    # 8500Node variants
    '8500Node_cbat':     {'system_ref': '8500Node', 'bat_act_num': float('inf')},
    '8500Node_soc':      {'system_ref': '8500Node', 'soc_w': 10000.0 / 33, 'dis_w': 100.0 / 33},
    '8500Node_cbat_soc': {'system_ref': '8500Node', 'bat_act_num': float('inf'), 'soc_w': 10000.0 / 33, 'dis_w': 100.0 / 33},
}


def expand_legacy_variant(env_name: str) -> dict | None:
    """Expand a legacy VVC variant name to config parameters.

    Args:
        env_name: Legacy environment name (e.g., '13Bus_cbat')

    Returns:
        Dict of {system_ref + parameter overrides}, or None if not a legacy variant
    """
    return _VVC_LEGACY_VARIANTS.get(env_name)


@dataclass
class UnifiedConfig:
	"""统一配置对象

	Attributes:
		algo_args: 算法配置字典
		env_args: 环境配置字典 (已合并系统配置)
		system_config: 原始系统配置字典
		training_config: 训练策略配置字典 (可选)
	"""
	algo_args: Dict[str, Any] = field(default_factory=dict)
	env_args: Dict[str, Any] = field(default_factory=dict)
	system_config: Dict[str, Any] = field(default_factory=dict)
	training_config: Optional[Dict[str, Any]] = None


class ConfigLoader:
	"""统一配置加载器

	支持:
	- 加载算法配置 (configs/algos_cfgs/)
	- 加载环境配置 (configs/envs_cfgs/)
	- 解析系统引用 (configs/systems/)
	- 解析训练策略引用 (configs/training/)
	- 配置合并与覆盖
	"""

	def __init__(self, base_path: Optional[str] = None):
		"""初始化配置加载器

		Args:
			base_path: 项目根目录路径，默认自动检测
		"""
		self.base_path = base_path or self._get_project_root()
		self._cache: Dict[str, Dict[str, Any]] = {}  # 配置缓存

	def _get_project_root(self) -> str:
		"""获取项目根目录"""
		# 从 utils/ 目录向上一级
		current = os.path.dirname(os.path.abspath(__file__))
		return os.path.dirname(current)

	def _load_yaml(self, relative_path: str, use_cache: bool = True) -> Dict[str, Any]:
		"""加载 YAML 配置文件

		Args:
			relative_path: 相对于项目根目录的路径
			use_cache: 是否使用缓存

		Returns:
			配置字典
		"""
		full_path = os.path.join(self.base_path, relative_path)

		if use_cache and full_path in self._cache:
			return copy.deepcopy(self._cache[full_path])

		if not os.path.exists(full_path):
			raise FileNotFoundError(f"Config file not found: {full_path}")

		with open(full_path, "r", encoding="utf-8") as f:
			config = yaml.load(f, Loader=yaml.FullLoader) or {}

		if use_cache:
			self._cache[full_path] = copy.deepcopy(config)

		return config

	def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
		"""深度合并配置字典 (override 覆盖 base)

		Args:
			base: 基础配置
			override: 覆盖配置

		Returns:
			合并后的配置
		"""
		result = copy.deepcopy(base)

		for key, value in override.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._merge_configs(result[key], value)
			else:
				result[key] = copy.deepcopy(value)

		return result

	def load_system(self, system_name: str) -> Dict[str, Any]:
		"""加载系统配置

		Args:
			system_name: 系统名称 (如 '34Bus_PV')

		Returns:
			系统配置字典
		"""
		return self._load_yaml(f"configs/systems/{system_name}.yaml")

	def load_training(self, training_ref: str) -> Dict[str, Any]:
		"""加载训练策略配置

		Args:
			training_ref: 训练策略引用 (如 'curriculum/default')

		Returns:
			训练策略配置字典
		"""
		return self._load_yaml(f"configs/training/{training_ref}.yaml")

	def load(self, algo: str, env: str) -> UnifiedConfig:
		"""加载完整配置，自动解析引用

		Args:
			algo: 算法名称 (如 'happo')
			env: 环境名称 (如 'smartgrid')

		Returns:
			UnifiedConfig 对象
		"""
		# 加载算法配置
		algo_args = self._load_yaml(f"configs/algos_cfgs/{algo}.yaml")

		# 加载环境配置
		env_config = self._load_yaml(f"configs/envs_cfgs/{env}.yaml")

		# 解析系统引用
		system_config = {}
		if 'system_ref' in env_config:
			system_ref = env_config['system_ref']
			try:
				system_config = self.load_system(system_ref)
			except FileNotFoundError:
				print(f"Warning: System config '{system_ref}' not found, using env_config only")

		# 解析训练策略引用
		training_config = None
		if 'training_ref' in env_config:
			training_ref = env_config['training_ref']
			try:
				training_config = self.load_training(training_ref)
			except FileNotFoundError:
				print(f"Warning: Training config '{training_ref}' not found")

		# 构建 env_args
		env_args = self._build_env_args(env_config, system_config, training_config)

		return UnifiedConfig(
			algo_args=algo_args,
			env_args=env_args,
			system_config=system_config,
			training_config=training_config
		)

	def _build_env_args(
		self,
		env_config: Dict[str, Any],
		system_config: Dict[str, Any],
		training_config: Optional[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""构建 env_args 字典

		优先级 (高到低):
		1. env_config 中的显式值
		2. reward_overrides / constraint_overrides
		3. system_config 中的值
		4. training_config 中的值 (仅用于 power_system)
		5. environment_specific (向后兼容)

		Args:
			env_config: 环境配置
			system_config: 系统配置
			training_config: 训练策略配置

		Returns:
			构建好的 env_args 字典
		"""
		env_args = {}

		# 1. 从 system_config 提取基础配置
		if system_config:
			# 系统基本信息
			if 'system' in system_config:
				sys_info = system_config['system']
				env_args['system_name'] = sys_info.get('name')
				env_args['dss_file'] = sys_info.get('dss_file')
				env_args['dss_folder'] = sys_info.get('dss_folder')
				env_args['source_bus'] = sys_info.get('source_bus', 'sourcebus')

			# Episode 配置
			if 'episode' in system_config:
				env_args['max_episode_steps'] = system_config['episode'].get('max_steps')

			# 设备配置 -> env_specific_config (兼容旧代码)
			if 'devices' in system_config:
				env_args['env_specific_config'] = {
					'devices': system_config['devices'],
					'system_name': system_config.get('system', {}).get('name'),
					'dss_file': system_config.get('system', {}).get('dss_file'),
				}

			# 约束配置
			if 'constraints' in system_config:
				env_args['voltage_min'] = system_config['constraints'].get('voltage_min', 0.95)
				env_args['voltage_max'] = system_config['constraints'].get('voltage_max', 1.05)

			# 默认奖励权重
			if 'default_rewards' in system_config:
				if 'env_specific_config' not in env_args:
					env_args['env_specific_config'] = {}
				env_args['env_specific_config']['reward_weights'] = system_config['default_rewards']

			# 显示配置
			if 'display' in system_config:
				env_args['node_size'] = system_config['display'].get('node_size')
				env_args['shift'] = system_config['display'].get('shift')
				env_args['show_node_labels'] = system_config['display'].get('show_labels')

		# 2. 从 training_config 提取 power_system 配置
		if training_config:
			power_system = {
				'curriculum_learning': training_config.get('curriculum_learning', False),
				'phases': training_config.get('phases', []),
			}
			if 'penalties' in training_config:
				power_system['voltage_violation_penalty'] = training_config['penalties'].get('voltage_violation')
				power_system['power_loss_weight'] = training_config['penalties'].get('power_loss_weight')
				power_system['control_cost_weight'] = training_config['penalties'].get('control_cost_weight')
			if 'convergence' in training_config:
				power_system['convergence'] = training_config['convergence']

			env_args['power_system'] = power_system

		# 3. 从 env_config 提取直接配置 (覆盖上面的值)
		skip_keys = {'system_ref', 'training_ref', 'reward_overrides', 'constraint_overrides'}

		for key, value in env_config.items():
			if key in skip_keys:
				continue
			if key == 'environment_specific':
				# 向后兼容: 合并而非覆盖
				if 'env_specific_config' not in env_args:
					env_args['env_specific_config'] = {}
				env_args['env_specific_config'] = self._merge_configs(
					env_args['env_specific_config'],
					value
				)
			elif key == 'power_system':
				# 向后兼容: 合并而非覆盖
				if 'power_system' not in env_args:
					env_args['power_system'] = {}
				env_args['power_system'] = self._merge_configs(
					env_args['power_system'],
					value
				)
			elif key == 'env_args' and isinstance(value, dict):
				# 兼容嵌套 env_args: 段，将内容扁平化到顶层
				for sub_key, sub_value in value.items():
					env_args[sub_key] = sub_value
			elif key == 'episode_length':
				# 映射到 max_episode_steps
				env_args['max_episode_steps'] = value
			else:
				env_args[key] = value

		# 4. 应用奖励权重覆盖
		if 'reward_overrides' in env_config and env_config['reward_overrides']:
			if 'env_specific_config' not in env_args:
				env_args['env_specific_config'] = {}
			if 'reward_weights' not in env_args['env_specific_config']:
				env_args['env_specific_config']['reward_weights'] = {}
			env_args['env_specific_config']['reward_weights'].update(env_config['reward_overrides'])

		# 5. 应用约束覆盖
		if 'constraint_overrides' in env_config and env_config['constraint_overrides']:
			env_args.update(env_config['constraint_overrides'])

		return env_args

	def clear_cache(self):
		"""清除配置缓存"""
		self._cache.clear()


# 便捷函数
_default_loader: Optional[ConfigLoader] = None


def get_loader() -> ConfigLoader:
	"""获取默认配置加载器 (单例)"""
	global _default_loader
	if _default_loader is None:
		_default_loader = ConfigLoader()
	return _default_loader


def load_config(algo: str, env: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
	"""便捷函数: 加载配置并返回 (algo_args, env_args) 元组

	兼容 get_defaults_yaml_args() 的返回格式

	Args:
		algo: 算法名称
		env: 环境名称

	Returns:
		(algo_args, env_args) 元组
	"""
	loader = get_loader()
	config = loader.load(algo, env)
	return config.algo_args, config.env_args


def load_system_config(system_name: str) -> Dict[str, Any]:
	"""便捷函数: 加载系统配置

	Args:
		system_name: 系统名称

	Returns:
		系统配置字典
	"""
	return get_loader().load_system(system_name)


def load_training_config(training_ref: str) -> Dict[str, Any]:
	"""便捷函数: 加载训练策略配置

	Args:
		training_ref: 训练策略引用

	Returns:
		训练策略配置字典
	"""
	return get_loader().load_training(training_ref)
