"""
SmartGrid 环境配置加载器

简化版本 - 单一入口，输出 SmartGridConfig

配置优先级（从高到低）：
1. 代码中直接传入的参数
2. YAML 配置文件 (smartgrid.yaml)
3. JSON 环境信息文件 (environments_info.json)
4. 预定义配置 (PRESET_CONFIGS)
5. 默认值
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from envs.smartgrid.base_env.env_config import (
	SmartGridConfig,
	DeviceConfig,
	RewardWeights,
	PRESET_CONFIGS,
	get_preset,
)


class ConfigLoader:
	"""统一的配置加载器

	设计原则：
	- 单例模式，避免重复IO
	- 所有配置合并为 SmartGridConfig
	- 清晰的优先级规则
	"""

	_instance: Optional['ConfigLoader'] = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
			cls._instance._initialized = False
		return cls._instance

	def __init__(self):
		if self._initialized:
			return

		self.project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
		self.configs_dir = self.project_root / 'configs'

		# 一次性加载所有原始配置（用于兼容旧代码）
		self._system_info = self._load_json('sys_cfgs/system_info.json', 'system_info')
		self._environments_info = self._load_json('sys_cfgs/environments_info.json', 'environments')
		self._smartgrid_yaml = self._load_yaml('envs_cfgs/smartgrid.yaml')

		self._initialized = True

	def _load_json(self, relative_path: str, key: str) -> Dict[str, Any]:
		"""加载 JSON 配置文件"""
		file_path = self.configs_dir / relative_path
		if file_path.exists():
			with open(file_path, 'r', encoding='utf-8') as f:
				data = json.load(f)
				result = data.get(key, {})
				self._process_special_values(result)
				return result
		return {}

	def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
		"""加载 YAML 配置文件"""
		file_path = self.configs_dir / relative_path
		if file_path.exists():
			with open(file_path, 'r', encoding='utf-8') as f:
				return yaml.safe_load(f) or {}
		return {}

	def _process_special_values(self, config: Any) -> None:
		"""处理配置中的特殊值（如 'inf' 字符串）"""
		if isinstance(config, dict):
			for key, value in config.items():
				if value == 'inf':
					config[key] = float('inf')
				elif isinstance(value, dict):
					self._process_special_values(value)

	def get_config(
		self,
		env_name: str,
		overrides: Optional[Dict[str, Any]] = None
	) -> SmartGridConfig:
		"""获取环境配置 - 统一入口

		Args:
			env_name: 环境名称（如 '34Bus_pv', '13Bus'）
			overrides: 覆盖参数（优先级最高）

		Returns:
			SmartGridConfig: 完整的类型安全配置对象
		"""
		overrides = overrides or {}

		# 1. 尝试从预定义配置获取基础
		if env_name in PRESET_CONFIGS:
			base_config = get_preset(env_name)
		else:
			# 2. 从 environments_info.json 构建
			base_config = self._build_from_environments_info(env_name)

		# 3. 合并 smartgrid.yaml 中的环境特定配置
		base_config = self._merge_yaml_config(base_config)

		# 4. 应用覆盖参数
		base_config = self._apply_overrides(base_config, overrides)

		return base_config

	def _build_from_environments_info(self, env_name: str) -> SmartGridConfig:
		"""从 environments_info.json 构建配置"""
		if env_name not in self._environments_info:
			# 如果找不到，尝试模糊匹配
			matched = self._fuzzy_match_env_name(env_name)
			if matched:
				env_name = matched
			else:
				# 返回默认配置
				return SmartGridConfig(env_name=env_name)

		info = self._environments_info[env_name].copy()

		# 合并 system_info
		system_name = info.get('system_name', '')
		if system_name in self._system_info:
			info.update(self._system_info[system_name])

		# 转换为 SmartGridConfig
		return SmartGridConfig.from_dict(info)

	def _fuzzy_match_env_name(self, env_name: str) -> Optional[str]:
		"""模糊匹配环境名称

		处理路径形式的环境名（如 /home/xxx/node_systems/34Bus_PV_Aggressive）
		"""
		# 处理路径形式
		if 'node_systems' in env_name or env_name.startswith('/'):
			for known_env in ['34Bus_pv', '34Bus', '13Bus', '123Bus', '8500Node']:
				if known_env.lower().replace('_', '') in env_name.lower().replace('_', ''):
					return known_env

		# 处理变体名称
		name_lower = env_name.lower()
		for known_env in self._environments_info.keys():
			if known_env.lower() == name_lower:
				return known_env

		return None

	def _merge_yaml_config(self, config: SmartGridConfig) -> SmartGridConfig:
		"""合并 smartgrid.yaml 配置"""
		if not self._smartgrid_yaml:
			return config

		yaml_cfg = self._smartgrid_yaml

		# 从 environment_specific.devices 提取设备配置
		devices = yaml_cfg.get('environment_specific', {}).get('devices', {})

		if devices:
			# 调压器
			if 'regulators' in devices:
				reg_cfg = devices['regulators']
				config.regulator.action_num = reg_cfg.get('action_num', config.regulator.action_num)

			# 电池
			if 'batteries' in devices:
				bat_cfg = devices['batteries']
				action_num = bat_cfg.get('action_num', config.battery.action_num)
				if bat_cfg.get('action_space') == 'continuous':
					action_num = float('inf')
				config.battery.action_num = action_num

			# 光伏
			if 'pv_systems' in devices:
				pv_cfg = devices['pv_systems']
				config.pv.control_enabled = pv_cfg.get('control_enabled', config.pv.control_enabled)
				if pv_cfg.get('action_space') == 'continuous':
					config.pv.action_num = float('inf')
				elif 'action_num' in pv_cfg:
					config.pv.action_num = pv_cfg['action_num']

		# 从 environment_specific.reward_weights 提取奖励权重
		weights = yaml_cfg.get('environment_specific', {}).get('reward_weights', {})
		if weights:
			config.reward_weights.power_loss = weights.get('power_loss', config.reward_weights.power_loss)
			config.reward_weights.capacitor = weights.get('capacitor', config.reward_weights.capacitor)
			config.reward_weights.regulator = weights.get('regulator', config.reward_weights.regulator)
			config.reward_weights.battery_soc = weights.get('battery_soc', config.reward_weights.battery_soc)
			config.reward_weights.battery_discharge = weights.get('battery_discharge', config.reward_weights.battery_discharge)
			config.reward_weights.pv_control = weights.get('pv_control', config.reward_weights.pv_control)

		# 从 environment_specific.constraints 提取约束
		constraints = yaml_cfg.get('environment_specific', {}).get('constraints', {})
		if constraints:
			config.voltage_min = constraints.get('voltage_min', config.voltage_min)
			config.voltage_max = constraints.get('voltage_max', config.voltage_max)

		# 顶级配置
		if 'num_steps' in yaml_cfg:
			config.max_episode_steps = yaml_cfg['num_steps']
		if 'seed' in yaml_cfg:
			config.seed = yaml_cfg['seed']
		if 'dss_act' in yaml_cfg:
			config.dss_act = yaml_cfg['dss_act']

		return config

	def _apply_overrides(
		self,
		config: SmartGridConfig,
		overrides: Dict[str, Any]
	) -> SmartGridConfig:
		"""应用覆盖参数"""
		if not overrides:
			return config

		# 基础属性
		if 'max_episode_steps' in overrides:
			config.max_episode_steps = overrides['max_episode_steps']
		if 'episode_length' in overrides:
			config.max_episode_steps = overrides['episode_length']
		if 'num_steps' in overrides:
			config.max_episode_steps = overrides['num_steps']

		# 设备配置
		if 'reg_act_num' in overrides:
			config.regulator.action_num = overrides['reg_act_num']
		if 'bat_act_num' in overrides:
			config.battery.action_num = overrides['bat_act_num']
		if 'pv_act_num' in overrides:
			config.pv.action_num = overrides['pv_act_num']
		if 'pv_control' in overrides:
			config.pv.control_enabled = overrides['pv_control']

		# 运行时配置
		if 'for_LLM' in overrides:
			config.for_LLM = overrides['for_LLM']
		if 'llm_enhanced' in overrides:
			config.for_LLM = overrides['llm_enhanced']
		if 'dss_act' in overrides:
			config.dss_act = overrides['dss_act']
		if 'worker_idx' in overrides:
			config.worker_idx = overrides['worker_idx']

		# 系统配置
		if 'system_name' in overrides:
			config.system_name = overrides['system_name']
		if 'dss_file' in overrides:
			config.dss_file = overrides['dss_file']

		return config


# 全局单例
_loader = ConfigLoader()


def load_config(
	env_name: str,
	overrides: Optional[Dict[str, Any]] = None
) -> SmartGridConfig:
	"""加载环境配置 - 推荐的公共接口

	Args:
		env_name: 环境名称
		overrides: 覆盖参数

	Returns:
		SmartGridConfig 对象

	示例:
		config = load_config('34Bus_pv')
		config = load_config('34Bus_pv', {'max_episode_steps': 100})
	"""
	return _loader.get_config(env_name, overrides)


def get_env_config(env_name: str, env_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	"""获取环境配置 - 向后兼容接口

	返回旧格式的 info 字典，用于兼容现有代码。
	新代码应使用 load_config() 获取 SmartGridConfig。

	Args:
		env_name: 环境名称
		env_args: 覆盖参数（来自旧的 env_args）

	Returns:
		info 字典（旧格式）
	"""
	config = load_config(env_name, env_args)
	return config.to_info_dict()
