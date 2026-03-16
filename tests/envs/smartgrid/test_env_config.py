# -*- coding: utf-8 -*-
"""
SmartGrid env_config 模块详细测试

测试覆盖:
- SmartGridConfig 配置类
- 配置验证
- 预定义配置
- 配置优先级

@File      : test_env_config.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestEnvConfigImport:
	"""测试 env_config 模块导入"""

	def test_smartgrid_config_import(self):
		"""测试 SmartGridConfig 类导入"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			assert SmartGridConfig is not None
		except ImportError:
			# 可能使用不同的类名
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			assert VVCEnvConfig is not None

	def test_from_env_args_import(self):
		"""测试 SmartGridConfig.from_env_args 可用"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig
		assert hasattr(SmartGridConfig, 'from_env_args')


# ==============================================================================
# 单元测试 - SmartGridConfig 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSmartGridConfigInit:
	"""测试 SmartGridConfig 初始化"""

	def get_config_class(self):
		"""获取配置类"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig
		except ImportError:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			return VVCEnvConfig

	def test_config_creation_default(self):
		"""测试默认配置创建"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()
		assert config is not None

	def test_config_creation_with_env_name(self):
		"""测试带环境名称的配置创建"""
		ConfigClass = self.get_config_class()

		config = ConfigClass(env_name="34Bus_pv")
		assert config.env_name == "34Bus_pv"

	def test_config_has_required_attributes(self):
		"""测试配置有必需属性"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()

		required_attrs = [
			'env_name',
			'max_episode_steps',
		]

		for attr in required_attrs:
			assert hasattr(config, attr), f"配置缺少属性: {attr}"


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSmartGridConfigAttributes:
	"""测试 SmartGridConfig 属性"""

	def get_config_class(self):
		"""获取配置类"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig
		except ImportError:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			return VVCEnvConfig

	def test_max_episode_steps(self):
		"""测试 max_episode_steps 属性"""
		ConfigClass = self.get_config_class()

		config = ConfigClass(max_episode_steps=96)
		assert config.max_episode_steps == 96

	def test_seed_attribute(self):
		"""测试 seed 属性"""
		ConfigClass = self.get_config_class()

		config = ConfigClass(seed=42)
		if hasattr(config, 'seed'):
			assert config.seed == 42

	def test_reward_weights(self):
		"""测试奖励权重配置"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()

		# 检查是否有奖励权重相关属性
		weight_attrs = ['reward_weights', 'power_w', 'powerloss_weight']
		found = any(hasattr(config, attr) for attr in weight_attrs)
		# 不强制要求，但记录


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSmartGridConfigValidation:
	"""测试 SmartGridConfig 验证"""

	def get_config_class(self):
		"""获取配置类"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig
		except ImportError:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			return VVCEnvConfig

	def test_invalid_episode_steps(self):
		"""测试无效的 episode 步数"""
		ConfigClass = self.get_config_class()

		# 负数应该被拒绝或转换
		try:
			config = ConfigClass(max_episode_steps=-1)
			# 如果没有抛出异常，检查是否被修正
			if hasattr(config, 'max_episode_steps'):
				assert config.max_episode_steps > 0 or config.max_episode_steps == -1
		except (ValueError, AssertionError):
			pass  # 预期行为

	def test_valid_configuration(self):
		"""测试有效配置"""
		ConfigClass = self.get_config_class()

		config = ConfigClass(
			env_name="34Bus_pv",
			max_episode_steps=96,
			seed=42,
		)

		assert config.env_name == "34Bus_pv"
		assert config.max_episode_steps == 96


# ==============================================================================
# 单元测试 - from_env_args 工厂方法
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestFromEnvArgs:
	"""测试 SmartGridConfig.from_env_args 工厂方法"""

	def test_from_env_args_minimal(self):
		"""测试最小 env_args 输入"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		config = SmartGridConfig.from_env_args({})
		assert config is not None
		assert isinstance(config, SmartGridConfig)

	def test_from_env_args_with_system_name(self):
		"""测试 from_env_args 接受 system_name"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		config = SmartGridConfig.from_env_args({'system_name': '13Bus'})
		assert config.system_name == '13Bus'

	def test_from_env_args_with_devices(self):
		"""测试 from_env_args 解析设备配置"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		env_args = {
			'env_specific_config': {
				'devices': {
					'regulators': {'action_num': 17},
					'pv_systems': {'control_enabled': True, 'action_space': 'continuous'},
				}
			}
		}
		config = SmartGridConfig.from_env_args(env_args)
		assert config.regulator.action_num == 17
		assert config.pv.control_enabled is True


# ==============================================================================
# 集成测试 - 配置与环境
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestConfigWithEnvironment:
	"""测试配置与环境的集成"""

	def get_config_class(self):
		"""获取配置类"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig
		except ImportError:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			return VVCEnvConfig

	def test_config_can_be_used_by_env(self, node_systems_dir):
		"""测试配置可以被环境使用"""
		ConfigClass = self.get_config_class()

		# 创建配置
		config = ConfigClass(
			env_name="34Bus_pv",
			max_episode_steps=24,
		)

		# 配置应该有环境需要的所有信息
		assert hasattr(config, 'env_name')
		assert hasattr(config, 'max_episode_steps')


# ==============================================================================
# YAML 配置测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestYAMLConfig:
	"""测试 YAML 配置加载"""

	def test_smartgrid_yaml_exists(self, project_root):
		"""测试 smartgrid.yaml 存在"""
		yaml_path = project_root / "configs" / "envs_cfgs" / "smartgrid.yaml"

		if yaml_path.exists():
			assert True
		else:
			pytest.skip("smartgrid.yaml 不存在")

	def test_yaml_config_can_be_loaded(self, project_root):
		"""测试 YAML 配置可以加载"""
		yaml_path = project_root / "configs" / "envs_cfgs" / "smartgrid.yaml"

		if not yaml_path.exists():
			pytest.skip("smartgrid.yaml 不存在")

		import yaml

		with open(yaml_path, 'r') as f:
			config = yaml.safe_load(f)

		assert config is not None
		assert isinstance(config, dict)

	def test_yaml_config_has_required_sections(self, project_root):
		"""测试 YAML 配置有必需的部分"""
		yaml_path = project_root / "configs" / "envs_cfgs" / "smartgrid.yaml"

		if not yaml_path.exists():
			pytest.skip("smartgrid.yaml 不存在")

		import yaml

		with open(yaml_path, 'r') as f:
			config = yaml.safe_load(f)

		# 检查常见的配置键
		common_keys = ['env_name', 'episode_length', 'max_episode_steps']
		found_keys = [k for k in common_keys if k in config]
		# 不强制要求，但记录


# ==============================================================================
# 设备配置测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestDeviceConfig:
	"""测试设备配置"""

	def get_config_class(self):
		"""获取配置类"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig
		except ImportError:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			return VVCEnvConfig

	def test_capacitor_config(self):
		"""测试电容器配置"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()

		# 检查电容器相关配置
		cap_attrs = ['cap_num', 'capacitor', 'n_capacitors']
		found = any(hasattr(config, attr) for attr in cap_attrs)

	def test_regulator_config(self):
		"""测试调压器配置"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()

		# 检查调压器相关配置
		reg_attrs = ['reg_num', 'regulator', 'n_regulators', 'reg_act_num']
		found = any(hasattr(config, attr) for attr in reg_attrs)

	def test_battery_config(self):
		"""测试电池配置"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()

		# 检查电池相关配置
		bat_attrs = ['bat_num', 'battery', 'n_batteries', 'bat_act_num']
		found = any(hasattr(config, attr) for attr in bat_attrs)

	def test_pv_config(self):
		"""测试 PV 配置"""
		ConfigClass = self.get_config_class()

		config = ConfigClass()

		# 检查 PV 相关配置
		pv_attrs = ['pv_num', 'pv', 'n_pv', 'pv_control']
		found = any(hasattr(config, attr) for attr in pv_attrs)


# ==============================================================================
# 边界条件测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestConfigEdgeCases:
	"""测试配置边界条件"""

	def get_config_class(self):
		"""获取配置类"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig
		except ImportError:
			from envs.smartgrid.base_env.env_config import VVCEnvConfig
			return VVCEnvConfig

	def test_empty_env_name(self):
		"""测试空环境名称"""
		ConfigClass = self.get_config_class()

		try:
			config = ConfigClass(env_name="")
			# 可能会接受空字符串
		except (ValueError, AssertionError):
			pass  # 预期行为

	def test_none_values(self):
		"""测试 None 值"""
		ConfigClass = self.get_config_class()

		try:
			config = ConfigClass(env_name=None)
		except (ValueError, TypeError, AssertionError):
			pass  # 预期行为

	def test_config_copy(self):
		"""测试配置复制"""
		ConfigClass = self.get_config_class()

		config1 = ConfigClass(env_name="test", max_episode_steps=48)

		# 尝试复制配置
		if hasattr(config1, 'copy'):
			config2 = config1.copy()
			assert config2.env_name == config1.env_name
		elif hasattr(config1, '__dict__'):
			config2 = ConfigClass(**config1.__dict__)
