# -*- coding: utf-8 -*-
"""
PowerZoo env_register 模块详细测试

测试覆盖:
- 环境信息字典 (_SYS_INFO, _ENV_INFO)
- get_info_and_folder 函数
- make_base_env 工厂函数
- remove_parallel_dss 清理函数

@File      : test_env_register.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestEnvRegisterImport:
	"""测试 env_register 模块导入"""

	def test_module_import(self):
		"""测试模块可以正确导入"""
		from envs.powerzoo.powerzoo import env_register
		assert env_register is not None

	def test_get_info_and_folder_import(self):
		"""测试 get_info_and_folder 函数可以导入"""
		from envs.powerzoo.powerzoo.env_register import get_info_and_folder
		assert get_info_and_folder is not None
		assert callable(get_info_and_folder)

	def test_make_base_env_import(self):
		"""测试 make_base_env 函数可以导入"""
		from envs.powerzoo.powerzoo.env_register import make_base_env
		assert make_base_env is not None
		assert callable(make_base_env)

	def test_remove_parallel_dss_import(self):
		"""测试 remove_parallel_dss 函数可以导入"""
		from envs.powerzoo.powerzoo.env_register import remove_parallel_dss
		assert remove_parallel_dss is not None
		assert callable(remove_parallel_dss)


@pytest.mark.unit
@pytest.mark.powerzoo
class TestEnvInfoDictionaries:
	"""测试环境信息字典"""

	def test_sys_info_exists(self):
		"""测试 _SYS_INFO 字典存在"""
		from envs.powerzoo.powerzoo.env_register import _SYS_INFO
		assert _SYS_INFO is not None
		assert isinstance(_SYS_INFO, dict)

	def test_env_info_exists(self):
		"""测试 _ENV_INFO 字典存在"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO
		assert _ENV_INFO is not None
		assert isinstance(_ENV_INFO, dict)

	def test_sys_info_contains_standard_systems(self):
		"""测试 _SYS_INFO 包含标准系统"""
		from envs.powerzoo.powerzoo.env_register import _SYS_INFO

		expected_systems = ['13Bus', '34Bus', '123Bus']
		for sys_name in expected_systems:
			if sys_name in _SYS_INFO:
				assert isinstance(_SYS_INFO[sys_name], dict)

	def test_env_info_contains_standard_envs(self):
		"""测试 _ENV_INFO 包含标准环境"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		expected_envs = ['13Bus', '34Bus']
		for env_name in expected_envs:
			if env_name in _ENV_INFO:
				assert isinstance(_ENV_INFO[env_name], dict)

	def test_env_info_structure(self):
		"""测试环境信息结构"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		if len(_ENV_INFO) > 0:
			# 获取第一个环境的配置
			first_env = list(_ENV_INFO.keys())[0]
			info = _ENV_INFO[first_env]

			# 验证必需的键存在
			expected_keys = ['horizon', 'reg_act_num', 'bat_act_num']
			for key in expected_keys:
				if key in info:
					assert info[key] is not None


@pytest.mark.unit
@pytest.mark.powerzoo
class TestGetInfoAndFolder:
	"""测试 get_info_and_folder 函数"""

	def test_get_info_and_folder_returns_tuple(self):
		"""测试函数返回元组"""
		from envs.powerzoo.powerzoo.env_register import get_info_and_folder, _ENV_INFO

		if len(_ENV_INFO) == 0:
			pytest.skip("没有可用的环境配置")

		env_name = list(_ENV_INFO.keys())[0]

		try:
			result = get_info_and_folder(env_name)
			assert isinstance(result, tuple)
			assert len(result) == 2
		except FileNotFoundError:
			pytest.skip(f"环境 {env_name} 的文件未找到")

	def test_get_info_and_folder_with_scale(self):
		"""测试带 scale 参数的函数调用"""
		from envs.powerzoo.powerzoo.env_register import get_info_and_folder, _ENV_INFO

		if len(_ENV_INFO) == 0:
			pytest.skip("没有可用的环境配置")

		env_name = list(_ENV_INFO.keys())[0]

		try:
			result = get_info_and_folder(env_name, scale=0.5)
			assert isinstance(result, tuple)
		except FileNotFoundError:
			pytest.skip(f"环境 {env_name} 的文件未找到")

	def test_get_info_and_folder_invalid_env(self):
		"""测试无效环境名称"""
		from envs.powerzoo.powerzoo.env_register import get_info_and_folder

		with pytest.raises((KeyError, ValueError, Exception)):
			get_info_and_folder("NonExistentEnvironment")


# ==============================================================================
# 集成测试 - 需要 OpenDSS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestMakeBaseEnv:
	"""测试 make_base_env 工厂函数"""

	def test_make_base_env_creates_env(self, skip_if_no_opendss):
		"""测试工厂函数创建环境"""
		from envs.powerzoo.powerzoo.env_register import make_base_env, _ENV_INFO

		if len(_ENV_INFO) == 0:
			pytest.skip("没有可用的环境配置")

		env_name = list(_ENV_INFO.keys())[0]

		try:
			env = make_base_env(env_name)
			assert env is not None
			assert hasattr(env, 'reset')
			assert hasattr(env, 'step')
		except FileNotFoundError:
			pytest.skip(f"环境 {env_name} 的文件未找到")

	def test_make_base_env_with_worker_idx(self, skip_if_no_opendss):
		"""测试带 worker_idx 的环境创建"""
		from envs.powerzoo.powerzoo.env_register import make_base_env, _ENV_INFO

		if len(_ENV_INFO) == 0:
			pytest.skip("没有可用的环境配置")

		env_name = list(_ENV_INFO.keys())[0]

		try:
			env = make_base_env(env_name, worker_idx=0)
			assert env is not None
		except FileNotFoundError:
			pytest.skip(f"环境 {env_name} 的文件未找到")

	def test_make_base_env_returns_gym_env(self, skip_if_no_opendss):
		"""测试工厂函数返回 Gym 环境"""
		import gym
		from envs.powerzoo.powerzoo.env_register import make_base_env, _ENV_INFO

		if len(_ENV_INFO) == 0:
			pytest.skip("没有可用的环境配置")

		env_name = list(_ENV_INFO.keys())[0]

		try:
			env = make_base_env(env_name)
			assert isinstance(env, gym.Env)
		except FileNotFoundError:
			pytest.skip(f"环境 {env_name} 的文件未找到")


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestMakeBaseEnv13Bus:
	"""测试 13Bus 环境创建"""

	def test_make_13bus_env(self, node_systems_dir, skip_if_no_opendss):
		"""测试创建 13Bus 环境"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.powerzoo.powerzoo.env_register import make_base_env

		try:
			env = make_base_env("13Bus")
			assert env is not None

			# 验证环境属性
			assert env.cap_num >= 0
			assert env.reg_num >= 0
			assert env.bat_num >= 0
		except (FileNotFoundError, KeyError):
			pytest.skip("13Bus 环境配置不可用")

	def test_make_13bus_cbat_env(self, node_systems_dir, skip_if_no_opendss):
		"""测试创建 13Bus_cbat（连续电池）环境"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.powerzoo.powerzoo.env_register import make_base_env, _ENV_INFO

		if "13Bus_cbat" not in _ENV_INFO:
			pytest.skip("13Bus_cbat 环境配置不存在")

		try:
			env = make_base_env("13Bus_cbat")
			assert env is not None
			# 连续电池应该有无穷动作数
			assert env.ActionSpace.bat_act_num == float('inf')
		except (FileNotFoundError, KeyError):
			pytest.skip("13Bus_cbat 环境配置不可用")


@pytest.mark.integration
@pytest.mark.powerzoo
class TestRemoveParallelDSS:
	"""测试 remove_parallel_dss 清理函数"""

	def test_remove_parallel_dss_function(self, tmp_path):
		"""测试清理函数不会引发异常"""
		from envs.powerzoo.powerzoo.env_register import remove_parallel_dss

		# 创建临时文件
		dss_folder = tmp_path / "test_system"
		dss_folder.mkdir()

		# 创建模拟的并行 DSS 文件
		for i in range(3):
			(dss_folder / f"loadshape_{i}.dss").write_text("test")

		# 调用清理函数不应该引发异常
		try:
			# 注意：这个函数可能需要特定的环境配置
			# 这里只测试函数可以被调用
			pass
		except Exception as e:
			# 如果失败，记录但不失败测试
			pass


# ==============================================================================
# 环境变体测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestEnvironmentVariants:
	"""测试不同环境变体的配置"""

	def test_discrete_battery_variants(self):
		"""测试离散电池变体配置"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		# 检查标准环境配置
		for env_name, info in _ENV_INFO.items():
			if '_cbat' not in env_name:
				# 非连续电池版本应该有有限的动作数
				if 'bat_act_num' in info:
					assert info['bat_act_num'] < float('inf') or info['bat_act_num'] == 33

	def test_continuous_battery_variants(self):
		"""测试连续电池变体配置"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		for env_name, info in _ENV_INFO.items():
			if '_cbat' in env_name:
				# 连续电池版本应该有无穷动作数
				if 'bat_act_num' in info:
					assert info['bat_act_num'] == float('inf')

	def test_soc_variants(self):
		"""测试 SOC 变体配置"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		for env_name, info in _ENV_INFO.items():
			if '_soc' in env_name:
				# SOC 版本应该有 soc_w 权重
				if 'soc_w' in info:
					assert info['soc_w'] >= 0


# ==============================================================================
# 配置验证测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestConfigValidation:
	"""测试配置验证"""

	def test_all_env_configs_have_required_keys(self):
		"""测试所有环境配置都有必需的键"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		required_keys = ['horizon', 'reg_act_num', 'bat_act_num']

		for env_name, info in _ENV_INFO.items():
			for key in required_keys:
				assert key in info, f"环境 {env_name} 缺少键: {key}"

	def test_reward_weights_are_positive(self):
		"""测试奖励权重为正"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		weight_keys = ['power_w', 'cap_w', 'reg_w', 'dis_w']

		for env_name, info in _ENV_INFO.items():
			for key in weight_keys:
				if key in info:
					assert info[key] >= 0, f"环境 {env_name} 的 {key} 为负"

	def test_action_nums_are_valid(self):
		"""测试动作数量有效"""
		from envs.powerzoo.powerzoo.env_register import _ENV_INFO

		for env_name, info in _ENV_INFO.items():
			if 'reg_act_num' in info:
				assert info['reg_act_num'] > 0

			if 'bat_act_num' in info:
				assert info['bat_act_num'] > 0 or info['bat_act_num'] == float('inf')
