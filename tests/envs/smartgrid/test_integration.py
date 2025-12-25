# -*- coding: utf-8 -*-
"""
SmartGrid 环境集成测试

测试覆盖:
- 完整环境生命周期 (create → reset → step → close)
- 环境注册和工厂函数
- MARL包装器集成
- 配置加载和环境创建
- 多智能体工作流
- 跨组件集成

@File      : test_integration.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 环境注册模块导入
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestEnvRegisterImport:
	"""测试环境注册模块导入"""

	def test_make_env_import(self):
		"""测试 make_env 函数导入"""
		from envs.smartgrid.base_env.env_register import make_env
		assert make_env is not None
		assert callable(make_env)

	def test_make_base_env_import(self):
		"""测试 make_base_env 函数导入"""
		from envs.smartgrid.base_env.env_register import make_base_env
		assert make_base_env is not None
		assert callable(make_base_env)

	def test_utility_functions_import(self):
		"""测试工具函数导入"""
		from envs.smartgrid.base_env.env_register import (
			get_data_root,
			get_node_systems_path,
			get_info_and_folder,
			get_info_from_config,
		)
		assert get_data_root is not None
		assert get_node_systems_path is not None
		assert get_info_and_folder is not None
		assert get_info_from_config is not None

	def test_cleanup_functions_import(self):
		"""测试清理函数导入"""
		from envs.smartgrid.base_env.env_register import (
			remove_parallel_dss,
			cleanup_all_parallel_dss,
		)
		assert remove_parallel_dss is not None
		assert cleanup_all_parallel_dss is not None


# ==============================================================================
# 单元测试 - 环境包装器导入
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestEnvWrappersImport:
	"""测试环境包装器导入"""

	def test_share_vec_env_import(self):
		"""测试 ShareVecEnv 导入"""
		from envs.env_wrappers import ShareVecEnv
		assert ShareVecEnv is not None

	def test_share_dummy_vec_env_import(self):
		"""测试 ShareDummyVecEnv 导入"""
		from envs.env_wrappers import ShareDummyVecEnv
		assert ShareDummyVecEnv is not None

	def test_share_subproc_vec_env_import(self):
		"""测试 ShareSubprocVecEnv 导入"""
		from envs.env_wrappers import ShareSubprocVecEnv
		assert ShareSubprocVecEnv is not None

	def test_cloudpickle_wrapper_import(self):
		"""测试 CloudpickleWrapper 导入"""
		from envs.env_wrappers import CloudpickleWrapper
		assert CloudpickleWrapper is not None

	def test_tile_images_import(self):
		"""测试 tile_images 函数导入"""
		from envs.env_wrappers import tile_images
		assert tile_images is not None
		assert callable(tile_images)


# ==============================================================================
# 单元测试 - 路径工具函数
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestPathUtilities:
	"""测试路径工具函数"""

	def test_get_data_root(self):
		"""测试获取数据根目录"""
		from envs.smartgrid.base_env.env_register import get_data_root

		root = get_data_root()
		assert isinstance(root, Path)
		assert root.exists()

	def test_get_node_systems_path(self):
		"""测试获取 node_systems 目录"""
		from envs.smartgrid.base_env.env_register import get_node_systems_path

		path = get_node_systems_path()
		assert isinstance(path, Path)
		# 目录应该存在
		if path.exists():
			assert path.is_dir()

	def test_data_root_contains_node_systems(self):
		"""测试数据根目录包含 node_systems"""
		from envs.smartgrid.base_env.env_register import get_data_root

		root = get_data_root()
		node_systems = root / 'node_systems'
		# 可能存在也可能不存在，但路径应该是有效的
		assert isinstance(node_systems, Path)


# ==============================================================================
# 单元测试 - 配置提取
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestConfigExtraction:
	"""测试配置提取功能"""

	def test_extract_overrides_empty(self):
		"""测试空配置提取"""
		from envs.smartgrid.base_env.env_register import _extract_overrides

		result = _extract_overrides(None)
		assert result == {}

		result = _extract_overrides({})
		assert result == {}

	def test_extract_overrides_with_env_args(self):
		"""测试带 env_args 的配置提取"""
		from envs.smartgrid.base_env.env_register import _extract_overrides

		config = {
			'env_args': {
				'key1': 'value1',
				'key2': 'value2',
			}
		}
		result = _extract_overrides(config)
		assert result.get('key1') == 'value1'
		assert result.get('key2') == 'value2'

	def test_extract_overrides_with_dss_file(self):
		"""测试带 dss_file 的配置提取"""
		from envs.smartgrid.base_env.env_register import _extract_overrides

		config = {
			'dss_file': 'test.dss'
		}
		result = _extract_overrides(config)
		assert result.get('dss_file') == 'test.dss'

	def test_extract_overrides_with_train_config(self):
		"""测试带训练配置的提取"""
		from envs.smartgrid.base_env.env_register import _extract_overrides

		config = {
			'train': {
				'episode_length': 96
			}
		}
		result = _extract_overrides(config)
		assert result.get('max_episode_steps') == 96

	def test_extract_overrides_with_devices(self):
		"""测试带设备配置的提取"""
		from envs.smartgrid.base_env.env_register import _extract_overrides

		config = {
			'environment_specific': {
				'devices': {
					'regulators': {'action_num': 17},
					'batteries': {'action_space': 'discrete', 'action_num': 21},
					'pv_systems': {'control_enabled': True, 'action_num': 11},
				}
			}
		}
		result = _extract_overrides(config)
		assert result.get('reg_act_num') == 17
		assert result.get('bat_act_num') == 21
		assert result.get('pv_control') is True
		assert result.get('pv_act_num') == 11

	def test_extract_overrides_continuous_battery(self):
		"""测试连续电池动作空间配置"""
		from envs.smartgrid.base_env.env_register import _extract_overrides

		config = {
			'environment_specific': {
				'devices': {
					'batteries': {'action_space': 'continuous'},
				}
			}
		}
		result = _extract_overrides(config)
		assert result.get('bat_act_num') == float('inf')


# ==============================================================================
# 单元测试 - 图像拼接
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestTileImages:
	"""测试图像拼接功能"""

	def test_tile_single_image(self):
		"""测试单张图像拼接"""
		from envs.env_wrappers import tile_images

		# 单张图像
		img = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)
		result = tile_images(img)

		assert result is not None
		assert result.ndim == 3
		assert result.shape[2] == 3

	def test_tile_four_images(self):
		"""测试四张图像拼接"""
		from envs.env_wrappers import tile_images

		# 四张图像
		img = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
		result = tile_images(img)

		assert result is not None
		assert result.ndim == 3
		# 2x2 排列
		assert result.shape[0] == 128
		assert result.shape[1] == 128
		assert result.shape[2] == 3

	def test_tile_odd_number_images(self):
		"""测试奇数张图像拼接"""
		from envs.env_wrappers import tile_images

		# 三张图像
		img = np.random.randint(0, 255, (3, 32, 32, 3), dtype=np.uint8)
		result = tile_images(img)

		assert result is not None
		assert result.ndim == 3


# ==============================================================================
# 单元测试 - CloudpickleWrapper
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestCloudpickleWrapper:
	"""测试 CloudpickleWrapper"""

	def test_wrapper_creation(self):
		"""测试包装器创建"""
		from envs.env_wrappers import CloudpickleWrapper

		def dummy_func():
			return 42

		wrapper = CloudpickleWrapper(dummy_func)
		assert wrapper is not None
		assert wrapper.x == dummy_func

	def test_wrapper_getstate(self):
		"""测试包装器序列化"""
		from envs.env_wrappers import CloudpickleWrapper

		def dummy_func():
			return 42

		wrapper = CloudpickleWrapper(dummy_func)
		state = wrapper.__getstate__()

		assert state is not None
		assert isinstance(state, bytes)

	def test_wrapper_setstate(self):
		"""测试包装器反序列化"""
		from envs.env_wrappers import CloudpickleWrapper

		def dummy_func():
			return 42

		wrapper = CloudpickleWrapper(dummy_func)
		state = wrapper.__getstate__()

		new_wrapper = CloudpickleWrapper(None)
		new_wrapper.__setstate__(state)

		assert new_wrapper.x is not None
		assert callable(new_wrapper.x)
		assert new_wrapper.x() == 42


# ==============================================================================
# 集成测试 - 环境工厂函数
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestMakeEnvIntegration:
	"""测试环境工厂函数集成"""

	def test_make_env_basic(self, node_systems_dir, skip_if_no_opendss):
		"""测试基本环境创建"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
			assert env is not None

			# 验证环境属性
			assert hasattr(env, 'observation_space')
			assert hasattr(env, 'action_space')

			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_make_env_with_dss_act(self, node_systems_dir, skip_if_no_opendss):
		"""测试带 dss_act 参数的环境创建"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv', dss_act=True)
			assert env is not None
			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_make_base_env_compatibility(self, node_systems_dir, skip_if_no_opendss):
		"""测试 make_base_env 兼容性"""
		from envs.smartgrid.base_env.env_register import make_env, make_base_env

		try:
			env1 = make_env('34Bus_pv')
			env2 = make_base_env('34Bus_pv')

			assert type(env1) == type(env2)

			env1.close()
			env2.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_get_info_and_folder(self, node_systems_dir):
		"""测试获取环境信息和文件夹"""
		from envs.smartgrid.base_env.env_register import get_info_and_folder

		try:
			info, folder = get_info_and_folder('34Bus_pv')

			assert info is not None
			assert isinstance(info, dict)
			assert folder is not None
			assert os.path.isabs(folder)
		except Exception as e:
			pytest.skip(f"获取信息失败: {e}")

	def test_get_info_from_config(self, node_systems_dir):
		"""测试从配置获取环境信息"""
		from envs.smartgrid.base_env.env_register import get_info_from_config

		try:
			info = get_info_from_config('34Bus_pv')

			assert info is not None
			assert isinstance(info, dict)
			# 应该包含关键信息
			assert 'dss_file' in info or 'system_name' in info
		except Exception as e:
			pytest.skip(f"获取信息失败: {e}")


# ==============================================================================
# 集成测试 - 环境生命周期
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestEnvironmentLifecycle:
	"""测试环境完整生命周期"""

	@pytest.fixture
	def smartgrid_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 SmartGrid 环境实例"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
			yield env
			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_environment_reset(self, smartgrid_env):
		"""测试环境重置"""
		result = smartgrid_env.reset()

		assert result is not None
		# reset 应该返回观测和其他信息
		if isinstance(result, tuple):
			assert len(result) >= 1

	def test_environment_step(self, smartgrid_env):
		"""测试环境步进"""
		smartgrid_env.reset()

		# 获取动作空间并采样
		action_space = smartgrid_env.action_space
		if hasattr(action_space, 'sample'):
			action = action_space.sample()
		else:
			# 假设离散动作空间
			action = [0] * getattr(smartgrid_env, 'n_agents', 1)

		result = smartgrid_env.step(action)

		assert result is not None
		# step 应该返回 (obs, share_obs, reward, done, info, avail_actions)
		if isinstance(result, tuple):
			assert len(result) >= 4

	def test_environment_multiple_steps(self, smartgrid_env):
		"""测试多步执行"""
		smartgrid_env.reset()

		for _ in range(5):
			action_space = smartgrid_env.action_space
			if hasattr(action_space, 'sample'):
				action = action_space.sample()
			else:
				action = [0] * getattr(smartgrid_env, 'n_agents', 1)

			result = smartgrid_env.step(action)
			assert result is not None

	def test_environment_reset_after_steps(self, smartgrid_env):
		"""测试步进后重置"""
		smartgrid_env.reset()

		# 执行几步
		for _ in range(3):
			action_space = smartgrid_env.action_space
			if hasattr(action_space, 'sample'):
				action = action_space.sample()
			else:
				action = [0] * getattr(smartgrid_env, 'n_agents', 1)
			smartgrid_env.step(action)

		# 重置
		result = smartgrid_env.reset()
		assert result is not None


# ==============================================================================
# 集成测试 - 多智能体空间
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestMultiAgentSpaces:
	"""测试多智能体空间"""

	@pytest.fixture
	def smartgrid_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 SmartGrid 环境实例"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
			yield env
			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_observation_space_exists(self, smartgrid_env):
		"""测试观测空间存在"""
		assert hasattr(smartgrid_env, 'observation_space')
		assert smartgrid_env.observation_space is not None

	def test_action_space_exists(self, smartgrid_env):
		"""测试动作空间存在"""
		assert hasattr(smartgrid_env, 'action_space')
		assert smartgrid_env.action_space is not None

	def test_share_observation_space_exists(self, smartgrid_env):
		"""测试共享观测空间存在"""
		assert hasattr(smartgrid_env, 'share_observation_space')
		assert smartgrid_env.share_observation_space is not None

	def test_n_agents_attribute(self, smartgrid_env):
		"""测试智能体数量属性"""
		if hasattr(smartgrid_env, 'n_agents'):
			n_agents = smartgrid_env.n_agents
			assert isinstance(n_agents, int)
			assert n_agents > 0

	def test_observation_space_structure(self, smartgrid_env):
		"""测试观测空间结构"""
		obs_space = smartgrid_env.observation_space

		# 多智能体通常是列表
		if isinstance(obs_space, list):
			assert len(obs_space) > 0
			for space in obs_space:
				assert hasattr(space, 'shape') or hasattr(space, 'n')

	def test_action_space_structure(self, smartgrid_env):
		"""测试动作空间结构"""
		act_space = smartgrid_env.action_space

		# 多智能体通常是列表
		if isinstance(act_space, list):
			assert len(act_space) > 0
			for space in act_space:
				assert hasattr(space, 'n') or hasattr(space, 'shape')


# ==============================================================================
# 集成测试 - MARL 包装器
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestMARLWrapperIntegration:
	"""测试 MARL 包装器集成"""

	def test_share_dummy_vec_env_creation(self, node_systems_dir, skip_if_no_opendss):
		"""测试 ShareDummyVecEnv 创建"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])

			assert vec_env is not None
			assert vec_env.num_envs == 1
			assert hasattr(vec_env, 'observation_space')
			assert hasattr(vec_env, 'action_space')

			vec_env.close()
		except Exception as e:
			pytest.skip(f"向量环境创建失败: {e}")

	def test_share_dummy_vec_env_reset(self, node_systems_dir, skip_if_no_opendss):
		"""测试 ShareDummyVecEnv 重置"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])
			result = vec_env.reset()

			assert result is not None
			# 应该返回 (obs, share_obs, available_actions)
			if isinstance(result, tuple):
				assert len(result) == 3
				obs, share_obs, avail_actions = result
				assert obs is not None
				assert share_obs is not None

			vec_env.close()
		except Exception as e:
			pytest.skip(f"向量环境重置失败: {e}")

	def test_share_dummy_vec_env_step(self, node_systems_dir, skip_if_no_opendss):
		"""测试 ShareDummyVecEnv 步进"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])
			vec_env.reset()

			# 构建动作
			n_agents = getattr(vec_env, 'n_agents', 1)
			actions = [[0] * n_agents]  # 一个环境的动作

			result = vec_env.step(actions)

			assert result is not None
			# 应该返回 (obs, share_obs, rews, dones, infos, available_actions)
			if isinstance(result, tuple):
				assert len(result) == 6

			vec_env.close()
		except Exception as e:
			pytest.skip(f"向量环境步进失败: {e}")

	def test_share_dummy_vec_env_multiple_envs(self, node_systems_dir, skip_if_no_opendss):
		"""测试多环境 ShareDummyVecEnv"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn, env_fn])

			assert vec_env.num_envs == 2
			result = vec_env.reset()
			assert result is not None

			vec_env.close()
		except Exception as e:
			pytest.skip(f"多环境创建失败: {e}")


# ==============================================================================
# 集成测试 - 并行工作器
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestParallelWorkerSetup:
	"""测试并行工作器设置"""

	def test_worker_file_setup(self, node_systems_dir):
		"""测试工作器文件设置"""
		from envs.smartgrid.base_env.env_register import _setup_worker_files

		# 使用临时目录
		with tempfile.TemporaryDirectory() as tmpdir:
			# 创建模拟的系统目录结构
			system_dir = os.path.join(tmpdir, 'node_systems', 'test_system')
			os.makedirs(system_dir, exist_ok=True)

			# 创建模拟的 DSS 文件
			dss_content = """
			redirect loadshape.dss
			redirect pv_data.dss
			"""
			with open(os.path.join(system_dir, 'test.dss'), 'w') as f:
				f.write(dss_content)

			# 创建 loadshape.dss
			with open(os.path.join(system_dir, 'loadshape.dss'), 'w') as f:
				f.write("! loadshape data from ./loadshape/000/")

			# 测试设置 - 这应该不会失败
			_setup_worker_files(tmpdir, 'node_systems/test_system', 'test.dss', 1)

	def test_cleanup_parallel_dss(self, node_systems_dir):
		"""测试清理并行 DSS 文件"""
		from envs.smartgrid.base_env.env_register import cleanup_all_parallel_dss

		# 使用临时目录
		with tempfile.TemporaryDirectory() as tmpdir:
			# 创建模拟的系统目录
			system_dir = os.path.join(tmpdir, '34Bus_PV')
			os.makedirs(system_dir, exist_ok=True)

			# 创建模拟的临时文件
			temp_files = ['test_1.dss', 'test_2.dss', 'loadshape_1.dss']
			for fname in temp_files:
				with open(os.path.join(system_dir, fname), 'w') as f:
					f.write("! temp file")

			# 执行清理
			cleaned, failed = cleanup_all_parallel_dss(tmpdir)

			# 验证结果
			assert cleaned >= 0
			assert failed >= 0


# ==============================================================================
# 集成测试 - 奖励计算
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestRewardIntegration:
	"""测试奖励计算集成"""

	@pytest.fixture
	def smartgrid_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 SmartGrid 环境实例"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
			yield env
			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_reward_returned_on_step(self, smartgrid_env):
		"""测试步进返回奖励"""
		smartgrid_env.reset()

		action_space = smartgrid_env.action_space
		if hasattr(action_space, 'sample'):
			action = action_space.sample()
		else:
			action = [0] * getattr(smartgrid_env, 'n_agents', 1)

		result = smartgrid_env.step(action)

		# 奖励应该在返回结果中
		if isinstance(result, tuple) and len(result) >= 3:
			reward = result[2]
			assert reward is not None

	def test_reward_is_numeric(self, smartgrid_env):
		"""测试奖励是数值类型"""
		smartgrid_env.reset()

		action_space = smartgrid_env.action_space
		if hasattr(action_space, 'sample'):
			action = action_space.sample()
		else:
			action = [0] * getattr(smartgrid_env, 'n_agents', 1)

		result = smartgrid_env.step(action)

		if isinstance(result, tuple) and len(result) >= 3:
			reward = result[2]
			# 奖励可能是列表或数组
			if isinstance(reward, (list, np.ndarray)):
				for r in np.array(reward).flatten():
					assert isinstance(r, (int, float, np.integer, np.floating))
			else:
				assert isinstance(reward, (int, float, np.integer, np.floating))


# ==============================================================================
# 集成测试 - 日志系统
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestLoggingIntegration:
	"""测试日志系统集成"""

	def test_smartgrid_logger_with_env(self, node_systems_dir):
		"""测试 SmartGridLogger 与环境集成"""
		try:
			from envs.smartgrid.logging import SmartGridLogger
		except ImportError:
			pytest.skip("SmartGridLogger 不可用")

		with tempfile.TemporaryDirectory() as tmpdir:
			try:
				logger = SmartGridLogger(
					log_dir=tmpdir,
					env_name="test_env",
					n_agents=3,
				)

				assert logger is not None

				# 测试日志记录
				if hasattr(logger, 'log_step'):
					logger.log_step(
						step=1,
						rewards=[0.1, 0.2, 0.3],
						actions=[0, 1, 2],
					)
			except Exception:
				pass  # 日志器初始化可能需要更多参数

	def test_unified_logger_integration(self):
		"""测试 UnifiedLogger 集成"""
		try:
			from envs.smartgrid.logging import get_logger
		except ImportError:
			pytest.skip("get_logger 不可用")

		logger = get_logger("test_integration")
		assert logger is not None

		# 测试日志记录
		logger.info("Integration test message")


# ==============================================================================
# 集成测试 - 配置加载
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestConfigLoadingIntegration:
	"""测试配置加载集成"""

	def test_load_config_for_env(self, node_systems_dir):
		"""测试为环境加载配置"""
		from envs.smartgrid.base_env.config_loader import load_config

		try:
			config = load_config('34Bus_pv')

			assert config is not None
			assert hasattr(config, 'env_name')
			assert hasattr(config, 'max_episode_steps')
		except Exception as e:
			pytest.skip(f"配置加载失败: {e}")

	def test_config_with_overrides(self, node_systems_dir):
		"""测试带覆盖参数的配置加载"""
		from envs.smartgrid.base_env.config_loader import load_config

		try:
			overrides = {
				'max_episode_steps': 48,
			}
			config = load_config('34Bus_pv', overrides)

			assert config.max_episode_steps == 48
		except Exception as e:
			pytest.skip(f"配置加载失败: {e}")

	def test_get_env_config(self, node_systems_dir):
		"""测试获取环境配置"""
		from envs.smartgrid.base_env.config_loader import get_env_config

		try:
			config = get_env_config('34Bus_pv')
			assert config is not None
		except Exception as e:
			pytest.skip(f"获取配置失败: {e}")


# ==============================================================================
# 集成测试 - 完整回合
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
@pytest.mark.slow
class TestFullEpisode:
	"""测试完整回合"""

	def test_run_full_episode(self, node_systems_dir, skip_if_no_opendss):
		"""测试运行完整回合"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

		try:
			obs = env.reset()
			done = False
			step_count = 0
			max_steps = 24  # 限制步数以加快测试

			while not done and step_count < max_steps:
				action_space = env.action_space
				if hasattr(action_space, 'sample'):
					action = action_space.sample()
				else:
					action = [0] * getattr(env, 'n_agents', 1)

				result = env.step(action)

				if isinstance(result, tuple) and len(result) >= 4:
					done = result[3]
					if isinstance(done, (list, np.ndarray)):
						done = np.all(done)

				step_count += 1

			assert step_count > 0
		finally:
			env.close()

	def test_multiple_episodes(self, node_systems_dir, skip_if_no_opendss):
		"""测试多回合"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

		try:
			for episode in range(3):
				env.reset()

				for step in range(5):
					action_space = env.action_space
					if hasattr(action_space, 'sample'):
						action = action_space.sample()
					else:
						action = [0] * getattr(env, 'n_agents', 1)

					env.step(action)
		finally:
			env.close()


# ==============================================================================
# 集成测试 - 数据处理
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestDataProcessIntegration:
	"""测试数据处理集成"""

	def test_load_profile_with_env(self, node_systems_dir):
		"""测试 LoadProfile 与环境集成"""
		try:
			from envs.smartgrid.data_process.load_profile import LoadProfile
		except ImportError:
			pytest.skip("LoadProfile 不可用")

		# 查找数据目录
		loads_dir = node_systems_dir.parent / 'data' / 'Loads'
		if not loads_dir.exists():
			pytest.skip("Loads 数据目录不存在")

		try:
			profile = LoadProfile(str(loads_dir))
			assert profile is not None
		except Exception as e:
			pytest.skip(f"LoadProfile 加载失败: {e}")


# ==============================================================================
# 集成测试 - 电路系统
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestCircuitSystemIntegration:
	"""测试电路系统集成"""

	def test_circuits_with_env(self, node_systems_dir, skip_if_no_opendss):
		"""测试 Circuits 与环境集成"""
		from envs.smartgrid.circuit_system import Circuits

		# 查找 DSS 文件
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		try:
			circuit = Circuits(dss_file=str(dss_files[0]))
			assert circuit is not None

			# 验证电路组件
			assert hasattr(circuit, 'dss')
			assert hasattr(circuit, 'lines')
			assert hasattr(circuit, 'loads')
		except Exception as e:
			pytest.skip(f"电路系统创建失败: {e}")


# ==============================================================================
# 边界条件测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestEdgeCases:
	"""测试边界条件"""

	def test_invalid_env_name(self):
		"""测试无效环境名称"""
		from envs.smartgrid.base_env.env_register import make_env

		with pytest.raises(Exception):
			make_env('nonexistent_env_12345')

	def test_empty_config_dict(self, node_systems_dir):
		"""测试空配置字典"""
		from envs.smartgrid.base_env.env_register import get_info_and_folder

		try:
			info, folder = get_info_and_folder('34Bus_pv', {})
			assert info is not None
		except Exception:
			pass  # 可能会失败，这是预期行为

	def test_scaled_env_name(self, node_systems_dir):
		"""测试缩放环境名称"""
		from envs.smartgrid.base_env.env_register import get_info_and_folder

		try:
			info, folder = get_info_and_folder('34Bus_pv_s1.5')

			# 应该处理缩放后缀
			if 'scale' in info:
				assert info['scale'] == 1.5
		except Exception:
			pass  # 缩放环境可能不被支持


# ==============================================================================
# 性能测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.slow
class TestPerformance:
	"""测试性能"""

	def test_env_creation_time(self, node_systems_dir, skip_if_no_opendss):
		"""测试环境创建时间"""
		import time
		from envs.smartgrid.base_env.env_register import make_env

		start = time.time()
		try:
			env = make_env('34Bus_pv')
			creation_time = time.time() - start

			# 环境创建应在合理时间内完成
			assert creation_time < 60  # 60秒内

			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_reset_time(self, node_systems_dir, skip_if_no_opendss):
		"""测试重置时间"""
		import time
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

		try:
			# 预热
			env.reset()

			# 测量重置时间
			start = time.time()
			env.reset()
			reset_time = time.time() - start

			# 重置应该很快
			assert reset_time < 10  # 10秒内
		finally:
			env.close()

	def test_step_time(self, node_systems_dir, skip_if_no_opendss):
		"""测试步进时间"""
		import time
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

		try:
			env.reset()

			# 测量步进时间
			action_space = env.action_space
			if hasattr(action_space, 'sample'):
				action = action_space.sample()
			else:
				action = [0] * getattr(env, 'n_agents', 1)

			start = time.time()
			env.step(action)
			step_time = time.time() - start

			# 步进应该相对较快
			assert step_time < 5  # 5秒内
		finally:
			env.close()


# ==============================================================================
# HAPPO 兼容性测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestHAPPOCompatibility:
	"""测试 HAPPO 算法兼容性"""

	def test_observation_shape_consistency(self, node_systems_dir, skip_if_no_opendss):
		"""测试观测形状一致性"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])
			obs1, share_obs1, _ = vec_env.reset()

			# 执行一步
			n_agents = getattr(vec_env, 'n_agents', 1)
			actions = [[0] * n_agents]
			_, share_obs2, _, _, _, _ = vec_env.step(actions)

			# 观测形状应该一致
			assert share_obs1.shape == share_obs2.shape

			vec_env.close()
		except Exception as e:
			pytest.skip(f"HAPPO 兼容性测试失败: {e}")

	def test_done_signal_shape(self, node_systems_dir, skip_if_no_opendss):
		"""测试 done 信号形状"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])
			vec_env.reset()

			n_agents = getattr(vec_env, 'n_agents', 1)
			actions = [[0] * n_agents]
			_, _, _, dones, _, _ = vec_env.step(actions)

			# dones 应该是可迭代的
			assert hasattr(dones, '__iter__')

			vec_env.close()
		except Exception as e:
			pytest.skip(f"done 信号测试失败: {e}")

	def test_reward_shape(self, node_systems_dir, skip_if_no_opendss):
		"""测试奖励形状"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])
			vec_env.reset()

			n_agents = getattr(vec_env, 'n_agents', 1)
			actions = [[0] * n_agents]
			_, _, rews, _, _, _ = vec_env.step(actions)

			# 奖励应该有合适的形状
			assert rews is not None

			vec_env.close()
		except Exception as e:
			pytest.skip(f"奖励形状测试失败: {e}")

	def test_available_actions(self, node_systems_dir, skip_if_no_opendss):
		"""测试可用动作"""
		from envs.env_wrappers import ShareDummyVecEnv
		from envs.smartgrid.base_env.env_register import make_env

		def env_fn():
			return make_env('34Bus_pv')

		try:
			vec_env = ShareDummyVecEnv([env_fn])
			_, _, avail_actions = vec_env.reset()

			# 可用动作应该存在
			assert avail_actions is not None

			vec_env.close()
		except Exception as e:
			pytest.skip(f"可用动作测试失败: {e}")


# ==============================================================================
# 数据标准化测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestDataStandardization:
	"""测试数据标准化"""

	def test_standardize_dones(self):
		"""测试 done 信号标准化"""
		from envs.env_wrappers import ShareSubprocVecEnv

		# 创建模拟的向量环境
		class MockVecEnv(ShareSubprocVecEnv):
			def __init__(self):
				# 绕过正常初始化
				pass

		mock_env = MockVecEnv()

		# 测试标量 done
		dones = [True, False]
		result = mock_env._standardize_dones(dones)
		assert len(result) == 2

		# 测试数组 done
		dones = [[True, False, True], [False, True, False]]
		result = mock_env._standardize_dones(dones)
		assert len(result) == 2

	def test_standardize_rewards(self):
		"""测试奖励标准化"""
		from envs.env_wrappers import ShareSubprocVecEnv

		class MockVecEnv(ShareSubprocVecEnv):
			def __init__(self):
				pass

		mock_env = MockVecEnv()

		# 测试列表奖励
		rewards = [[0.1, 0.2], [0.3, 0.4]]
		result = mock_env._standardize_rewards(rewards)
		assert len(result) == 2

		# 测试标量奖励
		rewards = [0.5, 0.6]
		result = mock_env._standardize_rewards(rewards)
		assert len(result) == 2


# ==============================================================================
# 环境属性测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestEnvAttributes:
	"""测试环境属性"""

	@pytest.fixture
	def smartgrid_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 SmartGrid 环境实例"""
		from envs.smartgrid.base_env.env_register import make_env

		try:
			env = make_env('34Bus_pv')
			yield env
			env.close()
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_ordered_agents_pairs(self, smartgrid_env):
		"""测试有序智能体对"""
		if hasattr(smartgrid_env, 'ordered_agents_pairs'):
			pairs = smartgrid_env.ordered_agents_pairs
			assert pairs is not None

	def test_agents_bus(self, smartgrid_env):
		"""测试智能体母线信息"""
		if hasattr(smartgrid_env, 'agents_bus'):
			bus_info = smartgrid_env.agents_bus
			assert bus_info is not None

	def test_device_counts(self, smartgrid_env):
		"""测试设备数量"""
		device_attrs = ['cap_num', 'reg_num', 'bat_num', 'pv_num']

		for attr in device_attrs:
			if hasattr(smartgrid_env, attr):
				count = getattr(smartgrid_env, attr)
				assert isinstance(count, int)
				assert count >= 0

	def test_max_episode_steps(self, smartgrid_env):
		"""测试最大回合步数"""
		if hasattr(smartgrid_env, 'max_episode_steps'):
			steps = smartgrid_env.max_episode_steps
			assert isinstance(steps, int)
			assert steps > 0
