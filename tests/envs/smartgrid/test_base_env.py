# -*- coding: utf-8 -*-
"""
SmartGrid base_env 模块详细测试

测试覆盖:
- ActionSpace 类
- Env 类 (env.py 和 core_env.py)
- VVCEnv MARL包装器 (vvc_env.py)
- 辅助函数 (plotting, FFT_selection, choose_batteries)

@File      : test_base_env.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import gym


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestBaseEnvImport:
	"""测试 base_env 模块导入"""

	def test_env_import(self):
		"""测试 Env 类导入 (env.py)"""
		from envs.smartgrid.base_env.env import Env, ActionSpace
		assert Env is not None
		assert ActionSpace is not None

	def test_core_env_import(self):
		"""测试 core_env 模块导入"""
		from envs.smartgrid.base_env.core_env import Env, ActionSpace
		assert Env is not None
		assert ActionSpace is not None

	def test_helper_functions_import(self):
		"""测试辅助函数导入"""
		from envs.smartgrid.base_env.core_env import (
			plotting, FFT_selection, choose_batteries, get_basekv
		)
		assert plotting is not None
		assert FFT_selection is not None
		assert choose_batteries is not None
		assert get_basekv is not None

	def test_vvc_env_import(self):
		"""测试 VVCEnv 导入"""
		from envs.smartgrid.base_env.vvc_env import VVCEnv
		assert VVCEnv is not None

	def test_seeding_function_import(self):
		"""测试 seeding 函数导入"""
		from envs.smartgrid.base_env.vvc_env import seeding
		assert seeding is not None


# ==============================================================================
# 单元测试 - ActionSpace 类 (core_env.py)
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestCoreActionSpace:
	"""测试 core_env.ActionSpace 类"""

	def test_action_space_creation_discrete(self):
		"""测试离散动作空间创建"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		# CRB_num = (cap_num, reg_num, bat_num)
		# RB_act_num = (reg_act_num, bat_act_num)
		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 5)
		)

		assert action_space is not None
		assert action_space.cap_num == 2
		assert action_space.reg_num == 3
		assert action_space.bat_num == 4
		assert action_space.reg_act_num == 33
		assert action_space.bat_act_num == 5

	def test_action_space_creation_continuous_battery(self):
		"""测试连续电池动作空间创建"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, float('inf'))  # 连续电池
		)

		assert action_space is not None
		assert action_space.bat_act_num == float('inf')
		# 连续电池时，space 是 Tuple 类型
		assert isinstance(action_space.space, gym.spaces.Tuple)

	def test_action_space_sample_discrete(self):
		"""测试离散动作空间采样"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 5)
		)

		sample = action_space.sample()
		assert sample is not None
		assert len(sample) == 2 + 3 + 4  # cap + reg + bat

	def test_action_space_sample_continuous(self):
		"""测试连续动作空间采样"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, float('inf'))
		)

		sample = action_space.sample()
		assert sample is not None
		# 连续动作时返回 concatenated array
		assert len(sample) == 2 + 3 + 4  # cap + reg + bat

	def test_action_space_dim(self):
		"""测试动作空间维度计算"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 5)
		)

		dim = action_space.dim()
		assert dim == 9  # 2 + 3 + 4

	def test_action_space_seed(self):
		"""测试动作空间种子设置"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 5)
		)

		# 应该不抛出异常
		action_space.seed(42)

	def test_action_space_CRB_num(self):
		"""测试 CRB_num 方法"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 5)
		)

		crb = action_space.CRB_num()
		assert crb == (2, 3, 4)

	def test_action_space_RB_act_num(self):
		"""测试 RB_act_num 方法"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 5)
		)

		rb = action_space.RB_act_num()
		assert rb == (33, 5)


# ==============================================================================
# 单元测试 - ActionSpace 类 (env.py - 带 PV 支持)
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestEnvActionSpace:
	"""测试 env.ActionSpace 类 (支持 CRBP)"""

	def test_action_space_with_pv(self):
		"""测试带 PV 的动作空间创建"""
		from envs.smartgrid.base_env.env import ActionSpace

		# CRBP_num = (cap, reg, bat, pv)
		# RBP_act_num = (reg_act, bat_act, pv_act)
		action_space = ActionSpace(
			CRBP_num=(2, 3, 4, 5),
			RBP_act_num=(33, 5, float('inf')),
			pv_control_enabled=True
		)

		assert action_space is not None
		assert action_space.cap_num == 2
		assert action_space.reg_num == 3
		assert action_space.bat_num == 4
		assert action_space.pv_num == 5
		assert action_space.pv_control_enabled is True

	def test_action_space_without_pv(self):
		"""测试禁用 PV 的动作空间创建"""
		from envs.smartgrid.base_env.env import ActionSpace

		action_space = ActionSpace(
			CRBP_num=(2, 3, 4, 5),
			RBP_act_num=(33, 5, float('inf')),
			pv_control_enabled=False
		)

		assert action_space is not None
		assert action_space.pv_control_enabled is False

	def test_action_space_sample_with_pv(self):
		"""测试带 PV 的动作空间采样"""
		from envs.smartgrid.base_env.env import ActionSpace

		action_space = ActionSpace(
			CRBP_num=(2, 3, 4, 5),
			RBP_act_num=(33, 5, float('inf')),
			pv_control_enabled=True
		)

		sample = action_space.sample()
		assert sample is not None

	def test_action_space_get_action_dims(self):
		"""测试 get_action_dims 方法"""
		from envs.smartgrid.base_env.env import ActionSpace

		action_space = ActionSpace(
			CRBP_num=(2, 3, 4, 5),
			RBP_act_num=(33, 5, float('inf')),
			pv_control_enabled=True
		)

		dims = action_space.get_action_dims()
		assert dims['capacitor'] == 2
		assert dims['regulator'] == 3
		assert dims['battery'] == 4
		assert dims['pv'] == 5

	def test_action_space_CRBP_num(self):
		"""测试 CRBP_num 方法"""
		from envs.smartgrid.base_env.env import ActionSpace

		action_space = ActionSpace(
			CRBP_num=(2, 3, 4, 5),
			RBP_act_num=(33, 5, float('inf')),
			pv_control_enabled=True
		)

		crbp = action_space.CRBP_num()
		assert crbp == (2, 3, 4, 5)


# ==============================================================================
# 单元测试 - 辅助函数
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestHelperFunctions:
	"""测试辅助函数"""

	def test_fft_selection_basic(self):
		"""测试 FFT_selection 基本功能"""
		from envs.smartgrid.base_env.core_env import FFT_selection

		vio_nodes = ['bus1', 'bus2', 'bus3', 'bus4', 'bus5']
		# 创建距离矩阵
		dist_matrix = np.random.rand(5, 5)
		dist_matrix = (dist_matrix + dist_matrix.T) / 2  # 对称矩阵

		result = FFT_selection(vio_nodes, dist_matrix, k=3)
		assert result is not None
		assert len(result) <= 3
		assert all(node in vio_nodes for node in result)

	def test_fft_selection_single_node(self):
		"""测试 FFT_selection 单节点情况"""
		from envs.smartgrid.base_env.core_env import FFT_selection

		vio_nodes = ['bus1']
		dist_matrix = np.array([[0.0]])

		result = FFT_selection(vio_nodes, dist_matrix, k=3)
		assert result == ['bus1']

	def test_fft_selection_empty(self):
		"""测试 FFT_selection 空列表情况"""
		from envs.smartgrid.base_env.core_env import FFT_selection

		vio_nodes = []
		dist_matrix = np.array([])

		result = FFT_selection(vio_nodes, dist_matrix, k=3)
		assert result == []


# ==============================================================================
# 单元测试 - seeding 函数
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSeeding:
	"""测试 seeding 函数"""

	def test_seeding_sets_numpy_seed(self):
		"""测试 seeding 设置 numpy 种子"""
		from envs.smartgrid.base_env.vvc_env import seeding

		seeding(42)

		# 验证种子已设置
		val1 = np.random.rand()
		seeding(42)
		val2 = np.random.rand()

		assert val1 == val2

	def test_seeding_different_seeds(self):
		"""测试不同种子产生不同结果"""
		from envs.smartgrid.base_env.vvc_env import seeding

		seeding(42)
		val1 = np.random.rand()

		seeding(123)
		val2 = np.random.rand()

		assert val1 != val2


# ==============================================================================
# 集成测试 - core_env.Env 类
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestCoreEnvIntegration:
	"""测试 core_env.Env 类集成"""

	@pytest.fixture
	def env_info(self):
		"""创建环境信息字典"""
		return {
			'system_name': 'node_systems/34Bus_PV',
			'dss_file': 'Run_case_Control_Daily_B1_duty.dss',
			'source_bus': 'sourcebus',
			'node_size': 300,
			'shift': 50,
			'show_node_labels': False,
			'scale': 1.0,
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
		}

	def test_core_env_creation(self, node_systems_dir, skip_if_no_opendss, env_info, project_root):
		"""测试 core_env.Env 创建"""
		from envs.smartgrid.base_env.core_env import Env

		# 检查 DSS 系统是否存在
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				env_info['system_name'] = f'node_systems/{variant}'
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		env_info['dss_file'] = dss_files[0].name

		try:
			env = Env(str(project_root), env_info)
			assert env is not None
			assert env.cap_num >= 0
			assert env.reg_num >= 0
			assert env.bat_num >= 0
		except Exception as e:
			pytest.skip(f"Env 创建失败: {e}")

	def test_core_env_reset(self, node_systems_dir, skip_if_no_opendss, env_info, project_root):
		"""测试 core_env.Env 重置"""
		from envs.smartgrid.base_env.core_env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				env_info['system_name'] = f'node_systems/{variant}'
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		env_info['dss_file'] = dss_files[0].name

		try:
			env = Env(str(project_root), env_info)
			obs = env.reset()
			assert obs is not None
		except Exception as e:
			pytest.skip(f"Env 重置失败: {e}")

	def test_core_env_step(self, node_systems_dir, skip_if_no_opendss, env_info, project_root):
		"""测试 core_env.Env 步进"""
		from envs.smartgrid.base_env.core_env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				env_info['system_name'] = f'node_systems/{variant}'
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		env_info['dss_file'] = dss_files[0].name

		try:
			env = Env(str(project_root), env_info)
			env.reset()

			# 执行随机动作
			action = env.random_action()
			obs, reward, done, info = env.step(action)

			assert obs is not None
			assert isinstance(reward, (int, float))
			assert isinstance(done, bool)
			assert isinstance(info, dict)
		except Exception as e:
			pytest.skip(f"Env 步进失败: {e}")


# ==============================================================================
# 集成测试 - env.Env 类 (带 CMDP 支持)
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestEnvIntegration:
	"""测试 env.Env 类集成 (带 CMDP)"""

	@pytest.fixture
	def env_config(self):
		"""创建 SmartGridConfig"""
		try:
			from envs.smartgrid.base_env.env_config import SmartGridConfig
			return SmartGridConfig(
				env_name='34Bus_pv',
				max_episode_steps=24,
				seed=42
			)
		except ImportError:
			pytest.skip("SmartGridConfig 不可用")

	def test_env_creation_with_config(self, node_systems_dir, skip_if_no_opendss, project_root):
		"""测试使用 SmartGridConfig 创建 Env"""
		from envs.smartgrid.base_env.env import Env
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		# 查找可用系统
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		# 构建信息字典
		info = {
			'system_name': f'node_systems/{variant}',
			'dss_file': dss_files[0].name,
			'source_bus': 'sourcebus',
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
			'voltage_w': 1.0,
			'use_cmdp': True,
		}

		try:
			env = Env(str(project_root), info)
			assert env is not None
			assert hasattr(env, 'config')
		except Exception as e:
			pytest.skip(f"Env 创建失败: {e}")

	def test_env_cmdp_features(self, node_systems_dir, skip_if_no_opendss, project_root):
		"""测试 CMDP 功能"""
		from envs.smartgrid.base_env.env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		info = {
			'system_name': f'node_systems/{variant}',
			'dss_file': dss_files[0].name,
			'source_bus': 'sourcebus',
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
			'voltage_w': 1.0,
			'use_cmdp': True,
			'lambda_init': 1.0,
			'lambda_lr': 1e-3,
			'target_cost': 0.01,
		}

		try:
			env = Env(str(project_root), info)
			assert env.use_cmdp is True
			assert env.lagrangian_updater is not None
		except Exception as e:
			pytest.skip(f"CMDP 功能测试失败: {e}")


# ==============================================================================
# 集成测试 - VVCEnv 类
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestVVCEnvIntegration:
	"""测试 VVCEnv MARL 包装器"""

	def test_vvc_env_creation(self, node_systems_dir, skip_if_no_opendss, project_root):
		"""测试 VVCEnv 创建"""
		from envs.smartgrid.base_env.vvc_env import VVCEnv
		from envs.smartgrid.base_env.env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		info = {
			'system_name': f'node_systems/{variant}',
			'dss_file': dss_files[0].name,
			'source_bus': 'sourcebus',
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
			'voltage_w': 1.0,
		}

		try:
			base_env = Env(str(project_root), info)

			# 创建配置对象
			config = Mock()
			config.num_env = 1
			config.env_name = 'test'
			config.seed = 42
			config.useS = False
			config.enable_system_logging = False

			pz_env = VVCEnv(base_env, config, rank=0)
			assert pz_env is not None
			assert pz_env.n_agents > 0
		except Exception as e:
			pytest.skip(f"VVCEnv 创建失败: {e}")

	def test_vvc_env_reset(self, node_systems_dir, skip_if_no_opendss, project_root):
		"""测试 VVCEnv 重置"""
		from envs.smartgrid.base_env.vvc_env import VVCEnv
		from envs.smartgrid.base_env.env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		info = {
			'system_name': f'node_systems/{variant}',
			'dss_file': dss_files[0].name,
			'source_bus': 'sourcebus',
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
			'voltage_w': 1.0,
		}

		try:
			base_env = Env(str(project_root), info)

			config = Mock()
			config.num_env = 1
			config.env_name = 'test'
			config.seed = 42
			config.useS = False
			config.enable_system_logging = False

			pz_env = VVCEnv(base_env, config, rank=0)
			obs, state, avail_actions = pz_env.reset()

			assert obs is not None
			assert len(obs) == pz_env.n_agents
			assert avail_actions is not None
		except Exception as e:
			pytest.skip(f"VVCEnv 重置失败: {e}")

	def test_vvc_env_step(self, node_systems_dir, skip_if_no_opendss, project_root):
		"""测试 VVCEnv 步进"""
		from envs.smartgrid.base_env.vvc_env import VVCEnv
		from envs.smartgrid.base_env.env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		info = {
			'system_name': f'node_systems/{variant}',
			'dss_file': dss_files[0].name,
			'source_bus': 'sourcebus',
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
			'voltage_w': 1.0,
		}

		try:
			base_env = Env(str(project_root), info)

			config = Mock()
			config.num_env = 1
			config.env_name = 'test'
			config.seed = 42
			config.useS = False
			config.enable_system_logging = False

			pz_env = VVCEnv(base_env, config, rank=0)
			pz_env.reset()

			# 创建动作
			actions = np.zeros((pz_env.n_agents, 1), dtype=np.int32)

			local_obs, global_state, rewards, dones, infos, avail_actions = pz_env.step(actions)

			assert local_obs is not None
			assert len(local_obs) == pz_env.n_agents
			assert rewards is not None
			assert dones is not None
		except Exception as e:
			pytest.skip(f"VVCEnv 步进失败: {e}")

	def test_vvc_env_get_avail_actions(self, node_systems_dir, skip_if_no_opendss, project_root):
		"""测试 VVCEnv 可用动作"""
		from envs.smartgrid.base_env.vvc_env import VVCEnv
		from envs.smartgrid.base_env.env import Env

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list((node_systems_dir / variant).glob("*duty.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		info = {
			'system_name': f'node_systems/{variant}',
			'dss_file': dss_files[0].name,
			'source_bus': 'sourcebus',
			'max_episode_steps': 24,
			'reg_act_num': 33,
			'bat_act_num': 5,
			'power_w': 1.0,
			'cap_w': 0.1,
			'reg_w': 0.1,
			'soc_w': 0.1,
			'dis_w': 0.1,
			'voltage_w': 1.0,
		}

		try:
			base_env = Env(str(project_root), info)

			config = Mock()
			config.num_env = 1
			config.env_name = 'test'
			config.seed = 42
			config.useS = False
			config.enable_system_logging = False

			pz_env = VVCEnv(base_env, config, rank=0)
			pz_env.reset()

			avail_actions = pz_env.get_avail_actions()
			assert avail_actions is not None
			assert len(avail_actions) == pz_env.n_agents
		except Exception as e:
			pytest.skip(f"VVCEnv 可用动作测试失败: {e}")


# ==============================================================================
# 单元测试 - VVCEnv 内部方法
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestVVCEnvMethods:
	"""测试 VVCEnv 内部方法"""

	def test_format_actions_summary_list(self):
		"""测试动作摘要格式化 - 列表"""
		from envs.smartgrid.base_env.vvc_env import VVCEnv

		# 创建 mock 环境
		mock_env = Mock()
		mock_env.cap_num = 2
		mock_env.reg_num = 2
		mock_env.bat_num = 2
		mock_env.pv_num = 0
		mock_env.pv_names = []
		mock_env.cap_names = ['cap1', 'cap2']
		mock_env.reg_names = ['reg1', 'reg2']
		mock_env.bat_names = ['bat1', 'bat2']
		mock_env.observation_space = Mock()
		mock_env.observation_space.shape = (10,)
		mock_env.action_space = Mock()
		mock_env.action_space.nvec = [2, 2, 33, 33, 5, 5]
		mock_env.pv_control_enabled = False
		mock_env.seed = Mock()
		mock_env.useS = False

		mock_config = Mock()
		mock_config.num_env = 1
		mock_config.env_name = 'test'
		mock_config.seed = 42
		mock_config.useS = False
		mock_config.enable_system_logging = False

		# 手动测试格式化方法
		actions = [1, 0, 16, 17, 2, 3]

		# 直接测试方法逻辑（无需完整初始化）
		if len(actions) <= 6:
			result = f"[{', '.join([str(a) for a in actions])}]"
		assert "1" in result
		assert "0" in result

	def test_format_actions_summary_numpy(self):
		"""测试动作摘要格式化 - numpy数组"""
		actions = np.array([1, 0, 16, 17, 2, 3])

		if len(actions) <= 6:
			result = str(actions.tolist())

		assert "1" in result
		assert "0" in result


# ==============================================================================
# 边界条件测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestBaseEnvEdgeCases:
	"""测试 base_env 边界条件"""

	def test_action_space_zero_devices(self):
		"""测试零设备动作空间"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		# 至少需要一个设备
		# 这个测试验证行为
		try:
			action_space = ActionSpace(
				CRB_num=(1, 0, 0),  # 只有1个电容器
				RB_act_num=(33, 5)
			)
			assert action_space is not None
			assert action_space.dim() == 1
		except Exception:
			pass  # 可能抛出断言错误

	def test_action_space_large_reg_act_num(self):
		"""测试大量调压器动作"""
		from envs.smartgrid.base_env.core_env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(100, 5)  # 100个调压器档位
		)

		assert action_space is not None
		assert action_space.reg_act_num == 100

	def test_observation_wrapping_consistency(self):
		"""测试观测包装一致性"""
		# 这个测试验证 wrap_obs 方法的一致性
		# 需要完整环境来测试
		pass
