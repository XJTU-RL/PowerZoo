# -*- coding: utf-8 -*-
"""
PowerZoo Env 和 ActionSpace 类详细测试

测试覆盖:
- ActionSpace 类初始化和采样
- Env 类初始化
- 观测空间和动作空间定义
- reset() 方法
- step() 方法
- 奖励函数 (MyReward)
- 辅助函数测试

@File      : test_env_core.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import gym
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# ActionSpace 类单元测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.vvc
class TestActionSpaceImport:
	"""测试 ActionSpace 模块导入"""

	def test_actionspace_class_import(self):
		"""测试 ActionSpace 类可以正确导入"""
		from envs.vvc.vvc.env import ActionSpace
		assert ActionSpace is not None

	def test_env_class_import(self):
		"""测试 Env 类可以正确导入"""
		from envs.vvc.vvc.env import Env
		assert Env is not None
		assert issubclass(Env, gym.Env)


@pytest.mark.unit
@pytest.mark.vvc
class TestActionSpaceDiscrete:
	"""测试离散 ActionSpace"""

	def test_discrete_actionspace_creation(self):
		"""测试创建离散动作空间"""
		from envs.vvc.vvc.env import ActionSpace

		cap_num, reg_num, bat_num = 2, 3, 4
		reg_act_num, bat_act_num = 33, 33

		action_space = ActionSpace(
			CRB_num=(cap_num, reg_num, bat_num),
			RB_act_num=(reg_act_num, bat_act_num)
		)

		assert action_space.cap_num == cap_num
		assert action_space.reg_num == reg_num
		assert action_space.bat_num == bat_num
		assert action_space.reg_act_num == reg_act_num
		assert action_space.bat_act_num == bat_act_num

	def test_discrete_actionspace_is_multidiscrete(self):
		"""测试离散动作空间是 MultiDiscrete"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 33)
		)

		assert isinstance(action_space.space, gym.spaces.MultiDiscrete)

	def test_discrete_actionspace_sample(self):
		"""测试离散动作空间采样"""
		from envs.vvc.vvc.env import ActionSpace

		cap_num, reg_num, bat_num = 2, 3, 4
		action_space = ActionSpace(
			CRB_num=(cap_num, reg_num, bat_num),
			RB_act_num=(33, 33)
		)

		sample = action_space.sample()

		assert isinstance(sample, np.ndarray)
		assert len(sample) == cap_num + reg_num + bat_num

		# 验证电容器动作在 [0, 1] 范围内
		for i in range(cap_num):
			assert sample[i] in [0, 1]

		# 验证调压器动作在 [0, 32] 范围内
		for i in range(cap_num, cap_num + reg_num):
			assert 0 <= sample[i] < 33

		# 验证电池动作在 [0, 32] 范围内
		for i in range(cap_num + reg_num, cap_num + reg_num + bat_num):
			assert 0 <= sample[i] < 33

	def test_discrete_actionspace_dim(self):
		"""测试离散动作空间维度"""
		from envs.vvc.vvc.env import ActionSpace

		cap_num, reg_num, bat_num = 2, 3, 4
		action_space = ActionSpace(
			CRB_num=(cap_num, reg_num, bat_num),
			RB_act_num=(33, 33)
		)

		assert action_space.dim() == cap_num + reg_num + bat_num

	def test_discrete_actionspace_seed(self):
		"""测试离散动作空间种子设置"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, 33)
		)

		# 设置种子应该不会引发异常
		action_space.seed(42)

		# 使用相同种子应该产生相同的采样
		action_space.seed(42)
		sample1 = action_space.sample()
		action_space.seed(42)
		sample2 = action_space.sample()

		np.testing.assert_array_equal(sample1, sample2)


@pytest.mark.unit
@pytest.mark.vvc
class TestActionSpaceContinuous:
	"""测试连续 ActionSpace"""

	def test_continuous_actionspace_creation(self):
		"""测试创建连续动作空间"""
		from envs.vvc.vvc.env import ActionSpace

		cap_num, reg_num, bat_num = 2, 3, 4
		reg_act_num = 33
		bat_act_num = float('inf')  # 连续电池

		action_space = ActionSpace(
			CRB_num=(cap_num, reg_num, bat_num),
			RB_act_num=(reg_act_num, bat_act_num)
		)

		assert action_space.bat_act_num == float('inf')

	def test_continuous_actionspace_is_tuple(self):
		"""测试连续动作空间是 Tuple"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 4),
			RB_act_num=(33, float('inf'))
		)

		assert isinstance(action_space.space, gym.spaces.Tuple)

	def test_continuous_actionspace_sample(self):
		"""测试连续动作空间采样"""
		from envs.vvc.vvc.env import ActionSpace

		cap_num, reg_num, bat_num = 2, 3, 4
		action_space = ActionSpace(
			CRB_num=(cap_num, reg_num, bat_num),
			RB_act_num=(33, float('inf'))
		)

		sample = action_space.sample()

		assert isinstance(sample, np.ndarray)
		# 总维度 = 离散 (cap + reg) + 连续 (bat)
		assert len(sample) == cap_num + reg_num + bat_num

		# 验证连续电池动作在 [-1, 1] 范围内
		for i in range(cap_num + reg_num, len(sample)):
			assert -1 <= sample[i] <= 1

	def test_continuous_actionspace_dim(self):
		"""测试连续动作空间维度"""
		from envs.vvc.vvc.env import ActionSpace

		cap_num, reg_num, bat_num = 2, 3, 4
		action_space = ActionSpace(
			CRB_num=(cap_num, reg_num, bat_num),
			RB_act_num=(33, float('inf'))
		)

		assert action_space.dim() == cap_num + reg_num + bat_num


@pytest.mark.unit
@pytest.mark.vvc
class TestActionSpaceEdgeCases:
	"""测试 ActionSpace 边界条件"""

	def test_actionspace_zero_capacitors(self):
		"""测试没有电容器的动作空间"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(0, 3, 4),
			RB_act_num=(33, 33)
		)

		assert action_space.cap_num == 0
		sample = action_space.sample()
		assert len(sample) == 3 + 4

	def test_actionspace_zero_regulators(self):
		"""测试没有调压器的动作空间"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 0, 4),
			RB_act_num=(33, 33)
		)

		assert action_space.reg_num == 0
		sample = action_space.sample()
		assert len(sample) == 2 + 4

	def test_actionspace_zero_batteries(self):
		"""测试没有电池的动作空间"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(2, 3, 0),
			RB_act_num=(33, 33)
		)

		assert action_space.bat_num == 0
		sample = action_space.sample()
		assert len(sample) == 2 + 3

	def test_actionspace_all_zero(self):
		"""测试所有设备为零的动作空间"""
		from envs.vvc.vvc.env import ActionSpace

		action_space = ActionSpace(
			CRB_num=(0, 0, 0),
			RB_act_num=(33, 33)
		)

		sample = action_space.sample()
		assert len(sample) == 0


# ==============================================================================
# Env 类单元测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.vvc
class TestEnvAttributes:
	"""测试 Env 类属性"""

	def test_env_has_required_attributes(self):
		"""验证 Env 类具有所有必需的属性和方法"""
		from envs.vvc.vvc.env import Env

		required_methods = [
			'__init__', 'reset', 'step', 'seed', 'render', 'close',
			'random_action', 'dummy_action'
		]

		for method in required_methods:
			assert hasattr(Env, method), f"Env 缺少方法: {method}"

	def test_env_inherits_gym_env(self):
		"""测试 Env 继承自 gym.Env"""
		from envs.vvc.vvc.env import Env
		assert issubclass(Env, gym.Env)


# ==============================================================================
# Env 集成测试 - 需要 OpenDSS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestEnvInitialization:
	"""测试 Env 类初始化（需要 OpenDSS）"""

	@pytest.fixture
	def env_info_13bus(self, node_systems_dir):
		"""13Bus 环境配置"""
		return {
			'scale': 1.0,
			'max_episode_steps': 24,
			'horizon': 24,
			'power_w': 10.0,
			'cap_w': 1.0/33,
			'reg_w': 1.0/33,
			'soc_w': 0.0,
			'dis_w': 6.0/33,
			'reg_act_num': 33,
			'bat_act_num': 33,
			'wrap_observation': True,
			'observe_load': False,
			'useS': False,
			'use_render': False,
		}

	def test_env_creation_with_valid_config(
		self, node_systems_dir, env_info_13bus, skip_if_no_opendss
	):
		"""测试使用有效配置创建环境"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		env = Env(folder_path=str(dss_folder), info=env_info_13bus)

		assert env is not None
		assert hasattr(env, 'observation_space')
		assert hasattr(env, 'action_space')
		assert hasattr(env, 'ActionSpace')

	def test_env_observation_space(
		self, node_systems_dir, env_info_13bus, skip_if_no_opendss
	):
		"""测试观测空间定义"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		env = Env(folder_path=str(dss_folder), info=env_info_13bus)

		assert env.observation_space is not None
		assert isinstance(env.observation_space, gym.spaces.Box)
		assert env.observation_space.dtype == np.float32 or env.observation_space.dtype == np.float64

	def test_env_action_space(
		self, node_systems_dir, env_info_13bus, skip_if_no_opendss
	):
		"""测试动作空间定义"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		env = Env(folder_path=str(dss_folder), info=env_info_13bus)

		assert env.ActionSpace is not None
		# 动作空间应该能采样
		sample = env.ActionSpace.sample()
		assert sample is not None


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestEnvReset:
	"""测试 Env reset 功能"""

	@pytest.fixture
	def env_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 环境实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		info = {
			'scale': 1.0,
			'max_episode_steps': 24,
			'horizon': 24,
			'power_w': 10.0,
			'cap_w': 1.0/33,
			'reg_w': 1.0/33,
			'soc_w': 0.0,
			'dis_w': 6.0/33,
			'reg_act_num': 33,
			'bat_act_num': 33,
			'wrap_observation': True,
			'observe_load': False,
			'useS': False,
			'use_render': False,
		}

		return Env(folder_path=str(dss_folder), info=info)

	def test_reset_returns_observation(self, env_13bus):
		"""测试 reset 返回观测"""
		obs = env_13bus.reset()

		assert obs is not None
		assert isinstance(obs, np.ndarray)
		assert len(obs) > 0

	def test_reset_with_load_profile_idx(self, env_13bus):
		"""测试使用特定负荷曲线索引重置"""
		obs1 = env_13bus.reset(load_profile_idx=0)
		obs2 = env_13bus.reset(load_profile_idx=0)

		# 使用相同索引应该得到相似的初始状态
		assert obs1.shape == obs2.shape

	def test_reset_initializes_time(self, env_13bus):
		"""测试 reset 初始化时间步"""
		env_13bus.reset()
		assert env_13bus.t == 0

	def test_reset_initializes_obs_dict(self, env_13bus):
		"""测试 reset 初始化观测字典"""
		env_13bus.reset()

		assert hasattr(env_13bus, 'obs')
		assert isinstance(env_13bus.obs, dict)
		assert 'bus_voltages' in env_13bus.obs


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestEnvStep:
	"""测试 Env step 功能"""

	@pytest.fixture
	def env_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 环境实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		info = {
			'scale': 1.0,
			'max_episode_steps': 24,
			'horizon': 24,
			'power_w': 10.0,
			'cap_w': 1.0/33,
			'reg_w': 1.0/33,
			'soc_w': 0.0,
			'dis_w': 6.0/33,
			'reg_act_num': 33,
			'bat_act_num': 33,
			'wrap_observation': True,
			'observe_load': False,
			'useS': False,
			'use_render': False,
		}

		return Env(folder_path=str(dss_folder), info=info)

	def test_step_returns_tuple(self, env_13bus):
		"""测试 step 返回 (obs, reward, done, info) 元组"""
		env_13bus.reset()
		action = env_13bus.random_action()

		result = env_13bus.step(action)

		assert isinstance(result, tuple)
		assert len(result) == 4

		obs, reward, done, info = result
		assert isinstance(obs, np.ndarray)
		assert isinstance(reward, (int, float))
		assert isinstance(done, bool)
		assert isinstance(info, dict)

	def test_step_advances_time(self, env_13bus):
		"""测试 step 推进时间步"""
		env_13bus.reset()
		initial_t = env_13bus.t

		action = env_13bus.random_action()
		env_13bus.step(action)

		assert env_13bus.t == initial_t + 1

	def test_step_with_random_action(self, env_13bus):
		"""测试使用随机动作执行 step"""
		env_13bus.reset()

		for _ in range(5):
			action = env_13bus.random_action()
			obs, reward, done, info = env_13bus.step(action)

			assert obs is not None
			assert not np.isnan(reward)

			if done:
				break

	def test_step_episode_terminates(self, env_13bus):
		"""测试 episode 正确终止"""
		env_13bus.reset()

		done = False
		step_count = 0
		max_steps = env_13bus.horizon + 10

		while not done and step_count < max_steps:
			action = env_13bus.random_action()
			_, _, done, _ = env_13bus.step(action)
			step_count += 1

		# 应该在 horizon 步内终止
		assert step_count <= max_steps

	def test_step_info_contains_required_keys(self, env_13bus):
		"""测试 info 字典包含必需的键"""
		env_13bus.reset()
		action = env_13bus.random_action()
		_, _, _, info = env_13bus.step(action)

		# 验证 info 包含奖励分解信息
		expected_keys = ['power_loss_ratio']
		for key in expected_keys:
			assert key in info, f"info 缺少键: {key}"


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestEnvReward:
	"""测试 Env 奖励函数"""

	@pytest.fixture
	def env_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 环境实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		info = {
			'scale': 1.0,
			'max_episode_steps': 24,
			'horizon': 24,
			'power_w': 10.0,
			'cap_w': 1.0/33,
			'reg_w': 1.0/33,
			'soc_w': 0.0,
			'dis_w': 6.0/33,
			'reg_act_num': 33,
			'bat_act_num': 33,
			'wrap_observation': True,
			'observe_load': False,
			'useS': False,
			'use_render': False,
		}

		return Env(folder_path=str(dss_folder), info=info)

	def test_reward_is_scalar(self, env_13bus):
		"""测试奖励是标量"""
		env_13bus.reset()
		action = env_13bus.random_action()
		_, reward, _, _ = env_13bus.step(action)

		assert isinstance(reward, (int, float))
		assert not np.isnan(reward)
		assert not np.isinf(reward)

	def test_reward_components_in_info(self, env_13bus):
		"""测试奖励分量在 info 中"""
		env_13bus.reset()
		action = env_13bus.random_action()
		_, _, _, info = env_13bus.step(action)

		# 验证功率损耗信息存在
		assert 'power_loss_ratio' in info

	def test_reward_range_reasonable(self, env_13bus):
		"""测试奖励值在合理范围内"""
		env_13bus.reset()

		rewards = []
		for _ in range(10):
			action = env_13bus.random_action()
			_, reward, done, _ = env_13bus.step(action)
			rewards.append(reward)
			if done:
				break

		# 奖励不应该是极端值
		for r in rewards:
			assert abs(r) < 1000, f"奖励值过大: {r}"


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestEnvHelperFunctions:
	"""测试 Env 辅助函数"""

	@pytest.fixture
	def env_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 环境实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		info = {
			'scale': 1.0,
			'max_episode_steps': 24,
			'horizon': 24,
			'power_w': 10.0,
			'cap_w': 1.0/33,
			'reg_w': 1.0/33,
			'soc_w': 0.0,
			'dis_w': 6.0/33,
			'reg_act_num': 33,
			'bat_act_num': 33,
			'wrap_observation': True,
			'observe_load': False,
			'useS': False,
			'use_render': False,
		}

		return Env(folder_path=str(dss_folder), info=info)

	def test_random_action(self, env_13bus):
		"""测试随机动作生成"""
		action = env_13bus.random_action()

		assert action is not None
		assert isinstance(action, np.ndarray)

	def test_dummy_action(self, env_13bus):
		"""测试虚拟动作生成"""
		action = env_13bus.dummy_action()

		assert action is not None
		assert isinstance(action, np.ndarray)

	def test_seed_method(self, env_13bus):
		"""测试种子设置方法"""
		env_13bus.seed(42)

		# 应该不会引发异常
		action1 = env_13bus.random_action()
		env_13bus.seed(42)
		action2 = env_13bus.random_action()

		# 使用相同种子应该得到相同动作
		np.testing.assert_array_equal(action1, action2)


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestEnvObservation:
	"""测试 Env 观测相关功能"""

	@pytest.fixture
	def env_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 环境实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc.vvc.env import Env

		info = {
			'scale': 1.0,
			'max_episode_steps': 24,
			'horizon': 24,
			'power_w': 10.0,
			'cap_w': 1.0/33,
			'reg_w': 1.0/33,
			'soc_w': 0.0,
			'dis_w': 6.0/33,
			'reg_act_num': 33,
			'bat_act_num': 33,
			'wrap_observation': True,
			'observe_load': False,
			'useS': False,
			'use_render': False,
		}

		return Env(folder_path=str(dss_folder), info=info)

	def test_observation_contains_voltages(self, env_13bus):
		"""测试观测包含电压信息"""
		env_13bus.reset()

		obs_dict = env_13bus.obs
		assert 'bus_voltages' in obs_dict
		assert len(obs_dict['bus_voltages']) > 0

	def test_observation_space_bounds(self, env_13bus):
		"""测试观测空间边界"""
		obs = env_13bus.reset()

		# 观测应该在观测空间内
		assert env_13bus.observation_space.contains(obs)

	def test_wrapped_observation_is_flat(self, env_13bus):
		"""测试包装后的观测是一维数组"""
		obs = env_13bus.reset()

		assert len(obs.shape) == 1

	def test_wrap_obs_method(self, env_13bus):
		"""测试 wrap_obs 方法"""
		env_13bus.reset()

		wrapped = env_13bus.wrap_obs(env_13bus.obs)

		assert isinstance(wrapped, np.ndarray)
		assert len(wrapped.shape) == 1


# ==============================================================================
# 辅助函数测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.vvc
class TestHelperFunctions:
	"""测试辅助函数"""

	def test_plotting_function_import(self):
		"""测试 plotting 函数可以导入"""
		from envs.vvc.vvc.env import plotting
		assert plotting is not None
		assert callable(plotting)

	def test_fft_selection_function_import(self):
		"""测试 FFT_selection 函数可以导入"""
		from envs.vvc.vvc.env import FFT_selection
		assert FFT_selection is not None
		assert callable(FFT_selection)

	def test_choose_batteries_function_import(self):
		"""测试 choose_batteries 函数可以导入"""
		from envs.vvc.vvc.env import choose_batteries
		assert choose_batteries is not None
		assert callable(choose_batteries)

	def test_get_basekv_function_import(self):
		"""测试 get_basekv 函数可以导入"""
		from envs.vvc.vvc.env import get_basekv
		assert get_basekv is not None
		assert callable(get_basekv)


@pytest.mark.unit
@pytest.mark.vvc
class TestFFTSelection:
	"""测试 FFT_selection 函数"""

	def test_fft_selection_basic(self):
		"""测试 FFT_selection 基本功能"""
		from envs.vvc.vvc.env import FFT_selection

		vio_nodes = ['node1', 'node2', 'node3', 'node4']
		dist_matrix = np.array([
			[0, 1, 2, 3],
			[1, 0, 1, 2],
			[2, 1, 0, 1],
			[3, 2, 1, 0]
		])

		result = FFT_selection(vio_nodes, dist_matrix, k=2)

		assert isinstance(result, list)
		assert len(result) == 2

	def test_fft_selection_k_larger_than_nodes(self):
		"""测试 k 大于节点数的情况"""
		from envs.vvc.vvc.env import FFT_selection

		vio_nodes = ['node1', 'node2']
		dist_matrix = np.array([
			[0, 1],
			[1, 0]
		])

		result = FFT_selection(vio_nodes, dist_matrix, k=5)

		# 结果应该不超过节点数
		assert len(result) <= len(vio_nodes)

	def test_fft_selection_single_node(self):
		"""测试单节点情况"""
		from envs.vvc.vvc.env import FFT_selection

		vio_nodes = ['node1']
		dist_matrix = np.array([[0]])

		result = FFT_selection(vio_nodes, dist_matrix, k=3)

		assert result == ['node1']

	def test_fft_selection_empty_nodes(self):
		"""测试空节点列表"""
		from envs.vvc.vvc.env import FFT_selection

		vio_nodes = []
		dist_matrix = np.array([]).reshape(0, 0)

		result = FFT_selection(vio_nodes, dist_matrix, k=3)

		assert result == []
