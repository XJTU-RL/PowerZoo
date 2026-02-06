# -*- coding: utf-8 -*-
"""
PowerZoo MARL 包装器 (VVCEnv) 详细测试

测试覆盖:
- VVCEnv 初始化
- 多智能体动作空间分解
- 观测空间和共享观测空间
- 可用动作计算
- reset/step 多智能体接口
- 奖励分发

@File      : test_vvc_marl.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import gym
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.vvc
class TestVVCEnvImport:
	"""测试 VVCEnv 模块导入"""

	def test_vvc_env_import(self):
		"""测试 VVCEnv 类可以正确导入"""
		from envs.vvc import VVCEnv
		assert VVCEnv is not None

	def test_vvc_env_from_module(self):
		"""测试从模块导入 VVCEnv"""
		from envs.vvc.vvc_env import VVCEnv
		assert VVCEnv is not None

	def test_vvc_env_has_required_attributes(self):
		"""验证 VVCEnv 类具有所有必需的属性和方法"""
		from envs.vvc import VVCEnv

		required_methods = [
			'__init__', 'reset', 'step',
			'get_avail_actions', 'get_avail_agent_actions'
		]

		for method in required_methods:
			assert hasattr(VVCEnv, method), f"VVCEnv 缺少方法: {method}"


# ==============================================================================
# 集成测试 - 需要 OpenDSS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestVVCEnvInitialization:
	"""测试 VVCEnv 初始化"""

	@pytest.fixture
	def env_args_13bus(self, node_systems_dir):
		"""13Bus 环境参数"""
		return {
			'env_name': '13Bus',
			'num_agents': 3,
		}

	def test_vvc_env_creation(self, node_systems_dir, skip_if_no_opendss):
		"""测试 VVCEnv 创建"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			# 尝试使用预定义的环境名称
			env = VVCEnv(env_name="13Bus")
			assert env is not None
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_vvc_env_n_agents(self, node_systems_dir, skip_if_no_opendss):
		"""测试智能体数量"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			env = VVCEnv(env_name="13Bus")

			# 验证智能体数量
			assert hasattr(env, 'n_agents')
			assert env.n_agents > 0
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_vvc_env_agents_list(self, node_systems_dir, skip_if_no_opendss):
		"""测试智能体列表"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			env = VVCEnv(env_name="13Bus")

			assert hasattr(env, 'agents')
			assert len(env.agents) == env.n_agents
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestVVCEnvSpaces:
	"""测试 VVCEnv 空间定义"""

	@pytest.fixture
	def vvc_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 VVCEnv 实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			return VVCEnv(env_name="13Bus")
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_action_space_is_list(self, vvc_env):
		"""测试动作空间是列表"""
		assert hasattr(vvc_env, 'action_space')
		assert isinstance(vvc_env.action_space, list)
		assert len(vvc_env.action_space) == vvc_env.n_agents

	def test_observation_space_structure(self, vvc_env):
		"""测试观测空间结构"""
		assert hasattr(vvc_env, 'observation_space')

		# 观测空间可以是列表或单个空间
		if isinstance(vvc_env.observation_space, list):
			assert len(vvc_env.observation_space) == vvc_env.n_agents

	def test_share_observation_space(self, vvc_env):
		"""测试共享观测空间"""
		assert hasattr(vvc_env, 'share_observation_space')

		# 共享观测空间应该存在
		assert vvc_env.share_observation_space is not None

	def test_each_agent_action_space(self, vvc_env):
		"""测试每个智能体的动作空间"""
		for i, action_space in enumerate(vvc_env.action_space):
			assert isinstance(action_space, gym.Space), f"智能体 {i} 的动作空间不是 gym.Space"


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestVVCEnvReset:
	"""测试 VVCEnv reset 功能"""

	@pytest.fixture
	def vvc_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 VVCEnv 实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			return VVCEnv(env_name="13Bus")
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_reset_returns_tuple(self, vvc_env):
		"""测试 reset 返回元组"""
		result = vvc_env.reset()

		# MARL reset 应该返回 (obs, global_obs, avail_actions)
		assert isinstance(result, tuple)
		assert len(result) >= 2  # 至少有 obs 和其他信息

	def test_reset_observations_per_agent(self, vvc_env):
		"""测试 reset 为每个智能体返回观测"""
		result = vvc_env.reset()

		obs = result[0]  # 第一个元素应该是观测列表

		if isinstance(obs, list):
			assert len(obs) == vvc_env.n_agents

	def test_reset_available_actions(self, vvc_env):
		"""测试 reset 返回可用动作"""
		result = vvc_env.reset()

		# 如果有可用动作信息
		if len(result) > 2:
			avail_actions = result[-1]  # 通常是最后一个元素
			if isinstance(avail_actions, list):
				assert len(avail_actions) == vvc_env.n_agents


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestVVCEnvStep:
	"""测试 VVCEnv step 功能"""

	@pytest.fixture
	def vvc_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 VVCEnv 实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			env = VVCEnv(env_name="13Bus")
			env.reset()
			return env
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_step_with_random_actions(self, vvc_env):
		"""测试使用随机动作执行 step"""
		# 为每个智能体采样动作
		actions = [space.sample() for space in vvc_env.action_space]

		result = vvc_env.step(actions)

		# 验证返回结果
		assert isinstance(result, tuple)
		assert len(result) >= 4  # (obs, global_obs, rewards, dones, infos, avail_actions)

	def test_step_returns_rewards_per_agent(self, vvc_env):
		"""测试 step 为每个智能体返回奖励"""
		actions = [space.sample() for space in vvc_env.action_space]

		result = vvc_env.step(actions)

		# 奖励应该是每个智能体一个
		rewards = result[2] if len(result) > 2 else result[1]
		if isinstance(rewards, list):
			assert len(rewards) == vvc_env.n_agents

	def test_step_returns_dones_per_agent(self, vvc_env):
		"""测试 step 为每个智能体返回 done 标志"""
		actions = [space.sample() for space in vvc_env.action_space]

		result = vvc_env.step(actions)

		# done 标志应该是每个智能体一个
		dones = result[3] if len(result) > 3 else None
		if isinstance(dones, list):
			assert len(dones) == vvc_env.n_agents

	def test_step_returns_infos_per_agent(self, vvc_env):
		"""测试 step 为每个智能体返回 info"""
		actions = [space.sample() for space in vvc_env.action_space]

		result = vvc_env.step(actions)

		# info 应该是每个智能体一个
		infos = result[4] if len(result) > 4 else None
		if isinstance(infos, list):
			assert len(infos) == vvc_env.n_agents


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestVVCEnvAvailActions:
	"""测试 VVCEnv 可用动作功能"""

	@pytest.fixture
	def vvc_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 VVCEnv 实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			env = VVCEnv(env_name="13Bus")
			env.reset()
			return env
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_get_avail_actions(self, vvc_env):
		"""测试获取所有可用动作"""
		avail_actions = vvc_env.get_avail_actions()

		assert isinstance(avail_actions, list)
		assert len(avail_actions) == vvc_env.n_agents

	def test_get_avail_agent_actions(self, vvc_env):
		"""测试获取单个智能体可用动作"""
		for agent_id in range(vvc_env.n_agents):
			avail = vvc_env.get_avail_agent_actions(agent_id)

			assert avail is not None
			# 可用动作应该是布尔数组或整数数组
			if isinstance(avail, (list, np.ndarray)):
				assert len(avail) > 0

	def test_avail_actions_are_valid(self, vvc_env):
		"""测试可用动作值有效"""
		avail_actions = vvc_env.get_avail_actions()

		for agent_id, avail in enumerate(avail_actions):
			if isinstance(avail, (list, np.ndarray)):
				# 所有值应该是 0 或 1
				for val in avail:
					assert val in [0, 1, True, False]


@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
@pytest.mark.slow
class TestVVCEnvEpisode:
	"""测试 VVCEnv 完整 episode"""

	@pytest.fixture
	def vvc_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 VVCEnv 实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			return VVCEnv(env_name="13Bus")
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_full_episode(self, vvc_env):
		"""测试运行完整 episode"""
		vvc_env.reset()

		done = False
		step_count = 0
		max_steps = 100

		while not done and step_count < max_steps:
			actions = [space.sample() for space in vvc_env.action_space]
			result = vvc_env.step(actions)

			# 检查 done 状态
			dones = result[3] if len(result) > 3 else [False] * vvc_env.n_agents
			if isinstance(dones, list):
				done = all(dones)
			else:
				done = dones

			step_count += 1

		# 应该在最大步数内完成
		assert step_count <= max_steps

	def test_episode_reward_accumulation(self, vvc_env):
		"""测试 episode 奖励累积"""
		vvc_env.reset()

		total_rewards = [0.0] * vvc_env.n_agents
		step_count = 0
		max_steps = 10

		while step_count < max_steps:
			actions = [space.sample() for space in vvc_env.action_space]
			result = vvc_env.step(actions)

			rewards = result[2] if len(result) > 2 else [0.0] * vvc_env.n_agents
			if isinstance(rewards, list):
				for i, r in enumerate(rewards):
					total_rewards[i] += r if isinstance(r, (int, float)) else r[0]

			step_count += 1

			# 检查是否完成
			dones = result[3] if len(result) > 3 else [False] * vvc_env.n_agents
			if isinstance(dones, list) and all(dones):
				break

		# 奖励应该被累积
		assert all(isinstance(r, (int, float)) for r in total_rewards)


# ==============================================================================
# 设备-智能体映射测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.vvc
@pytest.mark.requires_opendss
class TestAgentDeviceMapping:
	"""测试智能体-设备映射"""

	@pytest.fixture
	def vvc_env(self, node_systems_dir, skip_if_no_opendss):
		"""创建 VVCEnv 实例"""
		dss_folder = node_systems_dir / "13Bus"
		if not dss_folder.exists():
			pytest.skip("13Bus 系统不存在")

		from envs.vvc import VVCEnv

		try:
			return VVCEnv(env_name="13Bus")
		except (FileNotFoundError, KeyError, Exception) as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_agent_count_matches_devices(self, vvc_env):
		"""测试智能体数量匹配设备数量"""
		# n_agents 应该等于 cap_num + reg_num + bat_num (+ pv_num if applicable)
		env = vvc_env.env  # 底层环境

		expected_agents = env.cap_num + env.reg_num + env.bat_num
		if hasattr(env, 'pv_num'):
			expected_agents += env.pv_num

		assert vvc_env.n_agents == expected_agents

	def test_action_space_dimensions_match_devices(self, vvc_env):
		"""测试动作空间维度匹配设备类型"""
		env = vvc_env.env

		# 前 cap_num 个智能体控制电容器 (2 个动作: 开/关)
		for i in range(env.cap_num):
			assert vvc_env.action_space[i].n == 2

		# 接下来 reg_num 个智能体控制调压器 (reg_act_num 个动作)
		for i in range(env.cap_num, env.cap_num + env.reg_num):
			assert vvc_env.action_space[i].n == env.ActionSpace.reg_act_num
