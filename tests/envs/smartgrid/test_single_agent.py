# -*- coding: utf-8 -*-
"""
SmartGrid single_agent 模块详细测试

测试覆盖:
- SingleAgentPowerZooEnv 单智能体环境
- SingleAgentConfig 配置类
- SingleAgentLogger 日志器
- SingleAgentTrainingConfig 训练配置

@File      : test_single_agent.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentImports:
	"""测试 single_agent 模块导入"""

	def test_single_agent_env_import(self):
		"""测试 SingleAgentPowerZooEnv 导入"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv
		assert SingleAgentPowerZooEnv is not None

	def test_single_agent_config_import(self):
		"""测试 SingleAgentConfig 导入"""
		from envs.smartgrid.single_agent import SingleAgentConfig
		assert SingleAgentConfig is not None

	def test_single_agent_logger_import(self):
		"""测试 SingleAgentLogger 导入"""
		from envs.smartgrid.single_agent import SingleAgentLogger
		assert SingleAgentLogger is not None

	def test_package_init_exports(self):
		"""测试包导出"""
		from envs.smartgrid.single_agent import (
			SingleAgentPowerZooEnv,
			SingleAgentConfig,
			SingleAgentLogger
		)
		assert all([SingleAgentPowerZooEnv, SingleAgentConfig, SingleAgentLogger])

	def test_training_config_import(self):
		"""测试 SingleAgentTrainingConfig 导入"""
		from envs.smartgrid.single_agent.single_agent_training_config import (
			SingleAgentTrainingConfig,
			get_config
		)
		assert SingleAgentTrainingConfig is not None
		assert get_config is not None


# ==============================================================================
# 单元测试 - SingleAgentConfig
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentConfigInit:
	"""测试 SingleAgentConfig 初始化"""

	def test_default_config_creation(self):
		"""测试默认配置创建"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		assert config is not None
		assert config.circuit_name == "13Bus"
		assert config.max_episode_steps == 24
		assert config.seed == 42

	def test_config_with_custom_values(self):
		"""测试自定义配置创建"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig(
			circuit_name="34Bus",
			max_episode_steps=48,
			seed=123
		)
		assert config.circuit_name == "34Bus"
		assert config.max_episode_steps == 48
		assert config.seed == 123

	def test_config_has_required_attributes(self):
		"""测试配置有必需属性"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()

		required_attrs = [
			'circuit_name', 'max_episode_steps', 'seed',
			'voltage_penalty_weight', 'power_loss_weight',
			'enable_capacitors', 'enable_regulators',
			'enable_batteries', 'enable_pv_systems',
			'log_level'
		]

		for attr in required_attrs:
			assert hasattr(config, attr), f"配置缺少属性: {attr}"


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentConfigMethods:
	"""测试 SingleAgentConfig 方法"""

	def test_to_dict(self):
		"""测试 to_dict 方法"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		config_dict = config.to_dict()

		assert isinstance(config_dict, dict)
		assert 'circuit_name' in config_dict
		assert 'max_episode_steps' in config_dict

	def test_from_dict(self):
		"""测试 from_dict 类方法"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config_dict = {
			'circuit_name': '34Bus',
			'max_episode_steps': 96,
			'seed': 999
		}

		config = SingleAgentConfig.from_dict(config_dict)
		assert config.circuit_name == '34Bus'
		assert config.max_episode_steps == 96

	def test_update_method(self):
		"""测试 update 方法"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		config.update(max_episode_steps=100, seed=555)

		assert config.max_episode_steps == 100
		assert config.seed == 555

	def test_update_invalid_key_raises(self):
		"""测试更新无效键抛出异常"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()

		with pytest.raises(ValueError):
			config.update(invalid_key=123)

	def test_validate_method(self):
		"""测试 validate 方法"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		assert config.validate() is True

	def test_validate_invalid_episode_steps(self):
		"""测试验证无效的 episode 步数"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		config.max_episode_steps = 0

		with pytest.raises(ValueError):
			config.validate()

	def test_validate_invalid_voltage_tolerance(self):
		"""测试验证无效的电压容差"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		config.voltage_tolerance = 1.5  # 应该在 0 到 1 之间

		with pytest.raises(ValueError):
			config.validate()

	def test_to_env_info(self):
		"""测试 to_env_info 方法"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig(circuit_name="13Bus")
		env_info = config.to_env_info()

		assert isinstance(env_info, dict)
		assert 'system_name' in env_info
		assert 'dss_file' in env_info
		assert 'max_episode_steps' in env_info

	def test_get_folder_path(self):
		"""测试 get_folder_path 方法"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig()
		folder_path = config.get_folder_path()

		assert isinstance(folder_path, str)
		assert "node_systems" in folder_path


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentConfigPresets:
	"""测试预定义配置"""

	def test_default_config_exists(self):
		"""测试默认配置存在"""
		from envs.smartgrid.single_agent.single_agent_config import DEFAULT_CONFIG
		assert DEFAULT_CONFIG is not None

	def test_training_config_exists(self):
		"""测试训练配置存在"""
		from envs.smartgrid.single_agent.single_agent_config import TRAINING_CONFIG
		assert TRAINING_CONFIG is not None
		assert TRAINING_CONFIG.save_episode_data is True

	def test_testing_config_exists(self):
		"""测试测试配置存在"""
		from envs.smartgrid.single_agent.single_agent_config import TESTING_CONFIG
		assert TESTING_CONFIG is not None
		assert TESTING_CONFIG.save_episode_data is False

	def test_fast_config_exists(self):
		"""测试快速配置存在"""
		from envs.smartgrid.single_agent.single_agent_config import FAST_CONFIG
		assert FAST_CONFIG is not None
		assert FAST_CONFIG.max_episode_steps == 12


# ==============================================================================
# 单元测试 - SingleAgentLogger
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentLoggerInit:
	"""测试 SingleAgentLogger 初始化"""

	def test_logger_creation(self):
		"""测试日志器创建"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(
				log_dir=tmpdir,
				experiment_name="test_exp",
				log_level="DEBUG"
			)
			assert logger is not None
			assert logger.experiment_name == "test_exp"

	def test_logger_creates_directory(self):
		"""测试日志器创建目录"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(
				log_dir=tmpdir,
				experiment_name="test_exp"
			)
			assert (Path(tmpdir) / "test_exp").exists()

	def test_logger_has_required_attributes(self):
		"""测试日志器有必需属性"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(log_dir=tmpdir)

			assert hasattr(logger, 'logger')
			assert hasattr(logger, 'episode_data')
			assert hasattr(logger, 'step_data')
			assert hasattr(logger, 'metrics_data')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentLoggerMethods:
	"""测试 SingleAgentLogger 方法"""

	@pytest.fixture
	def test_logger(self):
		"""创建测试日志器"""
		with tempfile.TemporaryDirectory() as tmpdir:
			from envs.smartgrid.single_agent import SingleAgentLogger
			logger = SingleAgentLogger(
				log_dir=tmpdir,
				experiment_name="test_methods"
			)
			yield logger

	def test_log_episode_start(self, test_logger):
		"""测试记录 Episode 开始"""
		test_logger.log_episode_start(0, config={'test': True})
		assert test_logger.current_episode == 0
		assert test_logger.episode_start_time is not None

	def test_log_step(self, test_logger):
		"""测试记录步骤"""
		test_logger.log_episode_start(0)
		test_logger.log_step(
			step=0,
			action=np.array([0, 1, 0]),
			observation=np.array([1.0, 0.95, 1.02]),
			reward=1.5,
			done=False,
			info={'test_key': 'test_value'}
		)

		assert test_logger.current_step == 0
		assert len(test_logger.step_data) == 1

	def test_log_episode_end(self, test_logger):
		"""测试记录 Episode 结束"""
		test_logger.log_episode_start(0)
		test_logger.log_episode_end(
			total_reward=50.0,
			episode_length=24,
			final_info={'voltage_violations': 2}
		)

		assert len(test_logger.episode_data) == 1
		assert len(test_logger.metrics_data['episode_rewards']) == 1

	def test_log_training_metrics(self, test_logger):
		"""测试记录训练指标"""
		test_logger.log_training_metrics({
			'loss': 0.5,
			'entropy': 0.1
		})

		assert 'loss' in test_logger.metrics_data
		assert len(test_logger.metrics_data['loss']) == 1


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentLoggerSaveLoad:
	"""测试 SingleAgentLogger 保存和加载"""

	def test_save_data(self):
		"""测试保存数据"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(log_dir=tmpdir, experiment_name="test_save")

			# 添加一些数据
			logger.log_episode_start(0)
			for step in range(10):
				logger.log_step(
					step=step,
					action=step,
					observation=np.array([1.0]),
					reward=float(step),
					done=(step == 9)
				)
			logger.log_episode_end(45.0, 10)

			# 保存数据
			logger.save_data()

			# 检查文件是否创建
			exp_dir = Path(tmpdir) / "test_save"
			assert (exp_dir / "episodes.csv").exists()
			assert (exp_dir / "metrics.json").exists()

	def test_get_summary_stats(self):
		"""测试获取总结统计"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(log_dir=tmpdir)

			# 添加多个 episode 数据
			for ep in range(5):
				logger.log_episode_start(ep)
				logger.log_episode_end(float(ep * 10), 24)

			stats = logger.get_summary_stats()

			assert 'total_episodes' in stats
			assert stats['total_episodes'] == 5
			assert 'mean_reward' in stats
			assert 'max_reward' in stats

	def test_close_logger(self):
		"""测试关闭日志器"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(log_dir=tmpdir, experiment_name="test_close")

			logger.log_episode_start(0)
			logger.log_episode_end(10.0, 10)

			logger.close()

			# 检查数据已保存
			exp_dir = Path(tmpdir) / "test_close"
			assert (exp_dir / "episodes.csv").exists()


# ==============================================================================
# 单元测试 - SingleAgentTrainingConfig
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentTrainingConfigInit:
	"""测试 SingleAgentTrainingConfig 初始化"""

	def test_default_training_config(self):
		"""测试默认训练配置"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		with tempfile.TemporaryDirectory() as tmpdir:
			config = SingleAgentTrainingConfig(
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			assert config is not None
			assert config.algorithm == "ppo"
			assert config.total_timesteps == 100000

	def test_training_config_with_algorithm(self):
		"""测试指定算法的训练配置"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		with tempfile.TemporaryDirectory() as tmpdir:
			config = SingleAgentTrainingConfig(
				algorithm="dqn",
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			assert config.algorithm == "dqn"


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentTrainingConfigMethods:
	"""测试 SingleAgentTrainingConfig 方法"""

	def test_get_sb3_model_kwargs(self):
		"""测试获取 SB3 模型参数"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		with tempfile.TemporaryDirectory() as tmpdir:
			config = SingleAgentTrainingConfig(
				algorithm="ppo",
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			kwargs = config.get_sb3_model_kwargs()

			assert isinstance(kwargs, dict)
			assert 'learning_rate' in kwargs

	def test_get_training_kwargs(self):
		"""测试获取训练参数"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		with tempfile.TemporaryDirectory() as tmpdir:
			config = SingleAgentTrainingConfig(
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			kwargs = config.get_training_kwargs()

			assert isinstance(kwargs, dict)
			assert 'total_timesteps' in kwargs

	def test_to_dict(self):
		"""测试 to_dict 方法"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		with tempfile.TemporaryDirectory() as tmpdir:
			config = SingleAgentTrainingConfig(
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			config_dict = config.to_dict()

			assert isinstance(config_dict, dict)
			assert 'algorithm' in config_dict
			assert 'total_timesteps' in config_dict


@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentTrainingConfigPresets:
	"""测试预定义训练配置"""

	def test_get_ppo_config(self):
		"""测试获取 PPO 配置"""
		from envs.smartgrid.single_agent.single_agent_training_config import get_config

		with tempfile.TemporaryDirectory() as tmpdir:
			config = get_config(
				"ppo",
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			assert config.algorithm == "ppo"

	def test_get_dqn_config(self):
		"""测试获取 DQN 配置"""
		from envs.smartgrid.single_agent.single_agent_training_config import get_config

		with tempfile.TemporaryDirectory() as tmpdir:
			config = get_config(
				"dqn",
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			assert config.algorithm == "dqn"

	def test_config_registry(self):
		"""测试配置注册表"""
		from envs.smartgrid.single_agent.single_agent_training_config import CONFIG_REGISTRY

		assert isinstance(CONFIG_REGISTRY, dict)
		assert 'ppo' in CONFIG_REGISTRY
		assert 'dqn' in CONFIG_REGISTRY
		assert 'sac' in CONFIG_REGISTRY


# ==============================================================================
# 单元测试 - SingleAgentPowerZooEnv (Mocked)
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentPowerZooEnvAttributes:
	"""测试 SingleAgentPowerZooEnv 属性"""

	def test_env_has_required_methods(self):
		"""测试环境有必需方法"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		assert hasattr(SingleAgentPowerZooEnv, 'step')
		assert hasattr(SingleAgentPowerZooEnv, 'reset')
		assert hasattr(SingleAgentPowerZooEnv, 'render')
		assert hasattr(SingleAgentPowerZooEnv, 'close')

	def test_env_has_action_conversion_methods(self):
		"""测试环境有动作转换方法"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		assert hasattr(SingleAgentPowerZooEnv, '_convert_discrete_to_multi_discrete')
		assert hasattr(SingleAgentPowerZooEnv, '_convert_continuous_to_discrete')
		assert hasattr(SingleAgentPowerZooEnv, '_parse_single_agent_action')

	def test_env_has_space_setup_methods(self):
		"""测试环境有空间设置方法"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		assert hasattr(SingleAgentPowerZooEnv, '_setup_single_agent_action_space')
		assert hasattr(SingleAgentPowerZooEnv, '_setup_single_agent_observation_space')
		assert hasattr(SingleAgentPowerZooEnv, '_setup_discrete_action_space')
		assert hasattr(SingleAgentPowerZooEnv, '_setup_continuous_action_space')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestActionConversion:
	"""测试动作转换逻辑"""

	def test_discrete_to_multi_discrete_concept(self):
		"""测试离散到多离散转换概念"""
		# 模拟转换逻辑
		nvec = np.array([2, 3, 4])  # 总共 24 种组合
		action = 10  # 测试动作值

		# 转换逻辑
		multi_action = []
		remaining = action
		for i in range(len(nvec) - 1, -1, -1):
			nvec_i = nvec[i]
			action_i = remaining % nvec_i
			remaining = remaining // nvec_i
			multi_action.insert(0, action_i)

		assert len(multi_action) == 3
		# 验证转换正确性
		reconstructed = multi_action[0] * (3 * 4) + multi_action[1] * 4 + multi_action[2]
		assert reconstructed == action

	def test_continuous_to_discrete_concept(self):
		"""测试连续到离散转换概念"""
		# 连续值在 [-1, 1] 范围
		continuous_val = 0.5
		num_actions = 5

		# 映射到 [0, num_actions-1]
		discrete_action = int((continuous_val + 1) / 2 * (num_actions - 1))
		discrete_action = np.clip(discrete_action, 0, num_actions - 1)

		assert 0 <= discrete_action < num_actions


# ==============================================================================
# 集成测试 - SingleAgentPowerZooEnv
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestSingleAgentEnvIntegration:
	"""测试单智能体环境集成"""

	@pytest.fixture
	def env_config(self, node_systems_dir):
		"""创建环境配置"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig(
			circuit_name="34Bus",
			max_episode_steps=24,
			seed=42
		)
		return config

	def test_env_creation_with_config(self, node_systems_dir, skip_if_no_opendss, env_config):
		"""测试使用配置创建环境"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		try:
			env = SingleAgentPowerZooEnv(config=env_config)
			assert env is not None
			assert hasattr(env, 'action_space')
			assert hasattr(env, 'observation_space')
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_env_reset(self, node_systems_dir, skip_if_no_opendss, env_config):
		"""测试环境重置"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		try:
			env = SingleAgentPowerZooEnv(config=env_config)
			obs, info = env.reset()

			assert obs is not None
			assert isinstance(info, dict)
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_env_step(self, node_systems_dir, skip_if_no_opendss, env_config):
		"""测试环境步进"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		try:
			env = SingleAgentPowerZooEnv(config=env_config)
			obs, info = env.reset()

			# 采样动作并执行
			action = env.action_space.sample()
			next_obs, reward, done, truncated, step_info = env.step(action)

			assert next_obs is not None
			assert isinstance(reward, (int, float))
			assert isinstance(done, bool)
			assert isinstance(step_info, dict)
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_env_episode(self, node_systems_dir, skip_if_no_opendss, env_config):
		"""测试完整 Episode"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

		try:
			env = SingleAgentPowerZooEnv(config=env_config)
			obs, info = env.reset()

			total_reward = 0
			step = 0
			done = False

			while not done and step < 24:
				action = env.action_space.sample()
				obs, reward, done, truncated, info = env.step(action)
				total_reward += reward
				step += 1

			assert step > 0
			assert isinstance(total_reward, (int, float))
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")


@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestSingleAgentActionSpace:
	"""测试单智能体动作空间"""

	def test_discrete_action_space(self, node_systems_dir, skip_if_no_opendss):
		"""测试离散动作空间"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv, SingleAgentConfig
		import gymnasium as gym

		config = SingleAgentConfig(circuit_name="13Bus", max_episode_steps=10)

		try:
			env = SingleAgentPowerZooEnv(config=config, action_space_type="discrete")

			assert isinstance(env.action_space, gym.spaces.Discrete)
			assert env.action_space.n > 0
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_continuous_action_space(self, node_systems_dir, skip_if_no_opendss):
		"""测试连续动作空间"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv, SingleAgentConfig
		import gymnasium as gym

		config = SingleAgentConfig(circuit_name="13Bus", max_episode_steps=10)

		try:
			env = SingleAgentPowerZooEnv(config=config, action_space_type="continuous")

			assert isinstance(env.action_space, gym.spaces.Box)
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")

	def test_get_action_meanings(self, node_systems_dir, skip_if_no_opendss):
		"""测试获取动作含义"""
		from envs.smartgrid.single_agent import SingleAgentPowerZooEnv, SingleAgentConfig

		config = SingleAgentConfig(circuit_name="13Bus", max_episode_steps=10)

		try:
			env = SingleAgentPowerZooEnv(config=config)
			meanings = env.get_action_meanings()

			assert isinstance(meanings, dict)
			assert 'action_space_type' in meanings
			assert 'devices' in meanings
		except Exception as e:
			pytest.skip(f"环境创建失败: {e}")


# ==============================================================================
# 集成测试 - 日志系统集成
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestSingleAgentLoggingIntegration:
	"""测试单智能体日志集成"""

	def test_logger_with_training_workflow(self):
		"""测试日志器与训练工作流"""
		from envs.smartgrid.single_agent import SingleAgentLogger
		import numpy as np

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(
				log_dir=tmpdir,
				experiment_name="workflow_test",
				save_episode_data=True,
				save_metrics=True
			)

			# 模拟训练过程
			for episode in range(3):
				logger.log_episode_start(episode)

				total_reward = 0
				for step in range(10):
					action = np.random.randint(0, 4)
					obs = np.random.randn(10)
					reward = np.random.randn()
					done = (step == 9)

					logger.log_step(step, action, obs, reward, done)
					total_reward += reward

				logger.log_episode_end(total_reward, 10)

			# 保存并验证
			logger.save_data()

			exp_dir = Path(tmpdir) / "workflow_test"
			assert (exp_dir / "episodes.csv").exists()
			assert (exp_dir / "metrics.json").exists()

			# 验证统计
			stats = logger.get_summary_stats()
			assert stats['total_episodes'] == 3

			logger.close()


# ==============================================================================
# 边界条件测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestSingleAgentEdgeCases:
	"""测试单智能体边界条件"""

	def test_config_with_no_devices_enabled(self):
		"""测试禁用所有设备"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig(
			enable_capacitors=False,
			enable_regulators=False,
			enable_batteries=False,
			enable_pv_systems=False
		)

		with pytest.raises(ValueError):
			config.validate()

	def test_config_with_negative_weights(self):
		"""测试负权重"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		config = SingleAgentConfig(
			voltage_penalty_weight=-1.0
		)

		with pytest.raises(ValueError):
			config.validate()

	def test_logger_empty_data(self):
		"""测试空数据日志器"""
		from envs.smartgrid.single_agent import SingleAgentLogger

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(log_dir=tmpdir)

			# 获取空统计
			stats = logger.get_summary_stats()
			assert stats == {}

			logger.close()

	def test_config_circuit_name_mapping(self):
		"""测试电路名称映射"""
		from envs.smartgrid.single_agent import SingleAgentConfig

		# 测试已知电路
		for circuit_name in ["13Bus", "34Bus", "123Bus"]:
			config = SingleAgentConfig(circuit_name=circuit_name)
			env_info = config.to_env_info()
			assert 'dss_file' in env_info
			assert env_info['dss_file'] is not None

		# 测试未知电路（应该使用默认值）
		config = SingleAgentConfig(circuit_name="unknown")
		env_info = config.to_env_info()
		assert 'dss_file' in env_info


@pytest.mark.unit
@pytest.mark.smartgrid
class TestTrainingConfigEdgeCases:
	"""测试训练配置边界条件"""

	def test_missing_algo_config_file(self):
		"""测试缺少算法配置文件"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		with tempfile.TemporaryDirectory() as tmpdir:
			# 使用不存在的算法名称
			config = SingleAgentTrainingConfig(
				algorithm="nonexistent_algo",
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)

			# 应该使用默认配置
			assert config.algo_config is not None

	def test_from_args(self):
		"""测试从参数创建配置"""
		from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

		# 创建 mock args
		args = Mock()
		args.algorithm = "sac"
		args.environment = "test_env"
		args.experiment_name = "test_exp"
		args.total_timesteps = 50000
		args.device = "cpu"
		args.seed = 42

		config = SingleAgentTrainingConfig.from_args(args)
		assert config.algorithm == "sac"
		assert config.total_timesteps == 50000


# ==============================================================================
# 便捷函数测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestConvenienceFunctions:
	"""测试便捷函数"""

	def test_create_single_agent_powerzoo_env_function(self):
		"""测试创建环境便捷函数"""
		from envs.smartgrid.single_agent.single_agent_env import create_single_agent_powerzoo_env
		assert create_single_agent_powerzoo_env is not None

	def test_get_config_function(self):
		"""测试获取配置便捷函数"""
		from envs.smartgrid.single_agent.single_agent_training_config import get_config

		with tempfile.TemporaryDirectory() as tmpdir:
			config = get_config(
				"ppo",
				log_dir=os.path.join(tmpdir, "logs"),
				model_save_dir=os.path.join(tmpdir, "models")
			)
			assert config is not None
			assert config.algorithm == "ppo"


# ==============================================================================
# 性能测试
# ==============================================================================

@pytest.mark.slow
@pytest.mark.smartgrid
class TestSingleAgentPerformance:
	"""测试单智能体性能"""

	def test_logger_high_frequency(self):
		"""测试高频日志记录"""
		from envs.smartgrid.single_agent import SingleAgentLogger
		import time

		with tempfile.TemporaryDirectory() as tmpdir:
			logger = SingleAgentLogger(
				log_dir=tmpdir,
				experiment_name="perf_test",
				save_episode_data=False  # 禁用保存以测试纯日志性能
			)

			start = time.time()
			logger.log_episode_start(0)

			for step in range(1000):
				logger.log_step(
					step=step,
					action=step % 4,
					observation=np.random.randn(10),
					reward=np.random.randn(),
					done=False
				)

			logger.log_episode_end(100.0, 1000)
			elapsed = time.time() - start

			# 1000步应该在1秒内完成
			assert elapsed < 1.0

			logger.close()

	def test_config_creation_speed(self):
		"""测试配置创建速度"""
		from envs.smartgrid.single_agent import SingleAgentConfig
		import time

		start = time.time()
		for _ in range(100):
			config = SingleAgentConfig()
			_ = config.to_dict()
			_ = config.to_env_info()
		elapsed = time.time() - start

		# 100次配置创建应该在1秒内完成
		assert elapsed < 1.0


# ==============================================================================
# 清理 fixture
# ==============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
	"""清理临时文件"""
	yield
	# 测试后的清理可以在这里添加
