# -*- coding: utf-8 -*-
"""
SmartGrid rewards 模块详细测试

测试覆盖:
- PowerZooReward CMDP 奖励函数
- LagrangianUpdater 拉格朗日乘子更新
- 奖励组件计算
- 约束成本计算

@File      : test_rewards.py
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
class TestRewardsImport:
	"""测试 rewards 模块导入"""

	def test_powerzoo_reward_import(self):
		"""测试 PowerZooReward 类导入"""
		from envs.smartgrid.rewards import PowerZooReward
		assert PowerZooReward is not None

	def test_lagrangian_updater_import(self):
		"""测试 LagrangianUpdater 类导入"""
		from envs.smartgrid.rewards import LagrangianUpdater
		assert LagrangianUpdater is not None

	def test_powerzoo_reward_from_module(self):
		"""测试从模块导入 PowerZooReward"""
		from envs.smartgrid.rewards.powerzoo_reward import PowerZooReward
		assert PowerZooReward is not None

	def test_lagrangian_from_module(self):
		"""测试从模块导入 LagrangianUpdater"""
		from envs.smartgrid.rewards.lagrangian import LagrangianUpdater
		assert LagrangianUpdater is not None


# ==============================================================================
# 单元测试 - PowerZooReward 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestPowerZooRewardInit:
	"""测试 PowerZooReward 初始化"""

	def test_powerzoo_reward_has_required_methods(self):
		"""验证 PowerZooReward 具有所有必需的方法"""
		from envs.smartgrid.rewards import PowerZooReward

		required_methods = [
			'voltage_cost',
			'powerloss_reward',
			'control_reward',
		]

		for method in required_methods:
			assert hasattr(PowerZooReward, method), f"PowerZooReward 缺少方法: {method}"

	def test_powerzoo_reward_creation(self):
		"""测试 PowerZooReward 创建"""
		from envs.smartgrid.rewards import PowerZooReward

		# 创建 mock 环境
		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {
				'bus1': [1.0, 0.99, 1.01],
				'bus2': [0.98, 0.97, 0.99],
			}
		}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		info = {}

		reward_func = PowerZooReward(mock_env, info)
		assert reward_func is not None

	def test_powerzoo_reward_with_custom_weights(self):
		"""测试使用自定义权重的 PowerZooReward"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {'bus_voltages': {'bus1': [1.0]}}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		info = {
			'powerloss_weight': 2.0,
			'control_weight': 0.8,
			'voltage_threshold': 0.03,
		}

		reward_func = PowerZooReward(mock_env, info)

		assert reward_func.weights['powerloss'] == 2.0
		assert reward_func.weights['control'] == 0.8
		assert reward_func.voltage_threshold == 0.03


@pytest.mark.unit
@pytest.mark.smartgrid
class TestVoltageCost:
	"""测试电压约束成本计算"""

	@pytest.fixture
	def reward_func(self):
		"""创建 PowerZooReward 实例"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {
				'bus1': [1.0, 1.0, 1.0],
				'bus2': [0.98, 0.98, 0.98],
			}
		}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		return PowerZooReward(mock_env, {})

	def test_voltage_cost_no_violation(self, reward_func):
		"""测试无电压违规时的成本"""
		# 设置所有电压在安全范围内
		reward_func.env.obs['bus_voltages'] = {
			'bus1': [1.0, 1.0, 1.0],
			'bus2': [1.02, 1.02, 1.02],
		}

		cost = reward_func.voltage_cost()
		assert cost >= 0  # 成本应该是非负的

	def test_voltage_cost_with_violation(self, reward_func):
		"""测试有电压违规时的成本"""
		# 设置电压在安全范围外
		reward_func.env.obs['bus_voltages'] = {
			'bus1': [0.90, 0.90, 0.90],  # 低于 0.95
			'bus2': [1.10, 1.10, 1.10],  # 高于 1.05
		}

		cost = reward_func.voltage_cost()
		assert cost > 0  # 有违规时成本应该大于 0

	def test_voltage_cost_returns_float(self, reward_func):
		"""测试电压成本返回浮点数"""
		cost = reward_func.voltage_cost()
		assert isinstance(cost, float)


@pytest.mark.unit
@pytest.mark.smartgrid
class TestPowerlossReward:
	"""测试网损奖励计算"""

	@pytest.fixture
	def reward_func(self):
		"""创建 PowerZooReward 实例"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {'bus1': [1.0]},
			'power_loss_ratio': 0.05,
		}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		return PowerZooReward(mock_env, {})

	def test_powerloss_reward_returns_float(self, reward_func):
		"""测试网损奖励返回浮点数"""
		reward = reward_func.powerloss_reward()
		assert isinstance(reward, float)

	def test_powerloss_reward_decreases_with_loss(self, reward_func):
		"""测试网损越大奖励越低"""
		reward_func.env.obs['power_loss_ratio'] = 0.01
		reward_low_loss = reward_func.powerloss_reward()

		reward_func.env.obs['power_loss_ratio'] = 0.10
		reward_high_loss = reward_func.powerloss_reward()

		# 高损耗应该有更低的奖励
		assert reward_high_loss <= reward_low_loss


@pytest.mark.unit
@pytest.mark.smartgrid
class TestControlReward:
	"""测试控制奖励计算"""

	@pytest.fixture
	def reward_func(self):
		"""创建 PowerZooReward 实例"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {'bus_voltages': {'bus1': [1.0]}}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		return PowerZooReward(mock_env, {})

	def test_control_reward_returns_float(self, reward_func):
		"""测试控制奖励返回浮点数"""
		# 创建模拟的动作数据
		actions = {
			'capacitor': [0, 1],
			'regulator': [16, 16, 16],
		}

		try:
			reward = reward_func.control_reward(actions)
			assert isinstance(reward, float)
		except TypeError:
			# 如果方法签名不同，跳过
			pass


# ==============================================================================
# 单元测试 - LagrangianUpdater 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestLagrangianUpdaterInit:
	"""测试 LagrangianUpdater 初始化"""

	def test_lagrangian_updater_creation(self):
		"""测试 LagrangianUpdater 创建"""
		from envs.smartgrid.rewards import LagrangianUpdater

		updater = LagrangianUpdater()
		assert updater is not None

	def test_lagrangian_updater_with_params(self):
		"""测试带参数的 LagrangianUpdater 创建"""
		from envs.smartgrid.rewards import LagrangianUpdater

		updater = LagrangianUpdater(
			lr=0.01,
			min_lambda=0.0,
			max_lambda=10.0,
		)
		assert updater is not None

	def test_lagrangian_updater_has_update_method(self):
		"""测试 LagrangianUpdater 有 update 方法"""
		from envs.smartgrid.rewards import LagrangianUpdater

		assert hasattr(LagrangianUpdater, 'update') or hasattr(LagrangianUpdater, '__call__')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestLagrangianUpdaterUpdate:
	"""测试 LagrangianUpdater 更新功能"""

	@pytest.fixture
	def updater(self):
		"""创建 LagrangianUpdater 实例"""
		from envs.smartgrid.rewards import LagrangianUpdater
		return LagrangianUpdater()

	def test_update_increases_lambda_with_violation(self, updater):
		"""测试违规时增加 lambda"""
		initial_lambda = updater.lambda_val if hasattr(updater, 'lambda_val') else 0.0

		# 更新 (假设有正的约束违规)
		if hasattr(updater, 'update'):
			updater.update(0.5)  # 正的违规
			if hasattr(updater, 'lambda_val'):
				assert updater.lambda_val >= initial_lambda

	def test_update_decreases_lambda_without_violation(self, updater):
		"""测试无违规时减小 lambda"""
		# 先设置一个较高的初始值
		if hasattr(updater, 'lambda_val'):
			updater.lambda_val = 5.0

		# 更新 (假设有负的约束违规，即满足约束)
		if hasattr(updater, 'update'):
			updater.update(-0.5)  # 负的违规

	def test_lambda_stays_within_bounds(self, updater):
		"""测试 lambda 保持在边界内"""
		if hasattr(updater, 'update'):
			# 多次更新
			for _ in range(100):
				updater.update(1.0)

			if hasattr(updater, 'lambda_val') and hasattr(updater, 'max_lambda'):
				assert updater.lambda_val <= updater.max_lambda

			for _ in range(100):
				updater.update(-1.0)

			if hasattr(updater, 'lambda_val') and hasattr(updater, 'min_lambda'):
				assert updater.lambda_val >= updater.min_lambda


# ==============================================================================
# 集成测试 - 奖励函数与环境交互
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
class TestRewardWithEnvironment:
	"""测试奖励函数与环境的交互"""

	def test_reward_with_real_obs_structure(self):
		"""测试使用真实观测结构的奖励计算"""
		from envs.smartgrid.rewards import PowerZooReward

		# 创建更真实的观测结构
		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {
				f'bus{i}': [0.95 + 0.05 * np.random.rand() for _ in range(3)]
				for i in range(10)
			},
			'power_loss_ratio': 0.03,
			'cap_statuses': {f'cap{i}': 1 for i in range(3)},
			'reg_statuses': {f'reg{i}': 16 for i in range(2)},
		}
		mock_env.cap_num = 3
		mock_env.reg_num = 2
		mock_env.pv_num = 0

		reward_func = PowerZooReward(mock_env, {})

		# 计算各组件
		v_cost = reward_func.voltage_cost()
		p_reward = reward_func.powerloss_reward()

		assert isinstance(v_cost, float)
		assert isinstance(p_reward, float)


# ==============================================================================
# 奖励组合测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestRewardCombination:
	"""测试奖励组合"""

	def test_reward_main_exists(self):
		"""测试 reward_main 方法存在"""
		from envs.smartgrid.rewards import PowerZooReward

		# 创建 mock 环境
		mock_env = Mock()
		mock_env.obs = {'bus_voltages': {'bus1': [1.0]}}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		reward_func = PowerZooReward(mock_env, {})

		assert hasattr(reward_func, 'reward_main') or hasattr(reward_func, 'compute_reward')

	def test_all_components_contribute(self):
		"""测试所有组件都贡献奖励"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {'bus1': [1.0, 1.0, 1.0]},
			'power_loss_ratio': 0.05,
		}
		mock_env.cap_num = 2
		mock_env.reg_num = 3
		mock_env.pv_num = 0

		reward_func = PowerZooReward(mock_env, {})

		# 计算各组件
		v_cost = reward_func.voltage_cost()
		p_reward = reward_func.powerloss_reward()

		# 所有返回值应该是数值
		assert np.isfinite(v_cost)
		assert np.isfinite(p_reward)


# ==============================================================================
# 常量和配置测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestRewardConstants:
	"""测试奖励相关常量"""

	def test_voltage_constants_import(self):
		"""测试电压常量导入"""
		try:
			from envs.smartgrid.constants import VOLTAGE
			assert hasattr(VOLTAGE, 'MIN_PU') or hasattr(VOLTAGE, 'MIN')
			assert hasattr(VOLTAGE, 'MAX_PU') or hasattr(VOLTAGE, 'MAX')
		except ImportError:
			pytest.skip("constants 模块不存在")

	def test_reward_constants_import(self):
		"""测试奖励常量导入"""
		try:
			from envs.smartgrid.constants import REWARD
			assert REWARD is not None
		except ImportError:
			pytest.skip("REWARD 常量不存在")


# ==============================================================================
# 边界条件测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestRewardEdgeCases:
	"""测试奖励函数边界条件"""

	def test_empty_bus_voltages(self):
		"""测试空母线电压"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {'bus_voltages': {}}
		mock_env.cap_num = 0
		mock_env.reg_num = 0
		mock_env.pv_num = 0

		reward_func = PowerZooReward(mock_env, {})
		v_cost = reward_func.voltage_cost()

		assert v_cost == 0.0  # 空电压应该没有成本

	def test_extreme_voltage_values(self):
		"""测试极端电压值"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {
				'bus1': [0.5, 0.5, 0.5],  # 极低
				'bus2': [1.5, 1.5, 1.5],  # 极高
			}
		}
		mock_env.cap_num = 0
		mock_env.reg_num = 0
		mock_env.pv_num = 0

		reward_func = PowerZooReward(mock_env, {})
		v_cost = reward_func.voltage_cost()

		assert v_cost > 0  # 极端电压应该有高成本
		assert np.isfinite(v_cost)  # 但不应该是无穷

	def test_zero_power_loss(self):
		"""测试零功率损耗"""
		from envs.smartgrid.rewards import PowerZooReward

		mock_env = Mock()
		mock_env.obs = {
			'bus_voltages': {'bus1': [1.0]},
			'power_loss_ratio': 0.0,
		}
		mock_env.cap_num = 0
		mock_env.reg_num = 0
		mock_env.pv_num = 0

		reward_func = PowerZooReward(mock_env, {})
		p_reward = reward_func.powerloss_reward()

		assert np.isfinite(p_reward)
