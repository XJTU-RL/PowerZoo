"""用独立公式验算实际奖励；设备和能量另由原生物理测试验收。"""

import math
from types import SimpleNamespace

import pytest

from envs.smartgrid.base_env.env_config import SmartGridConfig, RewardWeights
from envs.smartgrid.rewards.vvc_reward import VVCReward
from envs.smartgrid.rewards.lagrangian import LagrangianUpdater


def fixture_reward(**kwargs):
    env = SimpleNamespace(
        obs={
            "bus_voltages": {"a": [1, 0.9], "b": [1.1]},
            "power_loss_ratio": 0.04,
            "pv_statuses": {"pv": [0.6, 0.1]},
        },
        cap_num=2,
        reg_num=2,
        bat_num=2,
        pv_num=1,
        t=1,
        horizon=2,
        pv_control_enabled=True,
        applied_control_vector=[0, 0, 0],
    )
    return env, VVCReward(env, SmartGridConfig(**kwargs))


def test_voltage_cost_uses_each_existing_phase():
    _, reward = fixture_reward(voltage_threshold=0.02)
    assert reward.voltage_cost() == pytest.approx((0.08**2 + 0.08**2) / 3)


@pytest.mark.parametrize("ratio", [0, 0.01, 0.04, 0.1])
def test_loss_formula_uses_fixed_reference(ratio):
    env, reward = fixture_reward(power_base=0.02)
    env.obs["power_loss_ratio"] = ratio
    assert reward.powerloss_reward() == pytest.approx(-math.log1p(ratio / 0.02))


def test_control_weights_and_terminal_soc_are_not_ignored():
    weights = RewardWeights(
        capacitor=2, regulator=3, battery_soc=5, battery_discharge=7, pv_control=11
    )
    env, reward = fixture_reward(reward_weights=weights, use_action_smoothing=False)
    args = ([1, 0], [0.1, 0.2], [0.2, 0.4], [0.5, 0], [0.1])
    base = 2 * 0.5 + 3 * 0.15 + 7 * 0.25 + 11 * 0.1
    assert reward.control_reward(*args) == pytest.approx(-base)
    env.t = 2
    assert reward.control_reward(*args) == pytest.approx(-base - 5 * 0.3)


def test_smoothing_uses_executed_levels_not_difference_of_differences():
    env, reward = fixture_reward(reward_weights=RewardWeights(action_smoothing=0.6))
    assert reward.control_reward([], [], [], []) == 0
    env.applied_control_vector = [0, 0.5, -1]
    assert reward.control_reward([], [], [], []) == pytest.approx(-0.6 * 1.5 / 3)
    assert reward.control_reward([], [], [], []) == 0
    reward.reset_episode()
    assert reward.previous_actions is None


def test_main_reward_preserves_declared_weights():
    _, reward = fixture_reward(
        reward_weights=RewardWeights(power_loss=2, control=3, pv=4),
        use_action_smoothing=False,
    )
    assert reward.reward_main([], [], [], [], []) == pytest.approx(
        -2 * math.log(3) + 4 * 0.6 * 0.5
    )


@pytest.mark.parametrize("cost,expected", [(0.5, 1.2), (0, 0.95), (100, 2), (-100, 0)])
def test_lagrangian_update_and_bounds(cost, expected):
    updater = LagrangianUpdater(
        init_lambda=1, lr=0.5, target_cost=0.1, lambda_min=0, lambda_max=2
    )
    assert updater.update(cost) == pytest.approx(expected)
    assert updater.update_count == 1 and updater.cost_history == [cost]
    updater.reset(0.5)
    assert (
        updater.lmbda == 0.5
        and updater.update_count == 0
        and updater.cost_history == []
    )
