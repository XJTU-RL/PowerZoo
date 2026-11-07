"""SmartGrid 当前正式配置契约；不以猜测类名、空断言或 skip 代替校验。"""

from dataclasses import asdict
import math

import pytest

from envs.smartgrid.base_env.env_config import (
    DataConfig,
    RewardWeights,
    SmartGridConfig,
)
from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError


def test_default_typed_sections_and_actual_action_fields():
    cfg = SmartGridConfig()
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.reward_weights, RewardWeights)
    assert cfg.RBP_act_num == (33, 33, math.inf)
    assert cfg.pv_control_enabled is True
    assert not {"capacitor", "regulator", "battery", "pv", "pv_data_source"} & set(
        asdict(cfg)
    )


def test_config_legacy_input_is_flattened_once_and_roundtrips():
    cfg = SmartGridConfig.from_env_args(
        {
            "regulator": {"action_num": 17},
            "battery": {"action_num": math.inf},
            "pv": {"control_enabled": False},
        }
    )
    assert cfg.RBP_act_num == (17, math.inf, math.inf)
    assert cfg.pv_control_enabled is False
    assert asdict(SmartGridConfig.from_dict(asdict(cfg))) == asdict(cfg)


def test_old_shared_device_metadata_is_normalized_at_input():
    cfg = SmartGridConfig.from_env_args(
        {
            "env_specific_config": {
                "devices": {
                    "regulators": {"action_num": 17},
                    "pv_systems": {
                        "control_enabled": True,
                        "action_space": "continuous",
                    },
                }
            }
        }
    )
    assert cfg.reg_act_num == 17 and cfg.pv_act_num == math.inf


@pytest.mark.parametrize(
    "values",
    [
        {"max_episode_steps": 0},
        {"max_episode_steps": 1441},
        {"reg_act_num": math.inf},
        {"reg_act_num": 1},
        {"bat_act_num": 4},
        {"pv_act_num": 1},
        {"voltage_min": 1.1},
        {"power_base": 0},
        {"lambda_init": -1},
        {"lambda_min": 2},
        {"lambda_lr": -1},
        {"lambda_momentum": 1},
        {"lambda_update_strategy": "made_up"},
        {"reward_weights": {"pv": -1}},
        {"data": {"invented": 1}},
        {"reg_act_num": None},
        {"useS": False},
        {"regulator": {"action_num": 17}, "reg_act_num": 33},
        {"pv": {"control_enabled": False}, "pv_control_enabled": True},
        {"pv_control": False, "pv_control_enabled": True},
        {"environment_specific": {"invented": 1}},
        {"environment_specific": {"devices": {"regulators": {"invented": 1}}}},
        {"environment_specific": {"devices": {"capacitors": {"action_num": 3}}}},
        {
            "environment_specific": {
                "devices": {"batteries": {"control_enabled": False}}
            }
        },
        {
            "environment_specific": {
                "devices": {"regulators": {"action_space": "invented"}}
            }
        },
        {"capacitor": {"action_num": 3}},
        {"battery": {"control_enabled": False}},
    ],
)
def test_invalid_or_unimplemented_settings_are_rejected(values):
    with pytest.raises(ConfigError):
        SmartGridConfig.from_env_args(values)


def test_actual_default_profile_has_explicit_data_source_and_physical_plan():
    cfg = ConfigLoader().load("happo", "smartgrid")
    assert cfg.env_args["data"]["source"] == "synthetic"
    assert cfg.env_args["pv_plan"] == "aggressive"
    assert cfg.env_args["use_lagrangian_reward"] is True
