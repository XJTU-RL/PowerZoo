"""配置精简的行为证据；不能用字典中出现过某个键代替实际生效。"""

from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError, build_dataclass, environment_class

ROOT = Path(__file__).resolve().parents[1]


def district_config(**changes):
    cfg = ConfigLoader().load("happo", "district_dispatch", values={"env": changes})
    return build_dataclass(environment_class("district_dispatch"), cfg.env_args)


def test_hourly_prices_have_declared_units_and_no_hidden_profile():
    cfg = district_config(
        market={"base_price_yuan_kwh": 0.7, "tou_multipliers": list(range(1, 25))}
    )
    for hour in range(24):
        buy, sell = cfg.market.price_at(hour * 3600)
        assert buy == pytest.approx(0.7 * (hour + 1))
        assert sell == pytest.approx(buy * cfg.market.sell_ratio)


def test_native_capacity_cannot_be_shadowed_by_task_config():
    for key in (
        "soc_min",
        "soc_init",
        "soc_max",
        "district_devices",
        "n_districts",
        "load_noise",
    ):
        with pytest.raises(ConfigError, match=key):
            district_config(**{key: 0.3})


@pytest.mark.parametrize(
    "family,key",
    [
        ("vvc", "irrad_dss"),
        ("smartgrid", "big2small"),
        ("dsr", "use_action_mask"),
        ("stackelberg", "tou_config"),
        ("district_dispatch", "connection_mode"),
    ],
)
def test_retired_no_op_fields_cannot_silently_enter_training(family, key):
    with pytest.raises(ConfigError, match=key):
        ConfigLoader().load("happo", family, values={"env": {key: True}})


def test_unused_system_knobs_and_component_keys_are_rejected():
    loader = ConfigLoader()
    system = loader.load_system("13Bus")
    system["devices"]["regulators"]["count"] = 900
    with pytest.raises(ConfigError, match="count"):
        loader._system_values(system, "vvc")
    with pytest.raises(ConfigError, match="consumer_delay"):
        loader.load(
            "happo",
            "stackelberg",
            values={"env": {"async_config": {"consumer_delay": 5}}},
        )
    with pytest.raises(ConfigError, match="sentivity_order"):
        loader.load("happo", "vvc", values={"algo": {"sentivity_order": True}})


def test_config_preview_does_not_import_simulation_or_training_dependencies():
    code = "from utils.config_pipeline import ConfigLoader; import sys; ConfigLoader().load('happo','smartgrid'); assert not {'numpy','pandas','dss','torch'} & set(sys.modules)"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_smartgrid_requires_a_consumer_for_lagrangian_penalty():
    with pytest.raises(ConfigError, match="use_cmdp"):
        ConfigLoader().load(
            "happo",
            "smartgrid",
            values={"env": {"use_cmdp": False, "use_lagrangian_reward": True}},
        )


def test_file_data_does_not_accept_inapplicable_generator_overrides():
    from envs.smartgrid.base_env.env_config import DataConfig

    with pytest.raises(ValueError, match="合成数据"):
        DataConfig(source="files", path="manifest.json", noise=0.5)


def test_shared_training_preset_overrides_focused_environment_recommendations():
    cfg = ConfigLoader().load("happo", "district_dispatch")
    assert cfg.algo_args["train"]["num_env_steps"] > 240
    smoke = ConfigLoader().load(
        experiment="configs/experiments/district_dispatch_smoke.yaml"
    )
    assert smoke.algo_args["train"]["num_env_steps"] == 240
    assert not list((ROOT / "configs/training").glob("*_default.yaml"))
