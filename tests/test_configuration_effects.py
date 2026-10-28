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


def district_core(config):
    from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore
    from envs.district_dispatch.core.loadprofile import DistrictLoadProfile

    core = DistrictDispatchCore.__new__(DistrictDispatchCore)
    core.config = config
    core.system_def = core._load_system_definition()
    core.districts = core._build_districts()
    core.max_obs_dim = config.get_max_obs_dim()
    core.current_step = 0
    core.load_profile = DistrictLoadProfile(config)
    return core


def test_price_and_hourly_multipliers_change_actual_step_price():
    from envs.district_dispatch.core.loadprofile import DistrictLoadProfile

    base = district_config()
    original = DistrictLoadProfile(base)
    doubled = DistrictLoadProfile(
        district_config(base_price_yuan_kwh=base.base_price_yuan_kwh * 2)
    )
    for step in range(base.max_episode_steps):
        assert doubled.get_step_data(step, False)["price"] == pytest.approx(
            original.get_electricity_price(step) * 2
        )
    hourly = list(range(1, 25))
    profile = DistrictLoadProfile(
        district_config(tou_multipliers=hourly, base_price_yuan_kwh=0.7)
    )
    for hour in range(24):
        assert profile.get_electricity_price(hour * 4) == pytest.approx(
            hourly[hour] * 0.7
        )
    flat = DistrictLoadProfile(
        district_config(tou_multipliers=[1.0] * 24, base_price_yuan_kwh=0.7)
    )
    assert flat.get_electricity_price(40) == 0.7
    assert flat.get_price_normalized(40) == 0.5


@pytest.mark.parametrize("soc,command", [(0.25, 1.0), (0.75, -1.0)])
def test_storage_limits_reach_actual_control_guard(soc, command):
    configs = [
        district_config(),
        district_config(
            soc_min=0.3, soc_max=0.7, soc_healthy_min=0.3, soc_healthy_max=0.7
        ),
    ]
    powers = []
    for config in configs:
        district = district_core(config).districts[0]
        for storage in district.storage_units:
            storage.soc = soc
        adapter = Mock()
        district.apply_controls(
            adapter,
            np.zeros(len(district.pv_units)),
            np.full(len(district.storage_units), command),
            np.zeros(len(district.ev_chargers)),
        )
        assert adapter.set_storage_power.called
        powers.append(adapter.set_storage_power.call_args.args[1])
    assert powers[0] * command > 0
    assert powers[1] == 0


def test_neighbor_option_changes_actual_observation_but_not_action_space():
    enabled = district_core(district_config())
    disabled = district_core(district_config(use_neighbor_obs=False))
    disabled._get_neighbor_info = Mock(side_effect=AssertionError("不应读取邻居"))
    data = {"time_of_day": 0.0}
    with_neighbors = enabled._build_all_obs(data)
    without = disabled._build_all_obs(data)
    assert enabled.max_obs_dim > disabled.max_obs_dim
    assert all(len(obs) == disabled.max_obs_dim for obs in without)
    assert all(len(obs) == enabled.max_obs_dim for obs in with_neighbors)
    assert enabled.config.get_max_action_dim() == disabled.config.get_max_action_dim()
    disabled._get_neighbor_info.assert_not_called()


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
