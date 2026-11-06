"""检查真实参数流转、错误边界、快照和消费者契约。"""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from utils.config_pipeline import ConfigLoader, parse_assignments, read_yaml
from utils.config_validation import ConfigError, build_dataclass

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "env",
    [
        "vvc",
        "smartgrid",
        "dsr",
        "dsr_13bus",
        "stackelberg_13bus",
        "stackelberg",
        "district_dispatch",
    ],
)
def test_profile_roundtrip(env, tmp_path):
    original = ConfigLoader().load("sequential_ppo" if env.startswith("stackelberg") else "happo", env)
    snapshot = tmp_path / "resolved.yaml"
    original.save(snapshot)
    replay = ConfigLoader().load(experiment=snapshot)
    assert replay.to_dict() == original.to_dict()
    assert snapshot.with_name("resolved.sources.yaml").exists()


def test_reference_order_and_source(tmp_path):
    profile = write(
        tmp_path / "env.yaml", {"system_ref": "34Bus", "env": {"voltage_min": 0.92}}
    )
    experiment = write(tmp_path / "experiment.yaml", {"env": {"voltage_min": 0.93}})
    config = ConfigLoader().load(
        "happo",
        "vvc",
        env_config=profile,
        experiment=experiment,
        overrides={"env.voltage_min": 0.94},
    )
    assert config.env_args["system_name"] == "34Bus"
    assert config.env_args["voltage_min"] == 0.94
    history = config.explain("env.voltage_min")["history"]
    assert [item["value"] for item in history][-4:] == [0.95, 0.92, 0.93, 0.94]
    assert history[-1]["source"] == "CLI --set"
    assert all(len(digest) == 64 for digest in config.files.values())


@pytest.mark.parametrize("value", [None, "does-not-exist"])
def test_invalid_reference_is_not_a_fallback(tmp_path, value):
    profile = write(
        tmp_path / "env.yaml", {"system_ref": value, "env": {"unknown_field": 1}}
    )
    with pytest.raises((FileNotFoundError, ConfigError)):
        ConfigLoader().load("happo", "vvc", env_config=profile)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train.epiosde_length": 20},
        {"model.lr": False},
        {"device.cuda": "false"},
        {"train.n_rollout_threads": 0},
        {"env.max_episode_steps": 0},
        {"env.voltage_min": 1.1},
        {"train.episode_length": 25},
        {"train.num_env_steps": 1},
        {"algo.actor_num_mini_batch": 999999},
        {"model.hidden_sizes": [32, "typo"]},
        {"model.lr": float("nan")},
        {"env.pv_control_enabled": True},
        {"env.seed": 42},
        {"seed.seed": -1},
    ],
)
def test_invalid_values_fail_before_runner(overrides):
    with pytest.raises(ConfigError):
        ConfigLoader().load("happo", "vvc", overrides=overrides)


def test_zero_seed_false_and_independent_horizons():
    config = ConfigLoader().load(
        "happo",
        "vvc",
        overrides={
            "seed.seed": 0,
            "device.cuda": False,
            "env.max_episode_steps": 9,
            "train.episode_length": 120,
            "model.lr": "1e-4",
        },
    )
    assert config.algo_args["seed"]["seed"] == config.env_args["seed"] == 0
    assert config.algo_args["device"]["cuda"] is False
    assert config.env_args["max_episode_steps"] == 9
    assert config.algo_args["train"]["episode_length"] == 120
    assert config.algo_args["model"]["lr"] == 0.0001


def test_missing_ref_and_duplicate_yaml(tmp_path):
    path = write(tmp_path / "env.yaml", {"system_ref": "missing-system"})
    with pytest.raises(FileNotFoundError):
        ConfigLoader().load("happo", "vvc", env_config=path)
    path.write_text(
        "env:\n  voltage_min: 0.91\n  voltage_min: 0.93\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="重复"):
        read_yaml(path)


def test_algorithm_catalog_can_be_read_by_training_ui():
    """One malformed optional profile must not prevent UI catalog discovery."""
    profiles = sorted((ROOT / "configs/algos_cfgs").glob("*.yaml"))
    assert profiles
    for path in profiles:
        assert isinstance(read_yaml(path), dict), path.name


def test_changes_are_not_cached(tmp_path):
    path = tmp_path / "profile.yaml"
    loader = ConfigLoader()
    for horizon in (7, 11):
        write(path, {"env": {"max_episode_steps": horizon}})
        assert (
            loader.load("happo", "vvc", env_config=path).env_args["max_episode_steps"]
            == horizon
        )


def test_frozen_snapshot_does_not_reload_system(tmp_path):
    snapshot = tmp_path / "resolved.yaml"
    ConfigLoader().load("happo", "district_dispatch").save(snapshot)
    loader = ConfigLoader()
    original_read = loader._read

    def read(path, files):
        assert "configs/systems" not in str(path)
        assert "configs/envs_cfgs" not in str(path)
        return original_read(path, files)

    loader._read = read
    replay = loader.load(experiment=snapshot)
    assert replay.env_args["regions"][2]["controls"][0] == "PVSystem.pv_d2_1"


def test_ui_preview_and_training_read_identical_values(tmp_path, monkeypatch):
    from training_ui.core import config_builder

    monkeypatch.setattr(config_builder, "CACHE_DIR", tmp_path)
    overrides = {
        "system_ref": "34Bus",
        "env": {"voltage_min": 0.93},
        "model": {"lr": 0.0002},
    }
    preview = config_builder.resolve_config("happo", "vvc", "ui", overrides)
    path, document = config_builder.build_config("happo", "vvc", "ui", overrides)
    replay = ConfigLoader().load(experiment=path)
    assert preview.to_dict() == document == replay.to_dict()


def test_legacy_ui_envelope(tmp_path):
    original = ConfigLoader().load("happo", "vvc")
    path = write(
        tmp_path / "legacy.yaml",
        {
            "algo_name": "happo",
            "env_name": "vvc",
            "exp_name": "legacy",
            "algo_args": original.algo_args,
            "env_args": original.env_args,
        },
    )
    assert ConfigLoader().load(experiment=path).algo_args == original.algo_args


def test_custom_device_values_reach_vvc():
    from envs.vvc.vvc.vvc_config import VVCConfig

    config = ConfigLoader().load(
        "happo",
        "vvc",
        values={
            "env": {
                "environment_specific": {"devices": {"batteries": {"action_num": 17}}}
            }
        },
    )
    consumer = build_dataclass(VVCConfig, config.env_args)
    assert consumer.bat_act_num == 17


def test_voltage_threshold_changes_actual_reward():
    from types import SimpleNamespace
    from envs.vvc.runtime_env import Env
    from envs.vvc.vvc.vvc_config import VVCConfig

    fake = SimpleNamespace(
        obs={"bus_voltages": {"bus": [0.94, 1.06]}, "power_loss_ratio": 0},
        config=VVCConfig(),
        circuit=SimpleNamespace(regulators={}),
        t=0,
        horizon=1,
    )
    default = Env._reward(fake, [], [], [], [], [])
    fake.config = VVCConfig(voltage_min=0.93, voltage_max=1.07)
    changed = Env._reward(fake, [], [], [], [], [])
    assert default[1]["vol_reward"] == pytest.approx(-0.02)
    assert changed[1]["vol_reward"] == 0


def test_smartgrid_validates_final_nested_config():
    with pytest.raises(ConfigError):
        ConfigLoader().load(
            "happo", "smartgrid", values={"env": {"battery": {"action_num": 1}}}
        )
    config = ConfigLoader().load(
        "happo", "smartgrid", values={"env": {"lambda_lr": 0.003}}
    )
    from envs.smartgrid.base_env.env_config import SmartGridConfig

    assert (
        build_dataclass(SmartGridConfig, config.env_args).to_info_dict()["lambda_lr"]
        == 0.003
    )


def test_smartgrid_weights_reach_reward_formula():
    from types import SimpleNamespace
    from envs.smartgrid.base_env.env_config import SmartGridConfig
    from envs.smartgrid.rewards.vvc_reward import VVCReward

    fake = SimpleNamespace(
        obs={"power_loss_ratio": 0.02}, cap_num=1, reg_num=0, pv_num=0
    )

    def reward(weight):
        resolved = ConfigLoader().load(
            "happo",
            "smartgrid",
            values={
                "env": {
                    "reward_weights": {"power_loss": weight, "capacitor": weight},
                    "use_action_smoothing": False,
                    "voltage_min": 0.92,
                    "voltage_max": 1.08,
                }
            },
        )
        info = build_dataclass(SmartGridConfig, resolved.env_args).to_info_dict()
        return VVCReward(fake, info)

    first, second = reward(1), reward(2)
    assert second.reward_main([], [], [], []) == pytest.approx(
        2 * first.reward_main([], [], [], [])
    )
    assert second.control_reward([1], [], [], []) == pytest.approx(
        2 * first.control_reward([1], [], [], [])
    )
    assert first.voltage_range == (0.92, 1.08)


def test_district_rejects_duplicate_capacity_and_uses_market_override():
    config = ConfigLoader().load(
        "happo", "district_dispatch", overrides={"env.market.base_price_yuan_kwh": 0.8}
    )
    assert config.env_args["market"]["base_price_yuan_kwh"] == 0.8
    with pytest.raises(ConfigError, match="district_devices"):
        ConfigLoader().load(
            "happo",
            "district_dispatch",
            overrides={"env.district_devices.2.storage_capacity_kwh": 400},
        )


@pytest.mark.parametrize(
    "field", ["tou_config", "tier_config", "reward_weights", "n1_security"]
)
def test_unsupported_stackelberg_features_are_explicit(field):
    with pytest.raises(ConfigError, match="未知参数"):
        ConfigLoader().load(
            "sequential_ppo", "stackelberg_13bus", values={"env": {field: {"enable": True}}}
        )


def test_district_consumes_market_costs_and_native_bindings():
    from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv
    from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore

    config = ConfigLoader().load(
        "happo",
        "district_dispatch",
        values={
            "env": {
                "market": {"base_price_yuan_kwh": 0.7},
                "costs": {"terminal_energy_yuan_kwh": 4.0},
            }
        },
    )
    consumer = DistrictDispatchEnv._parse_config(config.env_args)
    assert consumer.market.base_price_yuan_kwh == 0.7
    assert consumer.costs.terminal_energy_yuan_kwh == 4.0
    assert consumer.regions[2].controls == ["PVSystem.pv_d2_1", "Storage.ess_d2"]
    core = DistrictDispatchCore(consumer)
    try:
        assert core.circuit.batteries["ess_d2"].max_kwh == 300
        assert core.circuit.batteries["ess_d2"].soc == 0.5
    finally:
        core.close()


def test_pricing_config_reaches_native_task_once():
    from envs.stackelberg.pricing_env import SequentialPricingEnv
    from envs.stackelberg.stackelberg_config import StackelbergConfig
    cfg = ConfigLoader().load('sequential_ppo', 'stackelberg', values={'env': {'network_cost_yuan_per_hour': 77.0}})
    typed = build_dataclass(StackelbergConfig, cfg.env_args)
    env = SequentialPricingEnv(typed)
    try:
        assert env.config is typed
        assert env.config.network_cost_yuan_per_hour == 77.0
        assert env.groups == [['household'], ['business']]
    finally:
        env.close()


def test_preview_has_no_runtime_imports():
    code = (
        "from utils.config_pipeline import ConfigLoader; import sys; "
        "[ConfigLoader().load('sequential_ppo' if e.startswith('stackelberg') else 'happo', e) for e in ('vvc','smartgrid','dsr','district_dispatch','stackelberg_13bus')]; "
        "assert not {'torch', 'dss', 'opendssdirect', 'gradio'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=20)


@pytest.mark.parametrize(
    "args",
    [
        ["--seed", "42"],
        ["--typo", "1"],
        ["--cuda"],
        ["--set", 'model.lr=__import__("os").system("whoami")'],
    ],
)
def test_cli_rejects_bad_or_ambiguous_parameters(args):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/multi_agent/scripts/train.py"),
            "--env",
            "vvc",
            "--validate-config",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_worker_seed_set_before_construction(monkeypatch):
    from utils import envs_tools
    from envs.vvc import vvc_env

    observed = []

    class FakeEnv:
        def __init__(self, config, rank):
            import numpy as np

            assert (
                np.random.random() == np.random.RandomState(config.seed).random_sample()
            )
            observed.append((config.seed, rank))

        def seed(self, value):
            observed.append(value)

    monkeypatch.setattr(vvc_env, "VVCEnv", FakeEnv)
    config = ConfigLoader().load("happo", "vvc")
    envs_tools.make_power_env("vvc", config.env_args, 2, 2000)
    assert observed == [(2000, 2), 2000]
