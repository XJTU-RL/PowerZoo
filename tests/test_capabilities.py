"""静态能力检查完全绕过仿真和训练器。"""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from gymnasium import spaces
import pytest

from utils.capabilities import (
    DeviceLayout,
    device_layout,
    discover_capabilities,
    preflight,
)
from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError


@pytest.fixture
def config():
    return ConfigLoader().load(experiment="configs/experiments/vvc_13bus_smoke.yaml")


def test_catalog_is_serializable_and_does_not_import_runtime():
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys\n"
            "from scripts.check_capabilities import main\n"
            "assert main(['--backend', 'sb3_ppo', '--independent-replay', "
            "'--checkpoint-contract', 'legacy_physics_only']) == 0\n"
            "assert not {'torch', 'dss', 'runners', 'stable_baselines3'} & sys.modules.keys()",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["stages"]["preflight"]["status"] == "passed"
    catalog = discover_capabilities()
    assert catalog["environments"]["stackelberg"]["interaction"] == "sequential"
    assert catalog["environments"]["vvc"]["profiles"]["marl"]["mask"] == "structural"
    assert catalog["environments"]["dsr"]["profiles"]["marl"]["mask"] == "feasible"
    assert catalog["algorithms"]["ldcppo"]["termination"] == "terminated_and_truncated"
    json.dumps(catalog)


def test_centralized_policy_and_device_spaces_are_separate(config):
    result = preflight(config, backend="sb3_ppo", checkpoint="policy_reload")
    assert result.compatible and result.deferred
    assert result.action_kinds == ("continuous",)
    assert result.action_encoding == "box_quantizes_discrete_devices"
    assert result.device_action_kinds == ("discrete",)
    dsr = ConfigLoader().load(env="dsr")
    assert preflight(dsr, backend="sb3_ppo").action_kinds == ("multidiscrete",)
    assert not preflight(dsr, backend="ldcppo").compatible


def test_unknown_algorithms_are_explicit(config):
    assert "未知" in preflight(config, backend="sac_typo").issues[0]
    config.main_args["algo"] = "qmix_typo"
    assert "未知" in preflight(config).issues[0]
    config.main_args["env"] = "TYPO"
    with pytest.raises(ConfigError, match="未知环境"):
        preflight(config)



@pytest.mark.parametrize("algorithm", ["ippo", "qmix", "vdn", "coma"])
def test_discrete_baselines_preflight_matches_public_config(algorithm):
    config = ConfigLoader().load(
        algo=algorithm,
        experiment="configs/experiments/vvc_34bus_discrete_marl_quickstart.yaml",
    )
    structural = preflight(config, require_mask="structural", checkpoint="policy_reload")
    assert structural.compatible and structural.device_action_kinds == ("discrete",)
    assert structural.observation == "local"
    assert not preflight(config, require_mask="feasible").compatible

    config.env_args["bat_act_num"] = float("inf")
    assert not preflight(config).compatible


def test_cbess_binding_supports_explicit_validation_from_train():
    config = ConfigLoader().load(
        experiment="configs/experiments/cbess_ldcppo_quickstart.yaml"
    )
    assert preflight(config, backend="ldcppo", checkpoint="training_resume").compatible
    for backend in ("ldcppo", "sb3_sac", "sb3_td3"):
        result = preflight(
            config, backend=backend, checkpoint="policy_reload", independent_replay=True
        )
        assert result.compatible and result.checkpoint_contract == "binding_v1"
    config.env_args["split"] = "validation"
    assert not preflight(config, backend="ldcppo", independent_replay=True).compatible
    assert not preflight(config).compatible


def test_ldcppo_costs_and_vvc_action_mask_are_not_inferred(config):
    errors = preflight(config, backend="ldcppo").issues
    assert any("约束成本" in error for error in errors)
    assert preflight(config, require_mask="structural").compatible
    assert not preflight(config, require_mask="feasible").compatible
    assert not preflight(
        config, backend="sb3_ppo", require_mask="structural"
    ).compatible
    config.env_args["data"]["split"] = "test"
    assert not preflight(config, independent_replay=True).compatible


def test_mixed_actions_wait_for_native_layout(config):
    config.main_args["algo"] = "matd3"
    assert not preflight(config).compatible
    config.env_args["bat_act_num"] = float("inf")
    static = preflight(config)
    assert static.compatible and "mixed" in static.action_kinds
    mixed = DeviceLayout(
        ("Capacitor.x", "Storage.y"), ("discrete", "continuous"), False
    )
    assert not preflight(config, layout=mixed).compatible
    storage_only = DeviceLayout(("Storage.y",), ("continuous",), True)
    resolved = preflight(config, layout=storage_only)
    assert resolved.compatible and not resolved.deferred
    assert resolved.device_ids == ("Storage.y",)


def test_sharing_and_observation_requirements(config):
    config.algo_args["algo"]["share_param"] = True
    assert preflight(config).deferred
    discrete = DeviceLayout(
        ("RegControl.x", "Capacitor.y"), ("discrete", "discrete"), False
    )
    assert not preflight(config, layout=discrete).compatible
    config.algo_args["algo"]["share_param"] = "group"
    config.env_args["observation_mode"] = "local"
    assert preflight(config, layout=discrete).compatible
    config.env_args["state_type"] = "FP"
    assert not preflight(config, layout=discrete).compatible


def test_runtime_layout_reads_ids_and_checks_space_alignment():
    obs = spaces.Box(-1, 1, (2,))
    box = spaces.Box(-1, 1, (1,))
    discrete = spaces.Discrete(2)
    schema = [
        {"element": "Capacitor.a", "space": str(discrete)},
        {"element": "Storage.b", "space": str(box)},
    ]
    wrapper = SimpleNamespace(
        action_space=[discrete, box],
        observation_space=[obs, obs],
        get_physics_manifest=lambda: {"action_schema": schema},
    )
    layout = device_layout(wrapper)
    assert layout.device_ids == ("Capacitor.a", "Storage.b")
    assert layout.kind == "mixed" and not layout.homogeneous
    schema[0]["space"] = str(box)
    with pytest.raises(ConfigError, match="空间不一致"):
        device_layout(wrapper)
    with pytest.raises(ConfigError, match="唯一"):
        DeviceLayout(("RegControl.x", "RegControl.X"), ("discrete", "discrete"), True)


@pytest.mark.parametrize("family", ["vvc", "dsr"])
def test_current_checkpoint_does_not_advertise_held_out_replay(family):
    cfg = ConfigLoader().load(env=family)
    profile = discover_capabilities()["environments"][family]["profiles"]["centralized"]
    assert (profile["checkpoint_binding"], profile["evaluation"]) == (
        "binding_v1",
        "same_scenario_only",
    )
    for contract in (None, "binding_v1"):
        result = preflight(
            cfg,
            backend="sb3_ppo",
            checkpoint_contract=contract,
            checkpoint="policy_reload",
            independent_replay=True,
        )
        assert not result.compatible and any(
            "同场景" in issue for issue in result.issues
        )
        assert result.checkpoint_contract == "binding_v1"
        assert preflight(cfg, backend="sb3_ppo", checkpoint="policy_reload").compatible


def test_legacy_reference_scope(config):
    args = {"checkpoint_contract": "legacy_physics_only", "independent_replay": True}
    assert preflight(config, backend="sb3_ppo", **args).compatible
    for backend, cfg in (
        ("sb3_sac", config),
        ("sb3_ppo", ConfigLoader().load(env="dsr")),
        (None, config),
        ("ldcppo", ConfigLoader().load(env="cbess", algo="ldcppo")),
    ):
        assert not preflight(cfg, backend=backend, **args).compatible
    config.env_args["system_name"] = "34Bus"
    assert not preflight(config, backend="sb3_ppo", **args).compatible
