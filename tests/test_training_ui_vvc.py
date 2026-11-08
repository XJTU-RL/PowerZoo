"""The VVC editor shares the same system references and presets as the CLI."""

import pytest

from training_ui.core.config_builder import PROJECT_ROOT, resolve_config
from training_ui.core.vvc_config import (
    STANDARD_FEEDERS,
    YAML_FEEDER,
    describe_feeder,
    load_short_preset,
    select_feeder,
)
from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError


@pytest.mark.parametrize(
    "original, expected",
    [
        ({"system_ref": "13Bus"}, {"system_ref": "34Bus"}),
        ({"env": {"system_ref": "13Bus"}}, {"env": {"system_ref": "34Bus"}}),
        ({}, {"system_ref": "34Bus"}),
    ],
)
def test_select_feeder_preserves_other_overrides(original, expected):
    original.setdefault("env", {})["max_episode_steps"] = 12
    original["train"] = {"num_env_steps": 99}
    expected.setdefault("env", {})["max_episode_steps"] = 12
    expected["train"] = {"num_env_steps": 99}
    assert select_feeder(original, "34Bus") == expected


def test_select_feeder_rejects_conflicting_references():
    overrides = {"system_ref": "13Bus", "env": {"system_ref": "34Bus"}}
    with pytest.raises(ConfigError, match="system_ref.*不一致"):
        select_feeder(overrides, "123Bus")
    assert overrides == {"system_ref": "13Bus", "env": {"system_ref": "34Bus"}}


def test_vvc_editor_accepts_implicit_standard_feeder():
    config = resolve_config("happo", "vvc", "test", {})
    summary = describe_feeder({}, config.env_args, "13Bus")
    assert "实际系统 13Bus" in summary
    assert str((PROJECT_ROOT / "node_systems" / "13Bus" / "network.dss").resolve()) in summary


def test_vvc_editor_does_not_label_custom_dss_as_standard(tmp_path):
    (tmp_path / "custom.dss").write_text("clear", encoding="utf-8")
    overrides = {
        "system_ref": "13Bus",
        "env": {"dss_folder": str(tmp_path), "dss_file": "custom.dss"},
    }
    config = resolve_config("happo", "vvc", "test", overrides)
    with pytest.raises(ConfigError, match="标准馈线 13Bus 不一致"):
        describe_feeder(overrides, config.env_args, "13Bus")
    summary = describe_feeder(overrides, config.env_args, YAML_FEEDER)
    assert "（由 YAML 指定）" in summary
    assert str((tmp_path / "custom.dss").resolve()) in summary


@pytest.mark.parametrize("feeder", STANDARD_FEEDERS)
def test_vvc_short_presets_match_the_cli(feeder):
    algo, overrides = load_short_preset(feeder)
    ui = resolve_config(algo, "vvc", "test", overrides)
    cli = ConfigLoader(PROJECT_ROOT).load(
        experiment=f"configs/experiments/vvc_{feeder.lower()}_smoke.yaml",
        exp_name="test",
    )
    assert ui.to_dict() == cli.to_dict()
    assert f"实际系统 {feeder}" in describe_feeder(overrides, ui.env_args, feeder)
