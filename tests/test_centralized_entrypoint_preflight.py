"""集中 CLI 的静态与物理检查接点，不启动潮流或优化器。"""

import json
import sys
from types import SimpleNamespace

from gymnasium import spaces
import pytest

from utils import centralized_cli, envs_tools, training_setup
from utils.capabilities import DeviceLayout, preflight
from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError
from utils.single_agent import backends


VVC_SMOKE = "configs/experiments/vvc_13bus_smoke.yaml"
DSR_SMOKE = "configs/experiments/dsr_13bus_smoke.yaml"
CBESS_SMOKE = "configs/experiments/cbess_ldcppo_quickstart.yaml"


def _cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train_centralized.py", *map(str, args)])


def _forbid_prepare(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("静态不兼容配置不得进入 prepare_run")

    monkeypatch.setattr(training_setup, "prepare_run", forbid)


@pytest.mark.parametrize("backend", ["sb3_sac", "sb3_td3"])
def test_stock_vvc_rejects_continuous_backend_before_prepare(backend, monkeypatch, capsys):
    _cli(monkeypatch, "--config", VVC_SMOKE, "--backend", backend)
    _forbid_prepare(monkeypatch)
    with pytest.raises(SystemExit) as error:
        centralized_cli.main()
    assert error.value.code == 2
    assert f"{backend} 需要原生连续设备" in capsys.readouterr().err


@pytest.mark.parametrize("family, experiment", [("vvc", VVC_SMOKE), ("dsr", DSR_SMOKE)])
def test_known_bound_checkpoint_rejects_cross_split_before_prepare(
    family, experiment, tmp_path, monkeypatch, capsys
):
    cfg = ConfigLoader().load(experiment=experiment)
    cfg.env_args["data"]["split"] = "validation"
    target = tmp_path / "validation.environment.yaml"
    cfg.save(target)
    checkpoint = tmp_path / "policy.zip"
    checkpoint.touch()
    checkpoint.with_name("policy.zip.binding.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "sb3_ppo",
                "family": family,
                "scenario": {"split": "train", "sha256": "unverified"},
            }
        ),
        encoding="utf-8",
    )
    _cli(
        monkeypatch, "--config", target, "--output", checkpoint, "--evaluate",
        "--eval-scenario-sha256", "does-not-authorize-cross-split",
    )
    _forbid_prepare(monkeypatch)
    with pytest.raises(SystemExit) as error:
        centralized_cli.main()
    assert error.value.code == 2
    assert "仅支持同场景重载" in capsys.readouterr().err


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_evaluate_requests_policy_reload_without_claiming_legacy_proof(
    legacy, split, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.touch()
    if not legacy:
        checkpoint.with_name("policy.zip.binding.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "backend": "sb3_ppo",
                    "family": "vvc",
                    "scenario": {"split": split, "sha256": "unverified"},
                }
            ),
            encoding="utf-8",
        )
    cfg = ConfigLoader().load(experiment=VVC_SMOKE)
    cfg.env_args["data"]["split"] = split
    target = tmp_path / "target.environment.yaml"
    cfg.save(target)
    _cli(monkeypatch, "--config", target, "--output", checkpoint, "--evaluate")
    checks = []
    actual = centralized_cli._require_centralized_preflight

    def capture(cfg, **kwargs):
        report = actual(cfg, **kwargs)
        checks.append((kwargs["checkpoint"], report.checkpoint_contract))

    class Prepared(Exception):
        pass

    def stop(cfg):
        assert checks == [("policy_reload", "binding_v1")]
        raise Prepared

    monkeypatch.setattr(centralized_cli, "_require_centralized_preflight", capture)
    monkeypatch.setattr(training_setup, "prepare_run", stop)
    with pytest.raises(Prepared):
        centralized_cli.main()


@pytest.mark.parametrize(
    "switch, split_args, expected_split, mode",
    [
        ("--resume", [], "train", "training_resume"),
        ("--evaluate", ["--split", "validation", "--eval-scenario-sha256", "fixture"],
         "validation", "policy_reload"),
    ],
)
def test_cbess_preflight_uses_final_config_and_native_backend(
    switch, split_args, expected_split, mode, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    _cli(
        monkeypatch, "--config", CBESS_SMOKE, "--backend", "ldcppo", "--output",
        checkpoint, switch, *split_args,
    )
    calls = []
    actual = centralized_cli._require_centralized_preflight
    wrapper = SimpleNamespace(close=lambda: calls.append("close"))

    def capture(cfg, **kwargs):
        assert cfg.env_args["split"] == expected_split
        assert kwargs == {"backend": "ldcppo", "checkpoint": mode}
        calls.append("static")
        return actual(cfg, **kwargs)

    def prepare(cfg):
        assert cfg.env_args["split"] == expected_split
        calls.append("prepare")

    def build(family, env_args, rank, seed):
        assert family == "cbess" and env_args["split"] == expected_split
        calls.append("factory")
        return wrapper

    class NativeValidationReached(Exception):
        pass

    def validate_backend(backend, family, env=None):
        if env is not None:
            assert env is wrapper
            calls.append("native")
            raise NativeValidationReached

    monkeypatch.setattr(centralized_cli, "_require_centralized_preflight", capture)
    monkeypatch.setattr(training_setup, "prepare_run", prepare)
    monkeypatch.setattr(envs_tools, "make_power_env", build)
    monkeypatch.setattr(backends, "validate_backend", validate_backend)
    with pytest.raises(NativeValidationReached):
        centralized_cli.main()
    assert calls == ["static", "prepare", "factory", "native", "close"]


def test_cbess_validation_requires_digest_before_prepare(monkeypatch, capsys, tmp_path):
    _cli(
        monkeypatch, "--config", CBESS_SMOKE, "--backend", "ldcppo", "--output",
        tmp_path, "--evaluate", "--split", "validation",
    )
    _forbid_prepare(monkeypatch)
    with pytest.raises(SystemExit) as error:
        centralized_cli.main()
    assert error.value.code == 2
    assert "CBESS validation 需要 --eval-scenario-sha256" in capsys.readouterr().err


def test_nonstandard_vvc_keeps_mixed_devices_and_defers_off_policy_layout():
    cfg = ConfigLoader().load(experiment=VVC_SMOKE)
    cfg.env_args["bat_act_num"] = float("inf")
    assert not preflight(cfg, backend="sb3_sac").compatible
    # 同名自定义目录未必有标准馈线的机械设备，不能猜成混合布局。
    cfg.env_args["dss_folder"] = "/custom/node_systems/13Bus"
    mixed = DeviceLayout(("Capacitor.x", "Storage.y"), ("discrete", "continuous"), False)
    continuous = DeviceLayout(("Storage.y",), ("continuous",), True)
    assert preflight(cfg, backend="sb3_ppo", layout=mixed).compatible
    assert preflight(cfg, backend="sb3_sac").compatible
    assert not preflight(cfg, backend="sb3_sac", layout=mixed).compatible
    assert preflight(cfg, backend="sb3_sac", layout=continuous).compatible


@pytest.mark.parametrize("family", ["vvc", "dsr"])
def test_native_checks_before_central_adapter_and_cleans_failed_wrapper(
    family, monkeypatch
):
    cfg = ConfigLoader().load(env=family)
    action = spaces.Discrete(2)
    entry = (
        {"element": "Capacitor.a", "space": str(action)}
        if family == "vvc" else {"group": "feeders", "choices": [None, {}]}
    )
    calls = []

    def manifest():
        calls.append("layout")
        return {"action_schema": [entry]}

    def close():
        calls.append("close")
        raise RuntimeError("释放错误不应掩盖合同错误")

    wrapper = SimpleNamespace(
        env=object(), action_space=[action],
        observation_space=[spaces.Box(-1, 1, (3,))], get_physics_manifest=manifest,
        close=close,
    )
    monkeypatch.setattr(envs_tools, "make_power_env", lambda *args: wrapper)

    def adapt(core, *, action_space_type):
        assert core is wrapper.env
        assert action_space_type == ("discrete" if family == "dsr" else "continuous")
        calls.append("adapter")
        return wrapper

    monkeypatch.setattr(centralized_cli, "SingleAgentAdapter", adapt)
    assert centralized_cli.build_env(family, cfg, backend="sb3_ppo") is wrapper
    assert calls == ["layout", "adapter"]

    calls.clear()
    wrapper.observation_space = []
    with pytest.raises(ConfigError, match="数量不一致"):
        centralized_cli.build_env(family, cfg, backend="sb3_ppo")
    assert calls == ["layout", "close"]


@pytest.mark.parametrize("family", ["smartgrid", "district_dispatch", "vvc"])
def test_unprofiled_family_or_wrapper_without_layout_preserves_legacy_factory(
    family, monkeypatch
):
    cfg = ConfigLoader().load(env=family)
    wrapper = SimpleNamespace(env=object())
    if family != "vvc":
        wrapper.get_physics_manifest = lambda: pytest.fail("旧环境不受能力目录约束")
    monkeypatch.setattr(envs_tools, "make_power_env", lambda *args: wrapper)
    monkeypatch.setattr(centralized_cli, "SingleAgentAdapter", lambda core, **kwargs: core)
    assert centralized_cli.build_env(family, cfg, backend="sb3_ppo") is wrapper.env
