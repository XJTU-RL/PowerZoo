"""公共入口、运行期派生、worker配置和归档的实际契约。"""

from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import json

import pytest

from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass, environment_class
from utils.environment_registry import ENVIRONMENTS, environment_spec
from utils.training_setup import prepare_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", [n for n,s in ENVIRONMENTS.items() if s.interaction == "synchronous"])
def test_worker_uses_one_typed_config_and_registered_wrapper(family, monkeypatch):
    from utils import envs_tools

    spec = environment_spec(family)
    captured = []

    class Consumer:
        def __init__(self, config, rank):
            captured.append((config, rank))

        def seed(self, seed):
            captured.append(seed)

    monkeypatch.setattr(
        import_module(spec.wrapper_module), spec.wrapper_class, Consumer
    )
    cfg = ConfigLoader().load("happo", family)
    envs_tools.make_power_env(family, cfg.env_args, 2, 2000)
    typed, rank = captured[0]
    assert isinstance(typed, environment_class(family)) and rank == 2
    assert typed.seed == 2000 and captured[1] == 2000
    assert cfg.env_args["seed"] != 2000  # worker不能污染原始配置


@pytest.mark.parametrize("family", [n for n,s in ENVIRONMENTS.items() if s.interaction == "synchronous"])
def test_global_dss_engines_are_process_isolated_even_one_worker(family, monkeypatch):
    from utils import envs_tools

    monkeypatch.setattr(envs_tools, "ShareDummyVecEnv", lambda fns: "local")
    monkeypatch.setattr(envs_tools, "ShareSubprocVecEnv", lambda fns: "spawn")
    cfg = ConfigLoader().load("happo", family)
    assert envs_tools.make_train_env(family, 0, 1, cfg.env_args) == "spawn"


def test_runtime_preparation_freezes_seed_and_dataset_once(monkeypatch):
    monkeypatch.setattr("utils.training_setup.secrets.randbelow", lambda _: 123)
    cfg = ConfigLoader().load(
        experiment="configs/experiments/smartgrid_synthetic_smoke.yaml",
        overrides={"seed.seed_specify": False},
    )
    prepare_run(cfg)
    before = cfg.to_dict(), json.dumps(cfg.history)
    prepare_run(cfg)
    assert (cfg.to_dict(), json.dumps(cfg.history)) == before
    assert cfg.env_args["seed"] == cfg.algo_args["seed"]["seed"] == 123
    assert cfg.algo_args["seed"]["seed_specify"] is True
    data = cfg.env_args["data"]
    assert cfg.files[data["path"]] == data["manifest_sha256"]
    assert (
        sha256(Path(data["path"]).read_bytes()).hexdigest() == data["manifest_sha256"]
    )


def test_district_passes_typed_object_without_reparse(monkeypatch):
    from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv

    cfg = ConfigLoader().load("happo", "district_dispatch")
    typed = build_dataclass(environment_class("district_dispatch"), cfg.env_args)
    env = DistrictDispatchEnv.__new__(DistrictDispatchEnv)
    monkeypatch.setattr(
        "utils.config_validation.build_dataclass",
        lambda *a, **k: pytest.fail("重复构造"),
    )
    assert env._parse_config(typed) is typed


def test_focused_presets_inline_one_off_training_and_reuse_shared_smoke():
    import yaml

    for family in ENVIRONMENTS:
        profile = "stackelberg_13bus" if family == "stackelberg" else family
        document = yaml.safe_load(
            (ROOT / f"configs/envs_cfgs/{profile}.yaml").read_text(encoding="utf-8")
        )
        assert not str(document.get("training_ref", "")).endswith("_default")
        if family in ("dsr", "district_dispatch"):
            assert "train" in document
    for path in (ROOT / "configs/experiments").glob("*smoke.yaml"):
        cfg = ConfigLoader().load(experiment=path)
        assert cfg.algo_args["train"]["num_env_steps"] == 240
        assert cfg.algo_args["train"]["episode_length"] == 24


def test_ui_only_advertises_public_configuration_entries():
    from training_ui.core.registry import ALGO_REGISTRY, ENV_REGISTRY

    assert "vvc_single" not in ENV_REGISTRY
    assert "happo" in ALGO_REGISTRY and "dan_happo" not in ALGO_REGISTRY
    for algo in ALGO_REGISTRY:
        ConfigLoader().load(algo, "stackelberg" if algo == "sequential_ppo" else "vvc")
    for env in ENV_REGISTRY:
        ConfigLoader().load("sequential_ppo" if env.startswith("stackelberg") else "happo", env)


def test_archived_files_are_preserved_and_not_importable():
    manifest = json.loads(
        (ROOT / "archive/legacy-training/manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest["files"]:
        archived = ROOT / entry["archived"]
        assert archived.suffix == ".txt"
        assert (
            sha256(archived.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == entry["sha256_lf"]
        )
        if entry["original"] != "tests/envs/smartgrid/test_single_agent.py":
            assert not (ROOT / entry["original"]).exists()


def test_tensorboard_tags_never_become_filesystem_paths(tmp_path):
    import numpy as np
    from utils.tensorboard_writer import ScalarWriter

    writer = ScalarWriter(str(tmp_path / "logs"))
    tag = "group/" + "a" * 300
    writer.add_scalars(tag, {"metric": np.float32(3)}, 1)
    writer.add_scalar("direct", 2, 1)
    writer.export_scalars_to_json(str(tmp_path / "scalars.json"))
    writer.close()
    saved = json.loads((tmp_path / "scalars.json").read_text())
    assert saved[tag + "/metric"][0][-1] == 3
    assert saved["direct"][0][-1] == 2
    assert not any(p.is_dir() for p in (tmp_path / "logs").iterdir())


def test_parallel_shape_error_stops_instead_of_fabricating_zero_transition():
    import numpy as np
    from types import SimpleNamespace
    from envs.env_wrappers import ShareSubprocVecEnv

    env = ShareSubprocVecEnv.__new__(ShareSubprocVecEnv)
    result = ([np.zeros(3)], [np.zeros(3)], np.zeros((2, 1)), [False], [{}], None)
    env.remotes = [SimpleNamespace(recv=lambda: result)]
    with pytest.raises(RuntimeError, match="MARL"):
        env.step_wait()


def test_reward_consumes_shared_typed_config_without_legacy_projection(monkeypatch):
    from types import SimpleNamespace
    from envs.smartgrid.rewards.vvc_reward import VVCReward

    cfg = ConfigLoader().load(
        "happo",
        "smartgrid",
        overrides={"env.reward_weights.control": 3.0, "env.voltage_threshold": 0.03},
    )
    typed = build_dataclass(environment_class("smartgrid"), cfg.env_args)
    monkeypatch.setattr(
        type(typed), "to_info_dict", lambda self: pytest.fail("旧配置投影")
    )
    reward = VVCReward(SimpleNamespace(), typed)
    assert reward.weights["control"] == 3 and reward.voltage_threshold == 0.03
