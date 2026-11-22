"""锁定 IEEE34 原生连续储能、离策略更新与模型重放。"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.config_pipeline import ConfigLoader


EXPERIMENTS = Path("configs/experiments")
CONTINUOUS = EXPERIMENTS / "vvc_34bus_continuous_storage_quickstart.yaml"


def test_smoke_presets_use_the_standard_config_loader():
    for algo in ("maddpg", "matd3"):
        config = ConfigLoader().load(algo=algo, experiment=CONTINUOUS)
        assert config.env_args["continuous_storage_only"]
        assert config.env_args["bat_act_num"] == np.inf
        assert not config.algo_args["eval"]["use_eval"]


@pytest.fixture
def train_manifest(tmp_path):
    from envs.powergrid.scenario import COLUMNS, UNITS, VERSION

    profile = tmp_path / "train.csv"
    profile.write_text(
        ",".join(COLUMNS)
        + "\n"
        + "".join(f"{36000 + t * 300},0.8,0.8,0.85,0.5,30\n" for t in range(4)),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": VERSION,
                "units": UNITS,
                "origin": "synthetic-train-smoke",
                "episodes": [
                    {
                        "id": "train_smoke",
                        "scenario": "smoke",
                        "split": "train",
                        "steps": 4,
                        "step_seconds": 300,
                        "file": profile.name,
                        "sha256": sha256(profile.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(manifest)


def _config(algo, experiment, train_manifest, tmp_path, **overrides):
    return ConfigLoader().load(
        algo=algo,
        experiment=experiment,
        overrides={
            "env.data.source": "files",
            "env.data.path": train_manifest,
            "env.data.noise": 0.025,
            "logger.log_dir": str(tmp_path),
            **overrides,
        },
    )


@pytest.mark.requires_opendss
def test_storage_only_uses_unquantized_native_dispatch(train_manifest, tmp_path):
    from envs.vvc.vvc.env_register import make_base_env
    from envs.vvc.vvc.vvc_config import VVCConfig

    loaded = _config("maddpg", CONTINUOUS, train_manifest, tmp_path)
    env = make_base_env(VVCConfig.from_env_args(loaded.env_args))
    try:
        mechanical_state = (env.obs["cap_statuses"], env.obs["reg_statuses"])
        actions = [0.123, -0.457]
        _, _, _, info = env.step(actions)
        assert info["dss_converged"]
        assert (env.obs["cap_statuses"], env.obs["reg_statuses"]) == mechanical_state
        trace = info["control_trace"]
        assert list(trace) == ["Storage.batt1", "Storage.batt2"]
        for command, entry in zip(actions, trace.values()):
            for field in ("requested", "applied"):
                assert entry[field]["kw"] == pytest.approx(command * 100, abs=1e-4)
            assert entry["measured"]["kw"] * command > 0
        assert (
            env.circuit.batteries["batt1"].soc
            < 0.5
            < env.circuit.batteries["batt2"].soc
        )
        env.reset(load_profile_idx=0)
        _, _, _, next_info = env.step([0.131, -0.463])
        next_trace = next_info["control_trace"]
        assert (
            next_trace["Storage.batt1"]["measured"]["kw"]
            > trace["Storage.batt1"]["measured"]["kw"]
        )
        assert (
            next_trace["Storage.batt2"]["measured"]["kw"]
            < trace["Storage.batt2"]["measured"]["kw"]
        )
    finally:
        env.close()


def _replay(runner):
    observations, _, masks = runner.envs.reset()
    frames = []
    for _ in range(4):
        actions = runner.get_actions(observations, masks, add_random=False)
        observations, _, rewards, dones, infos, masks = runner.envs.step(actions)
        for worker, worker_actions in zip(infos, actions):
            info = worker[0]
            assert info["dss_converged"] and info["scenario_id"] == "train_smoke"
            trace = info["control_trace"]
            assert list(trace) == ["Storage.batt1", "Storage.batt2"]
            for entry, (command,) in zip(trace.values(), worker_actions):
                assert entry["requested"]["kw"] == pytest.approx(command * 100)
        frames.append((actions, observations.copy(), rewards, dones))
    assert dones.all()
    return frames


@pytest.mark.requires_opendss
@pytest.mark.parametrize("algo", ["maddpg", "matd3"])
def test_native_runner_updates_reloads_and_replays(algo, train_manifest, tmp_path):
    from runners import RUNNER_REGISTRY

    config = _config(algo, CONTINUOUS, train_manifest, tmp_path / "source")
    runner_class = RUNNER_REGISTRY[algo]
    runner = runner_class(config.runner_args(), config.algo_args, config.env_args)
    try:
        initial = [
            next(actor.actor.parameters()).detach().clone() for actor in runner.actor
        ]
        optimizer_steps = []
        for agent_id, actor in enumerate(runner.actor):
            actor.actor_optimizer.register_step_post_hook(
                lambda *args, agent_id=agent_id: optimizer_steps.append(agent_id)
            )
        runner.run()
        assert all(
            torch.isfinite(param).all()
            for actor in runner.actor
            for param in actor.actor.parameters()
        )
        assert all(
            not torch.equal(before, next(actor.actor.parameters()))
            for before, actor in zip(initial, runner.actor)
        )
        assert runner.total_it == 4
        assert runner.num_agents == 2
        assert runner.buffer.cur_size == 12
        assert all(space.shape == (1,) for space in runner.action_spaces)
        assert all(
            (np.abs(action[:12]) <= 1).all() for action in runner.buffer.actions
        )
        assert any(
            (action[:12] * 16 % 1 != 0).any() for action in runner.buffer.actions
        )
        assert optimizer_steps == [0, 1] * (2 if algo == "matd3" else 4)
        checkpoint = str(runner.save_dir)
        expected = _replay(runner)
    finally:
        runner.close()

    config = _config(
        algo,
        CONTINUOUS,
        train_manifest,
        tmp_path / "reload",
        **{"train.model_dir": checkpoint},
    )
    restored = runner_class(config.runner_args(), config.algo_args, config.env_args)
    try:
        for original, reloaded in zip(expected, _replay(restored)):
            for lhs, rhs in zip(original, reloaded):
                np.testing.assert_array_equal(lhs, rhs)
    finally:
        restored.close()


def test_legacy_ma_warmup_still_delegates_discrete_masks(monkeypatch):
    from runners.off_policy_base_runner import OffPolicyBaseRunner
    from runners.off_policy_ma_runner import OffPolicyMARunner

    runner = object.__new__(OffPolicyMARunner)
    runner.args = {"env": "smartgrid"}
    runner.env_args = {}
    masks = np.array([[[0, 1]]])

    def sample(self, available_actions):
        assert self is runner and available_actions is masks
        return np.array([[[1]]])

    monkeypatch.setattr(OffPolicyBaseRunner, "sample_actions", sample)
    np.testing.assert_array_equal(runner.sample_actions(masks), [[[1]]])


def test_storage_policy_binding_rejects_layout_change_before_model_read(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from runners.off_policy_base_runner import OffPolicyBaseRunner
    from runners.off_policy_ma_runner import OffPolicyMARunner

    runner = object.__new__(OffPolicyMARunner)
    runner.args = {"env": "vvc"}
    runner.env_args = {"continuous_storage_only": True}
    runner.algo_args = {"train": {"model_dir": tmp_path}}
    runner.envs = SimpleNamespace(
        get_physics_manifest=lambda: {"action_schema_sha256": "different"}
    )
    (tmp_path / "physics.manifest.json").write_text(
        json.dumps({"action_schema_sha256": "saved-layout"}), encoding="utf-8"
    )

    def must_not_restore(*_):
        pytest.fail("物理动作映射不符，不得读取策略")

    monkeypatch.setattr(OffPolicyBaseRunner, "restore", must_not_restore)
    with pytest.raises(ValueError, match="action_schema_sha256"):
        runner.restore()
