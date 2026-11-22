"""Opt-in real OpenDSS smoke through scripts/train.py's public entry function."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from envs.powergrid.scenario import COLUMNS, UNITS, VERSION
from utils.config_pipeline import ConfigLoader
from utils.config_validation import ConfigError

EXPERIMENT = "configs/experiments/vvc_34bus_discrete_marl_quickstart.yaml"
ALGORITHMS = ("ippo", "qmix", "vdn", "coma")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_discrete_vvc_configuration(algorithm):
    config = ConfigLoader().load(algo=algorithm, experiment=EXPERIMENT)
    assert config.main_args["algo"] == algorithm
    assert ConfigLoader().load(algorithm, "vvc").env_args["observation_mode"] == "local"
    with pytest.raises(ConfigError, match="局部观测和离散设备动作"):
        ConfigLoader().load(
            algorithm, "vvc", overrides={"env.observation_mode": "global"}
        )
    with pytest.raises(ConfigError, match="局部观测和离散设备动作"):
        ConfigLoader().load(
            algo=algorithm, experiment=EXPERIMENT, overrides={"env.bat_act_num": np.inf}
        )
    with pytest.raises(ConfigError, match="EP.*状态"):
        ConfigLoader().load(
            algo=algorithm, experiment=EXPERIMENT, overrides={"env.state_type": "FP"}
        )


def _training_scenario(directory):
    hours = range(24)
    values = "\n".join(
        f"{hour * 3600},0.55,0.60,0.65,{max(0, np.sin(np.pi * (hour - 6) / 12))},25"
        for hour in hours
    )
    profile = directory / "train_sunny.csv"
    profile.write_text(",".join(COLUMNS) + "\n" + values + "\n", encoding="utf-8")
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": VERSION,
                "units": UNITS,
                "origin": "synthetic_smoke",
                "episodes": [
                    {
                        "id": "smoke_train_day",
                        "split": "train",
                        "scenario": "sunny",
                        "file": profile.name,
                        "sha256": sha256(profile.read_bytes()).hexdigest(),
                        "steps": 24,
                        "step_seconds": 3600,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _inference_trace(runner):
    obs, _, masks = runner.envs.reset()
    trace = []
    for _ in range(24):
        actions = runner.predict(obs, masks)
        np.testing.assert_array_equal(actions, runner.predict(obs, masks))
        for action, available in zip(actions[0, :, 0], masks[0]):
            assert available[int(action)] == 1
        obs, _, rewards, done, infos, masks = runner.envs.step(actions)
        assert not infos[0][0].get("physical_failure")
        trace.append((actions.copy(), rewards.copy()))
        if done.all():
            break
    assert len(trace) == 24 and done.all()
    return trace


@pytest.mark.requires_opendss
@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_discrete_vvc_public_smoke(algorithm, monkeypatch, tmp_path):
    import os

    if os.environ.get("POWERZOO_DISCRETE_VVC_SMOKE") != "1":
        pytest.skip("Set POWERZOO_DISCRETE_VVC_SMOKE=1 on the simulation server")

    from runners import RUNNER_REGISTRY
    from utils.train_cli import main

    factory = RUNNER_REGISTRY[algorithm]
    runs = []

    class RecordedRunner(factory):
        def run(self):
            super().run()
            if algorithm in {"qmix", "vdn"}:
                assert self.total_updates > 0 and self.learner.optimizer.state
                assert self.replay.size >= 24
                assert all(torch.isfinite(p).all() for p in self.learner.parameters)
            else:
                assert all(actor.actor_optimizer.state for actor in self.actor)
                assert all(
                    torch.isfinite(p).all()
                    for actor in self.actor
                    for p in actor.actor.parameters()
                )
            result = self.get_result()["train_physics"]
            assert np.isfinite(result["reward"])
            assert result["steps"] == 24
            runs.append((Path(self.save_dir), _inference_trace(self)))

    monkeypatch.setitem(RUNNER_REGISTRY, algorithm, RecordedRunner)
    options = [
        "--config",
        EXPERIMENT,
        "--algo",
        algorithm,
        "--set",
        f"logger.log_dir={tmp_path / 'runs'}",
        "--set",
        "env.data.source=files",
        "--set",
        f"env.data.path={_training_scenario(tmp_path)}",
    ]
    if algorithm in {"qmix", "vdn"}:
        options += ["--set", "algo.batch_size=8"]
    if algorithm == "ippo":
        options += ["--set", "algo.ppo_epoch=1", "--set", "algo.critic_epoch=1"]
    config = main(options)
    model_dir, expected = runs[0]
    checkpoint = model_dir / "checkpoint_episode_2"
    assert checkpoint.is_dir() and any(checkpoint.glob("*.pt"))
    config.algo_args["train"]["model_dir"] = str(checkpoint)
    config.algo_args["logger"]["log_dir"] = str(tmp_path / "restored")
    runner = factory(config.runner_args(), config.algo_args, config.env_args)
    try:
        for (actions, rewards), (expected_actions, expected_rewards) in zip(
            _inference_trace(runner), expected
        ):
            np.testing.assert_array_equal(actions, expected_actions)
            np.testing.assert_allclose(rewards, expected_rewards)
        if algorithm in {"qmix", "vdn"}:
            assert runner.total_updates > 0 and runner.learner.optimizer.state
        else:
            assert all(actor.actor_optimizer.state for actor in runner.actor)
    finally:
        runner.close()
