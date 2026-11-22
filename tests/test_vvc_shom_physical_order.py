"""锁定 IEEE34 SHOM 设备响应、优化顺序与模型重放。"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.actors.shom_order import physical_agent_order, physical_agent_scores
from utils.config_pipeline import ConfigLoader


EXPERIMENTS = Path("configs/experiments")
SHOM = EXPERIMENTS / "vvc_34bus_shom_native_quickstart.yaml"


def test_native_columns_bind_by_device_and_ties_use_stable_id():
    device_ids = ["Storage.z", "Storage.a", "Capacitor.c"]
    payload = {
        "mode": "native",
        "scenario_id": "mock",
        "time_s": 300,
        "model_sha256": "model",
        "action_schema_sha256": "actions",
        "control_basis": "one_normalized_device_command",
        "nodes": ["bus.1", "bus.2"],
        "device_columns": ["Capacitor.c", "Storage.a", "Storage.z"],
        "matrix": [[-4, 1, 0], [1, 0, 1]],
    }
    info = {"scenario_id": "mock", "time_s": 300, "voltage_sensitivity": payload}
    scores = physical_agent_scores(
        "native", [(0, info)], device_ids, "model", "actions"
    )
    np.testing.assert_array_equal(scores, [1, 1, 5])
    assert physical_agent_order(scores, device_ids=device_ids) == [2, 1, 0]
    assert physical_agent_order(scores, reverse=True, device_ids=device_ids) == [
        1,
        0,
        2,
    ]
    with pytest.raises(ValueError, match="does not match"):
        physical_agent_scores("native", [(0, info)], device_ids, "model", "wrong")
    payload["device_columns"] = ["Storage.z"] * 3
    with pytest.raises(ValueError, match="one normalized response"):
        physical_agent_scores("native", [(0, info)], device_ids, "model", "actions")


def test_paper_retains_reward_weighted_absolute_rows():
    devices = ["Capacitor.c", "RegControl.r"]
    steps = []
    for time, reward, rows in [(300, -1, [4, 3, 1000]), (600, -3, [1, 9, 1000])]:
        metadata = {"scenario_id": "mock", "time_s": time}
        steps.append(
            (
                reward,
                {
                    **metadata,
                    "voltage_sensitivity": {
                        **metadata,
                        "mode": "paper",
                        "model_sha256": "model",
                        "action_schema_sha256": "actions",
                        "nodes": ["cap.1", "reg.1", "load.1"],
                        "device_nodes": dict(zip(devices, ["cap.1", "reg.1"])),
                        "matrix": np.diag(rows),
                    },
                },
            )
        )
    scores = physical_agent_scores("paper", steps, devices, "model", "actions")
    np.testing.assert_allclose(scores, [1.75, 7.5])
    assert physical_agent_order(scores) == [1, 0]
    assert physical_agent_order(scores, reverse=True) == [0, 1]


def test_smoke_presets_use_the_standard_config_loader():
    config = ConfigLoader().load(experiment=SHOM)
    assert config.algo_args["algo"]["sensitivity_mode"] == "native"
    assert config.env_args["sensitivity_measurement"] == "native"
    assert (
        ConfigLoader().load(algo="shom").algo_args["algo"]["sensitivity_mode"]
        == "legacy"
    )


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
def test_native_columns_equal_independent_device_steps_and_leave_live_state_intact(
    train_manifest, tmp_path
):
    from envs.vvc.vvc.env_register import make_base_env
    from envs.vvc.vvc.vvc_config import VVCConfig

    loaded = _config("shom", SHOM, train_manifest, tmp_path)
    cfg = VVCConfig.from_env_args(loaded.env_args)
    native = make_base_env(cfg)
    ordinary = make_base_env(replace(cfg, sensitivity_measurement="disabled"))
    try:
        device_ids = [
            row["element"] for row in native.get_physics_manifest()["action_schema"]
        ]
        commands = [int(space.n // 2) for space in native.agent_spaces]
        commands[device_ids.index("Storage.batt1")] += 1
        for step in range(2):
            obs, reward, done, info = native.step(commands)
            other_obs, other_reward, other_done, other_info = ordinary.step(commands)
            np.testing.assert_array_equal(obs, other_obs)
            assert (reward, done) == (other_reward, other_done)
            assert (
                list(info["control_trace"])
                == device_ids
                == list(other_info["control_trace"])
            )
            assert info["control_trace"] == other_info["control_trace"]
            assert info["time_s"] == other_info["time_s"]
            assert info["battery_avg_soc"] == other_info["battery_avg_soc"]
            assert "voltage_sensitivity" not in other_info
            sample = info.get("voltage_sensitivity")
            assert sample is not None, info.get("voltage_sensitivity_unavailable")
            assert sample["device_columns"] == sorted(device_ids)
            assert sample["operating_time_s"] + info["step_seconds"] == info["time_s"]
            if step == 0:
                measured = sample
                baseline = {
                    f"{bus}.{phase}": value
                    for bus, phases in ordinary.physical_state.voltages_pu.items()
                    for phase, value in phases.items()
                }

        for index, device in enumerate(measured["device_columns"]):
            meta = measured["perturbations"][device]
            assert meta["measured_before"] != pytest.approx(meta["measured_after"])
            alternative = commands.copy()
            field, request = meta["field"], meta["requested"]
            position = device_ids.index(device)
            if field == "state":
                alternative[position] = request
            elif field == "tap":
                reg = ordinary.circuit.regulators[device.split(".", 1)[1]]
                alternative[position] = round(
                    (request - reg.mintap)
                    / (reg.maxtap - reg.mintap)
                    * (cfg.reg_act_num - 1)
                )
            else:
                bat = ordinary.circuit.batteries[device.split(".", 1)[1]]
                alternative[position] = round(
                    (request / bat.max_kw + 1) * (cfg.bat_act_num - 1) / 2
                )
            ordinary.reset(load_profile_idx=0)
            _, _, _, info = ordinary.step(alternative)
            assert info["dss_converged"]
            variant = {
                f"{bus}.{phase}": value
                for bus, phases in ordinary.physical_state.voltages_pu.items()
                for phase, value in phases.items()
            }
            expected = (
                np.array([variant[node] - baseline[node] for node in measured["nodes"]])
                / meta["normalized_delta"]
            )
            assert np.linalg.norm(expected) > 1e-7
            np.testing.assert_allclose(
                measured["matrix"][:, index], expected, rtol=1e-4, atol=1e-6
            )
    finally:
        native.close()
        ordinary.close()


def _replay(runner):
    runner.prep_rollout()
    observations, _, masks = runner.envs.reset()
    frames = []
    for _ in range(4):
        actions = np.stack(
            [
                actor.act(
                    observations[:, i],
                    np.zeros((1, runner.recurrent_n, runner.rnn_hidden_size)),
                    np.ones((1, 1)),
                    runner._agent_available_actions(masks, i),
                    deterministic=True,
                )[0]
                .detach()
                .cpu()
                .numpy()
                for i, actor in enumerate(runner.actor)
            ],
            axis=1,
        )
        observations, _, rewards, dones, infos, masks = runner.envs.step(actions)
        assert all(worker[0]["dss_converged"] for worker in infos)
        frames.append((actions, observations.copy(), rewards, dones))
        assert all(worker[0]["scenario_id"] == "train_smoke" for worker in infos)
    assert dones.all()
    return frames


@pytest.mark.requires_opendss
@pytest.mark.parametrize("mode", ["native", "paper", "legacy"])
def test_shom_updates_reloads_and_replays(mode, train_manifest, tmp_path):
    from runners import RUNNER_REGISTRY

    overrides = {
        "algo.sensitivity_mode": mode,
        "env.sensitivity_measurement": "disabled" if mode == "legacy" else mode,
    }
    config = _config("shom", SHOM, train_manifest, tmp_path / "source", **overrides)
    runner_class = RUNNER_REGISTRY["shom"]
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
        if mode == "legacy":
            assert runner.last_update_order == list(range(runner.num_agents))
        else:
            ids = list(runner.last_sensitivity_scores)
            assert runner.last_update_order == physical_agent_order(
                list(runner.last_sensitivity_scores.values()),
                device_ids=ids if mode == "native" else None,
            )
            assert runner.last_update_order != list(range(runner.num_agents))
        updates_per_actor = (
            config.algo_args["algo"]["actor_num_mini_batch"]
            * config.algo_args["algo"]["ppo_epoch"]
        )
        assert optimizer_steps == [
            agent_id
            for agent_id in runner.last_update_order
            for _ in range(updates_per_actor)
        ]
        checkpoint = str(runner.save_dir)
        expected = _replay(runner)
    finally:
        runner.close()

    config = _config(
        "shom",
        SHOM,
        train_manifest,
        tmp_path / "reload",
        **{"train.model_dir": checkpoint, **overrides},
    )
    restored = runner_class(config.runner_args(), config.algo_args, config.env_args)
    try:
        for original, reloaded in zip(expected, _replay(restored)):
            for lhs, rhs in zip(original, reloaded):
                np.testing.assert_array_equal(lhs, rhs)
    finally:
        restored.close()


def test_paper_polar_jacobian_matches_finite_difference():
    from envs.vvc.voltage_sensitivity import polar_power_jacobian

    admittance = np.array([[1 - 3j, -1 + 3j], [-1 + 3j, 1 - 3j]])
    voltage = np.array([1.0 + 0j, 0.96 * np.exp(-0.04j)])
    blocks = polar_power_jacobian(admittance, voltage, np.ones(2), [1])

    def power(values):
        return values[1] * (admittance @ values).conj()[1]

    delta = 1e-6
    angle_plus, angle_minus = voltage.copy(), voltage.copy()
    angle_plus[1] *= np.exp(1j * delta)
    angle_minus[1] *= np.exp(-1j * delta)
    dangle = (power(angle_plus) - power(angle_minus)) / (2 * delta)
    mag_plus, mag_minus = voltage.copy(), voltage.copy()
    mag_plus[1] += delta * np.exp(-0.04j)
    mag_minus[1] -= delta * np.exp(-0.04j)
    dmag = (power(mag_plus) - power(mag_minus)) / (2 * delta)

    np.testing.assert_allclose(
        [block.item() for block in blocks],
        [dmag.imag, dangle.imag, dangle.real, dmag.real],
    )
