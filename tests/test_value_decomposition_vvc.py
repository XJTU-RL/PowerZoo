"""QMIX/VDN, masks and whole-episode retention using synthetic vector frames."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import runners.value_decomposition_runner as module
from runners.value_decomposition_replay import EpisodeReplay
from utils.config_pipeline import ConfigLoader


class DeviceVector:
    action_space = (gym.spaces.Discrete(2), gym.spaces.Discrete(33))
    observation_space = (gym.spaces.Box(-10, 10, (2,)), gym.spaces.Box(-10, 10, (3,)))
    share_observation_space = (gym.spaces.Box(-100, 100, (4,)),) * 2

    def get_physics_manifest(self):
        return {"task": "synthetic_vvc", "action_schema_sha256": "two-devices"}

    def close(self):
        pass


def frame(t, *, legal=True):
    obs = [[np.array([t, t + 2]), np.array([2, -t, 1])]]
    shared = np.full((1, 2, 4), 20 + t, dtype=np.float32)
    cap, tap = np.zeros(2), np.zeros(33)
    cap[1] = tap[32] = int(legal)
    return obs, shared, [[cap, tap]]


def make_runner(algo, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "make_train_env", lambda *args: DeviceVector())
    conf = ConfigLoader().load(
        algo=algo,
        env="vvc",
        overrides={
            "env.observation_mode": "local",
            "env.max_episode_steps": 2,
            "device.cuda": False,
            "device.torch_threads": 1,
            "eval.use_eval": False,
            "logger.log_dir": str(tmp_path),
            "algo.batch_size": 2,
            "algo.buffer_size": 4,
        },
    )
    return module.ValueDecompositionRunner(
        conf.runner_args(), conf.algo_args, conf.env_args
    )


def test_episode_replay_stages_workers_and_evicts_whole_episodes():
    replay = EpisodeReplay(4, 2)
    replay.append(0, "episode0-step1", ended=False)
    assert replay.size == 0
    replay.append(1, "episode1", ended=True)
    replay.append(0, "episode0-step2", ended=True)
    assert set(replay.sample(3)) == {"episode0-step1", "episode0-step2", "episode1"}

    replay.append(1, "episode2-step1", ended=False)
    replay.append(1, "episode2-step2", ended=False)
    replay.append(1, "episode2-step3", ended=True)
    assert replay.size == 3
    assert set(replay.sample(3)) == {
        "episode2-step1",
        "episode2-step2",
        "episode2-step3",
    }


@pytest.mark.parametrize("algo", ["qmix", "vdn"])
def test_masked_device_update_and_restore(monkeypatch, tmp_path, algo):
    runner = make_runner(algo, monkeypatch, tmp_path)
    try:
        current = frame(1)
        for terminal in (True, False):
            final_obs, final_state, final_masks = frame(
                2 if terminal else 3, legal=not terminal
            )
            info = {
                "TimeLimit.truncated": True,
                "terminated": terminal,
                "physical_failure": terminal,
                "original_obs": final_obs[0],
                "original_state": final_state[0],
                "original_avail_actions": final_masks[0],
            }
            actions = runner.predict(current[0], current[2], explore=True)
            np.testing.assert_array_equal(actions, [[[1], [32]]])
            obs, state, available = frame(0)
            runner._insert(
                current,
                actions,
                (
                    obs,
                    state,
                    np.full((1, 2, 1), -2.0, dtype=np.float32),
                    np.ones((1, 2), dtype=bool),
                    [[info, info]],
                    available,
                ),
            )
        batch = sorted(runner.replay.sample(2), key=lambda item: not item.terminated)
        terminal, truncated = batch
        assert terminal.terminated and not truncated.terminated
        assert not terminal.following.available[1].any()
        np.testing.assert_array_equal(truncated.following.state, np.full(4, 23))
        np.testing.assert_array_equal(truncated.following.obs[0], [3, 5])
        assert [len(mask) for mask in truncated.current.available] == [2, 33]

        learner = runner.learner
        target_inputs = []
        handle = learner.target_mixer.register_forward_hook(
            lambda _module, args, _output: target_inputs.append(args[1].cpu().numpy())
        )
        local = [np.stack([item.current.obs[i] for item in batch]) for i in range(2)]
        before = [
            policy(obs).detach().clone() for policy, obs in zip(learner.policies, local)
        ]
        mixer_before = [p.detach().clone() for p in learner.mixer.parameters()]
        metrics = learner.train_batch(batch)
        handle.remove()
        assert np.isfinite(metrics["td_loss"])
        np.testing.assert_array_equal(target_inputs, [[truncated.following.state]])
        assert all(
            not torch.equal(initial, policy(obs))
            for initial, policy, obs in zip(before, learner.policies, local)
        )
        q = torch.tensor([[2.0, 3.0]], requires_grad=True)
        state = torch.ones(1, 4)
        if algo == "qmix":
            assert any(
                not torch.equal(old, current)
                for old, current in zip(mixer_before, learner.mixer.parameters())
            )
            (gradient,) = torch.autograd.grad(learner.mixer(q, state).sum(), q)
            assert (gradient >= 0).all() and (gradient > 0).any()
            assert not torch.equal(learner.mixer(q, state), learner.mixer(q, state * 2))
        else:
            torch.testing.assert_close(learner.mixer(q, state), torch.tensor([[5.0]]))
            torch.testing.assert_close(
                learner.mixer(q, state * 2), torch.tensor([[5.0]])
            )

        with pytest.raises(ValueError, match="at least one"):
            runner.predict(current[0], frame(0, legal=False)[2])
        runner.total_updates = 1
        runner.save(Path(runner.save_dir))
        restored = make_runner(algo, monkeypatch, tmp_path / "restored")
        try:
            restored.restore(runner.save_dir)
            assert restored.total_updates == 1 and restored.learner.optimizer.state
            for original, new, obs in zip(
                learner.policies, restored.learner.policies, local
            ):
                torch.testing.assert_close(original(obs), new(obs))
            unmasked = [[np.ones(2), np.ones(33)]]
            np.testing.assert_array_equal(
                runner.predict(current[0], unmasked),
                restored.predict(current[0], unmasked),
            )
        finally:
            restored.close()
    finally:
        runner.close()
