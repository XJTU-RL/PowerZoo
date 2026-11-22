"""IPPO contracts using a scripted vector environment with no physical solver."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import runners.on_policy_base_runner as base
from runners.ippo_runner import IPPORunner
from utils.config_pipeline import ConfigLoader


class ScriptedEnv:
    observation_space = (
        gym.spaces.Box(-10, 10, (3,)),
        gym.spaces.Box(-10, 10, (3,)),
    )
    share_observation_space = (gym.spaces.Box(-10, 10, (5,)),) * 2
    action_space = (gym.spaces.Discrete(2), gym.spaces.Discrete(5))

    @staticmethod
    def frame(t):
        obs = np.array([[[t, -t, 0], [2, t, -t]]], dtype=np.float32)
        state = np.full((1, 2, 5), 99, dtype=np.float32)
        masks = [[np.array([0, 1]), np.array([0, 0, 0, 0, 1])]]
        return obs, state, masks

    def reset(self):
        self.t = 0
        return self.frame(self.t)

    def step(self, actions):
        np.testing.assert_array_equal(actions, [[[1], [4]]])
        self.t += 1
        obs, state, masks = self.frame(self.t)
        ended = self.t == 2
        info = {"terminated": ended, "physical_failure": ended}
        return (
            obs,
            state,
            np.full((1, 2, 1), -self.t, dtype=np.float32),
            np.full((1, 2), ended),
            [[info.copy(), info.copy()]],
            masks,
        )

    def close(self):
        pass


def build_runner(monkeypatch, tmp_path, model_dir=None):
    env = ScriptedEnv()
    monkeypatch.setattr(base, "make_train_env", lambda *args: env)
    monkeypatch.setattr(base, "get_num_agents", lambda *args: 2)
    conf = ConfigLoader().load(
        algo="ippo",
        env="vvc",
        overrides={
            "env.observation_mode": "local",
            "device.cuda": False,
            "device.torch_threads": 1,
            "train.n_rollout_threads": 1,
            "train.episode_length": 2,
            "train.num_env_steps": 2,
            "train.model_dir": None if model_dir is None else str(model_dir),
            "eval.use_eval": False,
            "logger.log_dir": str(tmp_path),
            "algo.ppo_epoch": 1,
            "algo.critic_epoch": 1,
        },
    )
    return IPPORunner(conf.runner_args(), conf.algo_args, conf.env_args)


def test_ippo_local_value_updates_and_checkpoint_replay(monkeypatch, tmp_path):
    runner = build_runner(monkeypatch, tmp_path)
    try:
        runner.warmup()
        value_parameters = [
            next(c.critic.parameters()).detach().clone() for c in runner.critics
        ]
        assert runner.actor[0].actor is not runner.actor[1].actor
        assert runner.critics[0].critic is not runner.critics[1].critic
        assert runner.value_normalizers[0] is not runner.value_normalizers[1]
        for step in range(2):
            values, actions, log_probs, states, critic_states = runner.collect(step)
            runner.insert(
                (
                    *runner.envs.step(actions),
                    values,
                    actions,
                    log_probs,
                    states,
                    critic_states,
                )
            )
        for actor, buffer in zip(runner.actor_buffer, runner.critic_buffers):
            np.testing.assert_array_equal(buffer.share_obs, actor.obs)
        runner.compute()
        for buffer in runner.critic_buffers:
            assert buffer.returns[1, 0, 0] == pytest.approx(-2)
        actor_info, critic_info = runner.train()
        assert len(actor_info) == 2
        assert np.isfinite(critic_info["value_loss"])
        assert all(actor.actor_optimizer.state for actor in runner.actor)
        for old, critic in zip(value_parameters, runner.critics):
            assert not torch.equal(old, next(critic.critic.parameters()))
        runner.save(episode=1)
        model_dir = Path(runner.save_dir) / "checkpoint_episode_1"
        runner.warmup()
        values, actions, *_ = runner.collect(0)

        restored = build_runner(monkeypatch, tmp_path / "restore", model_dir)
        try:
            restored.warmup()
            restored_values, restored_actions, *_ = restored.collect(0)
            np.testing.assert_array_equal(actions, restored_actions)
            np.testing.assert_allclose(values, restored_values)
            assert all(actor.actor_optimizer.state for actor in restored.actor)
            assert all(critic.critic_optimizer.state for critic in restored.critics)
        finally:
            restored.close()
    finally:
        runner.close()


def test_ippo_time_limit_bootstraps_final_local_observation(monkeypatch, tmp_path):
    runner = build_runner(monkeypatch, tmp_path)
    try:
        runner.warmup()
        for i, (critic, normalizer) in enumerate(
            zip(runner.critics, runner.value_normalizers)
        ):
            monkeypatch.setattr(
                critic,
                "get_values",
                lambda obs, states, masks, i=i: (
                    torch.as_tensor(obs[:, i : i + 1]),
                    torch.as_tensor(states),
                ),
            )
            monkeypatch.setattr(normalizer, "denormalize", lambda values: values)
        for step in range(2):
            values, actions, log_probs, states, critic_states = runner.collect(step)
            obs, share, rewards, dones, infos, masks = runner.envs.step(actions)
            if step == 1:
                infos[0][0] = {
                    "TimeLimit.truncated": True,
                    "bad_transition": True,
                    "original_obs": [np.array([3, 0]), np.array([0, 7, 0])],
                }
                obs, share, masks = runner.envs.reset()
            runner.insert(
                (
                    obs,
                    share,
                    rewards,
                    dones,
                    infos,
                    masks,
                    values,
                    actions,
                    log_probs,
                    states,
                    critic_states,
                )
            )
        runner.compute()
        for buffer, expected in zip(runner.critic_buffers, [3, 7]):
            assert buffer.returns[1, 0, 0] == pytest.approx(
                -2 + buffer.gamma * expected
            )
            assert buffer.rewards[1, 0, 0] == -2
    finally:
        runner.close()
