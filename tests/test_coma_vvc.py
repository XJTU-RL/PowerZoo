"""Counterfactual action values and training contracts without a physical solver."""

from pathlib import Path

import numpy as np
import torch

import runners.on_policy_base_runner as base
from algorithms.critics.coma_critic import counterfactual_advantage, lambda_targets
from runners.coma_runner import COMARunner
from tests.test_ippo_runner import ScriptedEnv
from utils.config_pipeline import ConfigLoader


class COMAEnv(ScriptedEnv):
    @staticmethod
    def frame(t):
        obs, _, _ = ScriptedEnv.frame(t)
        state = np.array([[[5 + t, 1, 3, 4, 5]] * 2], dtype=np.float32)
        masks = [[np.ones(2), np.array([1, 0, 1, 0, 1])]]
        return obs, state, masks

    def step(self, actions):
        assert actions[0, 0, 0] in (0, 1)
        assert actions[0, 1, 0] in (0, 2, 4)
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


def _runner(monkeypatch, directory, model_dir=None):
    monkeypatch.setattr(base, "make_train_env", lambda *args: COMAEnv())
    monkeypatch.setattr(base, "get_num_agents", lambda *args: 2)
    config = ConfigLoader().load(
        algo="coma",
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
            "logger.log_dir": str(directory),
        },
    )
    return COMARunner(config.runner_args(), config.algo_args, config.env_args)


def test_coma_legal_counterfactual_and_terminal_bootstrap():
    q = torch.tensor([[1.0, 99.0, 5.0]])
    advantage = counterfactual_advantage(
        q,
        torch.tensor([[2]]),
        torch.tensor([[0.25, 0.5, 0.25]]),
        torch.tensor([[1, 0, 1]]),
    )
    torch.testing.assert_close(advantage, torch.tensor([[2.0]]))
    rewards = torch.tensor([[[1.0]], [[-10.0]]])
    alive = torch.tensor([[[1.0]], [[0.0]]])
    next_q = torch.tensor([[[4.0]], [[1000.0]]])
    targets = lambda_targets(rewards, alive, next_q, 0.5, 0.5)
    torch.testing.assert_close(targets, torch.tensor([[[-0.5]], [[-10.0]]]))


def test_coma_real_update_and_restore_on_synthetic_frames(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    try:
        runner.warmup()
        states = torch.tensor([[5.0, 1.0, 3.0, 4.0, 5.0]] * 3)
        actions = torch.tensor([[0, 0], [1, 0], [0, 4]])
        q = runner.critic(states, actions)
        assert [head.shape[1] for head in q] == [2, 5]
        torch.testing.assert_close(q[0][0], q[0][1])
        torch.testing.assert_close(q[1][0], q[1][2])
        assert not torch.equal(q[0][0], q[0][2])
        assert not torch.equal(q[1][0], q[1][1])

        before = [
            [p.detach().clone() for p in actor.actor.parameters()]
            for actor in runner.actor
        ]
        for step in range(2):
            values, act, log_probs, rnn, critic_rnn = runner.collect(step)
            runner.insert(
                (*runner.envs.step(act), values, act, log_probs, rnn, critic_rnn)
            )
        runner.compute()
        torch.testing.assert_close(runner.targets[-1], torch.tensor([-2.0, -2.0]))
        actor_info, critic_info = runner.train()
        assert len(actor_info) == 2 and np.isfinite(critic_info["td_loss"])
        assert not torch.equal(q[0], runner.critic(states, actions)[0])
        assert all(
            any(
                not torch.equal(a, b)
                for a, b in zip(previous, actor.actor.parameters())
            )
            for previous, actor in zip(before, runner.actor)
        )
        runner.save()
        restored = _runner(monkeypatch, tmp_path / "restored", Path(runner.save_dir))
        try:
            assert restored.critic_optimizer.state
            for actual, expected in zip(
                restored.critic(states, actions), runner.critic(states, actions)
            ):
                torch.testing.assert_close(actual, expected)
            runner.warmup()
            restored.warmup()
            for original, new, buffer in zip(
                runner.actor, restored.actor, runner.actor_buffer
            ):
                inputs = (
                    buffer.obs[0],
                    buffer.rnn_states[0],
                    buffer.masks[0],
                    buffer.available_actions[0],
                )
                torch.testing.assert_close(
                    original.act(*inputs, deterministic=True)[0],
                    new.act(*inputs, deterministic=True)[0],
                )
        finally:
            restored.close()
    finally:
        runner.close()
