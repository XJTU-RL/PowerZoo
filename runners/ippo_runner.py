"""Independent PPO using the common on-policy rollout and evaluation loop."""

import numpy as np
import torch

from algorithms.critics.v_critic import VCritic
from common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from common.valuenorm import ValueNorm
from runners.discrete_on_policy_runner import DiscreteOnPolicyRunner
from utils.discrete_marl_config import require_discrete_spaces
from utils.trans_tools import _t2n


class IPPORunner(DiscreteOnPolicyRunner):
    """Every actor, value function, and advantage uses only its device's observation."""

    def _init_critics(self):
        require_discrete_spaces(self.envs.action_space)
        args = self.algo_args
        model = {**args["model"], **args["algo"]}
        buffer_args = {**args["train"], **model}
        spaces = self.envs.observation_space
        self.critics = [VCritic(model, space, device=self.device) for space in spaces]
        self.critic_buffers = [
            OnPolicyCriticBufferEP(buffer_args, space) for space in spaces
        ]
        self.value_normalizers = [
            ValueNorm(1, device=self.device) if args["train"]["use_valuenorm"] else None
            for _ in spaces
        ]
        self.value_normalizer = self.value_normalizers[0]
        self.truncation_bootstrap = np.zeros_like(self.critic_buffers[0].rewards)
        self.truncation_bootstrap = np.repeat(
            self.truncation_bootstrap[:, :, None, :], self.num_agents, axis=2
        )

    def _warmup_critics(self, share_obs):
        for buffer, actor in zip(self.critic_buffers, self.actor_buffer):
            buffer.share_obs[0] = actor.obs[0]

    def _collect_critics(self, step):
        outputs = [
            critic.get_values(
                buffer.share_obs[step],
                buffer.rnn_states_critic[step],
                buffer.masks[step],
            )
            for critic, buffer in zip(self.critics, self.critic_buffers)
        ]
        values, states = zip(*outputs)
        return np.stack([_t2n(v) for v in values], axis=1), np.stack(
            [_t2n(state) for state in states], axis=1
        )

    @torch.no_grad()
    def _insert_critics(
        self, share_obs, rewards, infos, values, rnn_states, masks, bad_masks
    ):
        ended = masks[:, 0, 0] == 0
        truncated = [
            worker
            for worker, info in enumerate(infos)
            if ended[worker]
            and info[0].get("TimeLimit.truncated", False)
            and not (info[0].get("physical_failure") or info[0].get("terminated"))
        ]
        keep_mask = np.ones_like(bad_masks)
        for i, (critic, buffer, actor, normalizer) in enumerate(
            zip(
                self.critics,
                self.critic_buffers,
                self.actor_buffer,
                self.value_normalizers,
            )
        ):
            bootstrap = self.truncation_bootstrap[buffer.step, :, i]
            bootstrap.fill(0)
            if buffer.use_proper_time_limits and truncated:
                final_obs = np.stack(
                    [infos[w][0]["original_obs"][i] for w in truncated]
                )
                final_value, _ = critic.get_values(
                    final_obs, rnn_states[truncated, i], keep_mask[truncated]
                )
                final_value = _t2n(final_value)
                if normalizer is not None:
                    final_value = normalizer.denormalize(final_value)
                bootstrap[truncated] = buffer.gamma * final_value
            states = rnn_states[:, i].copy()
            states[ended] = 0
            buffer.insert(
                actor.obs[buffer.step + 1],
                states,
                values[:, i],
                rewards[:, i],
                masks[:, i],
                keep_mask,
                [],
            )

    @torch.no_grad()
    def compute(self):
        values, _ = self._collect_critics(-1)
        for i, (buffer, normalizer) in enumerate(
            zip(self.critic_buffers, self.value_normalizers)
        ):
            # Fold the final-state value into the TD target while retaining raw rewards
            # for the common logger. The mask prevents GAE from crossing an auto-reset.
            rewards = buffer.rewards
            buffer.rewards = rewards + self.truncation_bootstrap[:, :, i]
            try:
                buffer.compute_returns(values[:, i], normalizer)
            finally:
                buffer.rewards = rewards

    def train(self):
        actors, critics = [], []
        for actor, actor_buffer, critic, buffer, normalizer in zip(
            self.actor,
            self.actor_buffer,
            self.critics,
            self.critic_buffers,
            self.value_normalizers,
        ):
            values = buffer.value_preds[:-1]
            if normalizer is not None:
                values = normalizer.denormalize(values)
            actors.append(actor.train(actor_buffer, buffer.returns[:-1] - values, "EP"))
            critics.append(critic.train(buffer, normalizer))
        return actors, {
            key: sum(info[key] for info in critics) / len(critics) for key in critics[0]
        }

    def _after_critic_update(self):
        for buffer in self.critic_buffers:
            buffer.after_update()

    def _critic_log_buffer(self):
        # Each buffer records the same team reward.
        return self.critic_buffers[0]

    def _prepare_critics(self, method, *args):
        for critic in self.critics:
            getattr(critic, method)(*args)

    def _save_critics(self, save_path):
        filename = "ippo_training.pt"
        torch.save(
            [
                {
                    "critic": critic.critic.state_dict(),
                    "critic_optimizer": critic.critic_optimizer.state_dict(),
                    "actor_optimizer": actor.actor_optimizer.state_dict(),
                    "normalizer": normalizer.state_dict()
                    if normalizer is not None
                    else None,
                }
                for critic, actor, normalizer in zip(
                    self.critics, self.actor, self.value_normalizers
                )
            ],
            save_path / filename,
        )
        return [filename]

    def _restore_critics(self, model_dir):
        if self.algo_args["render"]["use_render"]:
            return
        states = torch.load(
            model_dir / "ippo_training.pt", map_location=self.device, weights_only=True
        )
        if len(states) != self.num_agents or any(
            (item["normalizer"] is None) != (normalizer is None)
            for item, normalizer in zip(states, self.value_normalizers)
        ):
            raise ValueError(
                "IPPO checkpoint has incompatible agent count or value normalization"
            )
        for item, critic, actor, normalizer in zip(
            states, self.critics, self.actor, self.value_normalizers
        ):
            critic.critic.load_state_dict(item["critic"])
            critic.critic_optimizer.load_state_dict(item["critic_optimizer"])
            actor.actor_optimizer.load_state_dict(item["actor_optimizer"])
            if normalizer is not None:
                normalizer.load_state_dict(item["normalizer"])
