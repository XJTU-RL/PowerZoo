"""COMA using the common VVC on-policy collection and evaluation paths."""

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.critics.coma_critic import (
    COMACritic,
    counterfactual_advantage,
    lambda_targets,
)
from common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from runners.discrete_on_policy_runner import DiscreteOnPolicyRunner
from utils.discrete_marl_config import require_discrete_spaces
from utils.models_tools import update_linear_schedule


class COMARunner(DiscreteOnPolicyRunner):
    def _init_critics(self):
        spaces = self.envs.action_space
        require_discrete_spaces(spaces)
        model, settings = self.algo_args["model"], self.algo_args["algo"]
        state_space = self.envs.share_observation_space[0]
        self.critic = COMACritic(
            model, state_space.shape[0], [space.n for space in spaces]
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.critic_buffer = OnPolicyCriticBufferEP(
            {**model, **settings, **self.algo_args["train"]}, state_space
        )
        self.value_normalizer = None
        self.truncations = {}
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=model["critic_lr"],
            eps=model["opti_eps"],
            weight_decay=model["weight_decay"],
        )

    def _collect_critics(self, step):
        # Q-values require the joint action. They are computed from the complete rollout.
        return (
            np.zeros_like(self.critic_buffer.value_preds[step]),
            np.zeros_like(self.critic_buffer.rnn_states_critic[step]),
        )

    def _insert_critics(
        self, share_obs, rewards, infos, values, rnn_states, masks, bad_masks
    ):
        truncated = [
            (worker, info[0])
            for worker, info in enumerate(infos)
            if masks[worker, 0, 0] == 0
            and info[0].get("TimeLimit.truncated")
            and not (info[0].get("terminated") or info[0].get("physical_failure"))
        ]
        self.truncations[self.critic_buffer.step] = truncated
        super()._insert_critics(
            share_obs, rewards, infos, values, rnn_states, masks, bad_masks
        )

    def _bootstrap(self, state, obs, masks):
        actions = []
        for i, actor in enumerate(self.actor):
            local = np.stack([frame[i] for frame in obs])
            rnn = np.zeros(
                (len(obs), self.recurrent_n, self.rnn_hidden_size), dtype=np.float32
            )
            sampled, _ = actor.act(
                local,
                rnn,
                np.ones((len(obs), 1), dtype=np.float32),
                self._agent_available_actions(masks, i),
            )
            actions.append(sampled.long())
        joint = torch.cat(actions, dim=-1)
        return self._taken(self.target_critic(self._float(state), joint), joint)

    def _joint_actions(self, step):
        actions = [buffer.actions[step] for buffer in self.actor_buffer]
        return torch.as_tensor(
            np.concatenate(actions, axis=-1), dtype=torch.long, device=self.device
        )

    def _float(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @staticmethod
    def _taken(q_values, actions):
        return torch.cat(
            [q.gather(1, actions[:, i : i + 1]) for i, q in enumerate(q_values)], dim=1
        )

    @torch.no_grad()
    def compute(self):
        buffer = self.critic_buffer
        steps, workers = buffer.rewards.shape[:2]
        state = self._float(buffer.share_obs[:-1]).flatten(0, 1)
        actions = self._joint_actions(slice(None)).flatten(0, 1)
        q_taken = self._taken(self.target_critic(state, actions), actions).reshape(
            steps, workers, -1
        )
        last_alive = buffer.masks[-1, :, 0].astype(bool)
        last_q = torch.zeros_like(q_taken[-1])
        if last_alive.any():
            indices = np.flatnonzero(last_alive)
            last_q[torch.as_tensor(indices, device=self.device)] = self._bootstrap(
                buffer.share_obs[-1, indices],
                [
                    [actor.obs[-1, worker] for actor in self.actor_buffer]
                    for worker in indices
                ],
                [
                    [actor.available_actions[-1, worker] for actor in self.actor_buffer]
                    for worker in indices
                ],
            )
        next_q = torch.cat([q_taken[1:], last_q.unsqueeze(0)], dim=0)
        truncations = torch.zeros(
            (steps, workers, 1), dtype=torch.bool, device=self.device
        )
        if self.algo_args["train"]["use_proper_time_limits"]:
            for step, workers_and_info in self.truncations.items():
                for worker, info in workers_and_info:
                    next_q[step, worker : worker + 1] = self._bootstrap(
                        np.asarray([info["original_state"][0]]),
                        [info["original_obs"]],
                        [info["original_avail_actions"]],
                    )
                    truncations[step, worker] = True
        settings = self.algo_args["algo"]
        self.targets = lambda_targets(
            self._float(buffer.rewards),
            self._float(buffer.masks[1:]),
            next_q,
            settings["gamma"],
            settings["td_lambda"],
            truncations,
        ).flatten(0, 1)

    def train(self):
        state = self._float(self.critic_buffer.share_obs[:-1]).flatten(0, 1)
        actions = self._joint_actions(slice(None)).flatten(0, 1)
        q_values = self.critic(state, actions)
        loss = F.mse_loss(self._taken(q_values, actions), self.targets)
        self.critic_optimizer.zero_grad()
        loss.backward()
        settings = self.algo_args["algo"]
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), settings["max_grad_norm"]
        )
        self.critic_optimizer.step()

        with torch.no_grad():
            q_values = self.critic(state, actions)
        infos = []
        for i, (actor, buffer, q) in enumerate(
            zip(self.actor, self.actor_buffer, q_values)
        ):
            chosen = actions[:, i : i + 1]
            available = self._float(buffer.available_actions[:-1]).flatten(0, 1)
            log_prob, entropy, distribution = actor.evaluate_actions(
                self._float(buffer.obs[:-1]).flatten(0, 1),
                self._float(buffer.rnn_states[:-1]).flatten(0, 1),
                chosen,
                self._float(buffer.masks[:-1]).flatten(0, 1),
                available,
                self._float(buffer.active_masks[:-1]).flatten(0, 1),
            )
            advantage = counterfactual_advantage(
                q, chosen, distribution.probs.detach(), available
            )
            policy_loss = -(log_prob * advantage).mean()
            actor.actor_optimizer.zero_grad()
            (policy_loss - settings["entropy_coef"] * entropy).backward()
            torch.nn.utils.clip_grad_norm_(
                actor.actor.parameters(), settings["max_grad_norm"]
            )
            actor.actor_optimizer.step()
            infos.append(
                {"policy_loss": policy_loss.item(), "dist_entropy": entropy.item()}
            )

        with torch.no_grad():
            for target, current in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target.lerp_(current, settings["target_tau"])
        return infos, {"td_loss": loss.item()}

    def _prepare_critics(self, method, *args):
        if method == "lr_decay":
            update_linear_schedule(
                self.critic_optimizer, *args, self.algo_args["model"]["critic_lr"]
            )
        else:
            self.critic.train(method == "prep_training")

    def _save_critics(self, save_path):
        filename = "coma_training.pt"
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "target": self.target_critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_optimizers": [
                    actor.actor_optimizer.state_dict() for actor in self.actor
                ],
            },
            save_path / filename,
        )
        return [filename]

    def _restore_critics(self, model_dir):
        if self.algo_args["render"]["use_render"]:
            return
        states = torch.load(
            model_dir / "coma_training.pt", map_location=self.device, weights_only=True
        )
        if len(states["actor_optimizers"]) != self.num_agents:
            raise ValueError("COMA checkpoint has an incompatible device count")
        self.critic.load_state_dict(states["critic"])
        self.target_critic.load_state_dict(states["target"])
        self.critic_optimizer.load_state_dict(states["critic_optimizer"])
        for actor, optimizer in zip(self.actor, states["actor_optimizers"]):
            actor.actor_optimizer.load_state_dict(optimizer)
