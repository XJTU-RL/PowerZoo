"""Feedforward, heterogeneous discrete QMIX and VDN learners."""

from copy import deepcopy
from itertools import chain
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from models.value_function_models.agent_q_function import AgentQFunction
from models.value_function_models.mq_mixer import M_QMixer
from utils.trans_tools import DecayThenFlatSchedule


class SumMixer(nn.Module):
    """VDN's state-independent additive joint value."""

    def forward(self, agent_values, state):
        return agent_values.sum(dim=1, keepdim=True)


class DiscreteValueDecomposition:
    def __init__(
        self, algorithm, model, settings, obs_spaces, action_spaces, state_dim, device
    ):
        self.settings = settings
        self.device = device
        parameters = SimpleNamespace(
            **settings,
            model=model,
            hidden_size=model["hidden_sizes"][-1],
            gain=model["gain"],
            use_orthogonal=model["initialization_method"] == "orthogonal_",
        )
        self.policies = nn.ModuleList(
            AgentQFunction(parameters, obs.shape[0], action.n, device)
            for obs, action in zip(obs_spaces, action_spaces)
        )
        self.mixer = (
            M_QMixer(parameters, len(action_spaces), state_dim, device)
            if algorithm == "qmix"
            else SumMixer()
        )
        self.target_policies = deepcopy(self.policies).requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).requires_grad_(False)
        self.parameters = list(
            chain(self.policies.parameters(), self.mixer.parameters())
        )
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=model["lr"],
            eps=model["opti_eps"],
            weight_decay=model["weight_decay"],
        )
        self.exploration = DecayThenFlatSchedule(
            settings["epsilon_start"],
            settings["epsilon_finish"],
            settings["epsilon_anneal_time"],
            decay="linear",
        )

    @staticmethod
    def mask_q_values(q_values, available):
        mask = torch.as_tensor(available, device=q_values.device)
        if (
            mask.shape != q_values.shape
            or not ((mask == 0) | (mask == 1)).all()
            or not mask.bool().any(dim=1).all()
        ):
            raise ValueError(
                "Each device needs a binary mask with at least one available action"
            )
        return q_values.masked_fill(~mask.bool(), -torch.inf)

    @torch.no_grad()
    def predict(self, obs, masks, *, exploration_step=None):
        actions = []
        for i, policy in enumerate(self.policies):
            local = np.stack([frame[i] for frame in obs])
            available = np.stack([frame[i] for frame in masks])
            legal_q = self.mask_q_values(policy(local), available)
            action = legal_q.argmax(dim=1)
            if exploration_step is not None:
                epsilon = self.exploration.eval(exploration_step)
                logits = legal_q.new_zeros(legal_q.shape).masked_fill(
                    ~torch.as_tensor(available, device=self.device, dtype=torch.bool),
                    -torch.inf,
                )
                random_action = Categorical(logits=logits).sample()
                action = torch.where(
                    torch.rand(action.shape, device=self.device) < epsilon,
                    random_action,
                    action,
                )
            actions.append(action.cpu().numpy())
        return np.stack(actions, axis=1)[..., None]

    def train_batch(self, batch):
        reward = self._float([transition.reward for transition in batch]).unsqueeze(1)
        live = [transition for transition in batch if not transition.terminated]
        state = self._float(
            np.stack([transition.current.state for transition in batch])
        )
        joint_actions = self._long(
            np.stack([transition.actions for transition in batch])
        )
        chosen_values = []
        for i, policy in enumerate(self.policies):
            obs = np.stack([transition.current.obs[i] for transition in batch])
            available = np.stack(
                [transition.current.available[i] for transition in batch]
            )
            q = self.mask_q_values(policy(obs), available)
            chosen = q.gather(1, joint_actions[:, i : i + 1])
            if not chosen.isfinite().all():
                raise ValueError(
                    "Replay contains an unavailable action or a non-finite Q value"
                )
            chosen_values.append(chosen)
        total = self.mixer(torch.cat(chosen_values, dim=1), state).reshape(-1, 1)

        with torch.no_grad():
            if live:
                next_states = self._float(
                    np.stack([item.following.state for item in live])
                )
                next_values = []
                for i, (policy, target) in enumerate(
                    zip(self.policies, self.target_policies)
                ):
                    obs = np.stack([item.following.obs[i] for item in live])
                    available = np.stack([item.following.available[i] for item in live])
                    target_q = target(obs)
                    selection = (
                        policy(obs) if self.settings["use_double_q"] else target_q
                    )
                    action = self.mask_q_values(selection, available).argmax(
                        dim=1, keepdim=True
                    )
                    next_values.append(target_q.gather(1, action))
                next_total = self.target_mixer(
                    torch.cat(next_values, dim=1), next_states
                )
                indices = self._long(
                    [
                        i
                        for i, transition in enumerate(batch)
                        if not transition.terminated
                    ]
                )
                reward[indices] += self.settings["gamma"] * next_total.reshape(-1, 1)

        if self.settings["use_huber_loss"]:
            loss = F.huber_loss(total, reward, delta=self.settings["huber_delta"])
        else:
            loss = F.mse_loss(total, reward)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.parameters, self.settings["max_grad_norm"]
        )
        self.optimizer.step()
        self.update_targets()
        return {
            "td_loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "joint_q": total.detach().mean().item(),
        }

    @torch.no_grad()
    def update_targets(self):
        targets = chain(
            self.target_policies.parameters(), self.target_mixer.parameters()
        )
        for target, online in zip(targets, self.parameters):
            target.lerp_(online, self.settings["tau"])

    def _float(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _long(self, values):
        return torch.as_tensor(values, dtype=torch.long, device=self.device)
