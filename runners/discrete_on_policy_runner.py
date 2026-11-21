"""Common discrete action checks and deterministic local inference for IPPO and COMA."""

import numpy as np
import torch

from runners.on_policy_base_runner import OnPolicyBaseRunner
from utils.discrete_marl_config import validate_discrete_vvc


class DiscreteOnPolicyRunner(OnPolicyBaseRunner):
    def __init__(self, args, algo_args, env_args):
        validate_discrete_vvc(args["algo"], args["env"], algo_args, env_args)
        try:
            super().__init__(args, algo_args, env_args)
        except BaseException:
            self.close()
            raise

    def _agent_available_actions(self, available_actions, agent_id):
        masks = super()._agent_available_actions(available_actions, agent_id)
        count = self.envs.action_space[agent_id].n
        if (
            masks is None
            or masks.ndim != 2
            or masks.shape[1] != count
            or not np.all((masks == 0) | (masks == 1))
            or not masks.any(axis=1).all()
        ):
            raise ValueError(
                "Each device needs a binary mask with at least one available action"
            )
        return masks

    @torch.no_grad()
    def predict(self, obs, masks):
        """Return a vectorized deterministic device frame from local observations."""
        actions = []
        for i, actor in enumerate(self.actor):
            local = np.stack([frame[i] for frame in obs])
            rnn = np.zeros(
                (len(local), self.recurrent_n, self.rnn_hidden_size), dtype=np.float32
            )
            action, _ = actor.act(
                local,
                rnn,
                np.ones((len(local), 1), dtype=np.float32),
                self._agent_available_actions(masks, i),
                deterministic=True,
            )
            actions.append(action.cpu().numpy())
        return np.stack(actions, axis=1)
