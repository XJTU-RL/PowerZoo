# -*- coding: utf-8 -*-
"""
Runner for on-policy HA (Heterogeneous Agent) algorithms with decoupled ordering.

This module implements the OnPolicyHARunnerDecoupled class which handles training for
algorithms like SHOM/HAPPO that require sequential agent updates with
sensitivity-based ordering.

The ordering logic has been decoupled from the environment using:
- SensitivityCalculator: Abstract interface for computing agent sensitivities
- AgentOrderStrategy: Strategy pattern for determining agent update order

This is an alternative to OnPolicyHARunner that uses the decoupled architecture.
Use this runner when you need:
- Easy extension to new environments without modifying the runner
- Pluggable ordering strategies (sensitivity-based, fixed, random, custom)
- Better separation of concerns between environment and algorithm

Usage:
    To use this runner instead of OnPolicyHARunner, update your runner registry
    or instantiate directly:

    from runners.on_policy_ha_runner_decoupled import OnPolicyHARunnerDecoupled
    runner = OnPolicyHARunnerDecoupled(args, algo_args, env_args)
"""

import numpy as np
import torch
from utils.trans_tools import _t2n
from utils.agent_ordering import AgentOrderManager, create_order_manager
from runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunnerDecoupled(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms with decoupled sensitivity-based ordering.

    This class provides the same functionality as OnPolicyHARunner but uses a
    decoupled architecture that separates:
    - Sensitivity calculation (via SensitivityCalculator interface)
    - Agent ordering (via AgentOrderStrategy pattern)

    This design allows for easy extension to new environments and ordering
    strategies without modifying the core runner code.
    """

    def __init__(self, args, algo_args, env_args):
        """
        Initialize OnPolicyHARunnerDecoupled.

        Args:
            args: Command-line arguments (algo, env, exp_name).
            algo_args: Algorithm configuration.
            env_args: Environment configuration.
        """
        super().__init__(args, algo_args, env_args)

        # Initialize the agent order manager with decoupled components
        self._init_order_manager(args, algo_args, env_args)

    def _init_order_manager(self, args, algo_args, env_args):
        """
        Initialize the AgentOrderManager with appropriate strategy.

        This method sets up the decoupled ordering system based on configuration.
        The order manager combines sensitivity calculation and ordering strategy
        to provide a clean interface for the training loop.
        """
        # Get agent-bus mapping and agent-id mapping if available
        agents_bus_mapping = None
        agent_id_mapping = None

        if args["env"] == "powerzoo" and env_args.get("useS", False):
            # Get mappings from environment (these are set in base runner)
            if hasattr(self, 'get_agents_bus'):
                agents_bus_mapping = self.get_agents_bus
            if hasattr(self, 'get_ordered_agents_pairs'):
                agent_id_mapping = self.get_ordered_agents_pairs

        # Create the order manager using factory function
        self.order_manager = create_order_manager(
            env_name=args["env"],
            env_args=env_args,
            algo_args=algo_args,
            num_agents=self.num_agents,
            agents_bus_mapping=agents_bus_mapping,
            agent_id_mapping=agent_id_mapping
        )

    def train(self):
        """
        Train the model using sequential agent updates.

        The order of agent updates is determined by the AgentOrderManager,
        which can use various strategies (sensitivity-based, fixed, random).

        Returns:
            Tuple of (actor_train_infos, critic_train_info).
        """
        actor_train_infos = []

        # Factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # Compute advantages
        if self.value_normalizer is not None:
            advantages = (
                self.critic_buffer.returns[:-1]
                - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
            )
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # Normalize advantages for FP state type
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # Compute agent order using the decoupled order manager
        agent_order = self.order_manager.compute_order(
            num_agents=self.num_agents,
            buffer_infos=self.critic_buffer.infos if self.useS else None,
            verbose=True
        )

        # Sequential agent updates in computed order
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)

            # Reshape available actions
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # Compute action log probs for the actor before update
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # Update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )

            # Compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # Update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # Update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
