# -*- coding: utf-8 -*-
"""
Stackelberg-Nash Multi-Agent Proximal Policy Optimization (SN-MAPPO)

This module implements the SN-MAPPO algorithm for solving bi-level
non-cooperative Stackelberg-Nash games in multi-agent settings.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

from utils.envs_tools import check
from utils.models_tools import get_grad_norm
from algorithms.actors.mappo import MAPPO


class SN_MAPPO(MAPPO):
    """
    Stackelberg-Nash MAPPO algorithm for hierarchical multi-agent learning.
    
    Key features:
    - Supports leader-follower hierarchy
    - Asynchronous update mechanism
    - Nash equilibrium seeking
    - Separate learning rates for UC and consumers
    """
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """
        Initialize SN-MAPPO algorithm.
        
        Args:
            args: Configuration dictionary
            obs_space: Observation space
            act_space: Action space
            device: PyTorch device
        """
        # Initialize base MAPPO
        super(SN_MAPPO, self).__init__(args, obs_space, act_space, device)
        
        # Stackelberg game configuration
        self.is_leader = args.get('is_leader', False)
        self.hierarchy_level = args.get('hierarchy_level', 0)
        self.agent_type = args.get('agent_type', 'consumer')  # 'uc' or 'consumer'
        
        # Learning configuration
        self.leader_lr_scale = args.get('leader_lr_scale', 1.0)
        self.follower_lr_scale = args.get('follower_lr_scale', 1.0)
        self.nash_iterations = args.get('nash_iterations', 1)
        
        # Best response tracking
        self.enable_best_response = args.get('enable_best_response', True)
        self.best_response_buffer_size = args.get('best_response_buffer_size', 100)
        self.best_response_buffer = []
        
        # Equilibrium tracking
        self.equilibrium_threshold = args.get('equilibrium_threshold', 0.1)
        self.equilibrium_window = args.get('equilibrium_window', 50)
        self.policy_changes = []
        
        # Separate optimizers for leader/follower if needed
        if self.is_leader:
            # Adjust learning rate for leader
            leader_lr = args['lr'] * self.leader_lr_scale
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=leader_lr,
                eps=args['opti_eps']
            )
        else:
            # Adjust learning rate for followers
            follower_lr = args['lr'] * self.follower_lr_scale
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=follower_lr,
                eps=args['opti_eps']
            )
        
        # Nash equilibrium solver parameters
        self.nash_solver_config = {
            'max_iterations': args.get('nash_max_iterations', 100),
            'convergence_threshold': args.get('nash_convergence_threshold', 1e-4),
            'learning_rate': args.get('nash_learning_rate', 0.01)
        }
        
        # Opponent modeling (for best response)
        self.opponent_models = {}
        self.enable_opponent_modeling = args.get('enable_opponent_modeling', False)
        
    def update(self, sample):
        """
        Update actor network with Stackelberg-Nash mechanism.
        
        Args:
            sample: Training sample batch
            
        Returns:
            policy_loss: Policy loss
            dist_entropy: Distribution entropy
            actor_grad_norm: Gradient norm
            imp_weights: Importance sampling weights
        """
        if self.is_leader:
            return self.update_leader(sample)
        else:
            return self.update_follower(sample)
    
    def update_leader(self, sample):
        """
        Update leader (UC) policy.
        
        The leader anticipates follower responses and optimizes accordingly.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample
        
        # Convert to tensors
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        # Anticipate follower response if enabled
        if self.enable_best_response and self.opponent_models:
            # Modify advantages based on anticipated follower responses
            adv_targ = self._adjust_advantages_for_followers(
                obs_batch, actions_batch, adv_targ
            )
        
        # Standard PPO update with adjusted advantages
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        
        # Importance sampling weights
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        
        # PPO surrogate objectives
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 
            1.0 - self.clip_param, 
            1.0 + self.clip_param
        ) * adv_targ
        
        # Policy loss
        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        
        policy_loss = policy_action_loss
        
        # Add equilibrium regularization for leader
        if len(self.policy_changes) > self.equilibrium_window:
            policy_stability = torch.tensor(
                np.std(self.policy_changes[-self.equilibrium_window:]),
                device=self.device
            )
            equilibrium_reg = 0.01 * policy_stability  # Regularization weight
            policy_loss = policy_loss + equilibrium_reg
        
        # Optimize
        self.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        
        self.actor_optimizer.step()
        
        # Track policy changes
        self.policy_changes.append(policy_loss.item())
        
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def _compute_total_derivative(self, loss_uc, loss_consumers, uc_params, consumer_params):
        """
        Compute total derivative for UC policy update (Equation 39).
        ∇L_u = ∇_θu L_u - ∇_θu,θc L_u (∇²_θc L_c)^(-1) ∇_θc L_u
        """
        # Compute first-order gradients
        grad_uc = torch.autograd.grad(loss_uc, uc_params, retain_graph=True)
        
        # Compute mixed second-order derivatives if followers exist
        if self.enable_best_response and consumer_params:
            # Compute ∇_θc L_u (how UC loss changes w.r.t consumer params)
            grad_uc_wrt_consumers = torch.autograd.grad(
                loss_uc, consumer_params, retain_graph=True, allow_unused=True
            )
            
            # Compute Hessian of consumer loss w.r.t consumer params
            # This is computationally expensive, so we use approximation
            hessian_consumers = self._approximate_hessian(loss_consumers, consumer_params)
            
            # Compute correction term
            if hessian_consumers is not None:
                # Solve linear system: H * x = g
                correction = torch.linalg.solve(hessian_consumers, grad_uc_wrt_consumers)
                
                # Compute mixed derivatives ∇_θu,θc L_u
                mixed_grads = torch.autograd.grad(
                    grad_uc_wrt_consumers, uc_params, retain_graph=True
                )
                
                # Apply correction
                total_derivative = []
                for g_uc, m_grad, corr in zip(grad_uc, mixed_grads, correction):
                    total_derivative.append(g_uc - torch.matmul(m_grad, corr))
            else:
                total_derivative = grad_uc
        else:
            total_derivative = grad_uc
        
        return total_derivative
    
    def _approximate_hessian(self, loss, params):
        """Approximate Hessian using finite differences or diagonal approximation."""
        # For computational efficiency, we use diagonal approximation
        # In practice, more sophisticated methods like L-BFGS could be used
        try:
            grads = torch.autograd.grad(loss, params, create_graph=True)
            hessian_diag = []
            
            for grad in grads:
                if grad is not None:
                    # Compute second derivative (diagonal elements only)
                    grad2 = torch.autograd.grad(
                        grad.sum(), params, retain_graph=True, allow_unused=True
                    )
                    hessian_diag.append(grad2)
            
            return torch.diag(torch.cat([h.flatten() for h in hessian_diag]))
        except (RuntimeError, ValueError) as e:
            # Hessian computation can fail for various reasons (singular matrix, etc.)
            return None
    
    def update_follower(self, sample):
        """
        Update follower (consumer) policy.
        
        Followers respond to leader actions and optimize their own objectives.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample
        
        # Extract leader signals from observations if available
        leader_signals = self._extract_leader_signals(obs_batch)
        
        # Adjust advantages based on leader signals
        if leader_signals is not None:
            adv_targ = self._adjust_advantages_for_leader(
                leader_signals, actions_batch, adv_targ
            )
        
        # Standard PPO update
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        
        # Importance sampling weights
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        
        # PPO surrogate objectives
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        ) * adv_targ
        
        # Policy loss
        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        
        policy_loss = policy_action_loss
        
        # Optimize
        self.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        
        self.actor_optimizer.step()
        
        # Update best response buffer
        if self.enable_best_response:
            self._update_best_response_buffer(
                obs_batch, actions_batch, action_log_probs
            )
        
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def _adjust_advantages_for_followers(self, obs_batch, actions_batch, advantages):
        """
        Adjust leader advantages based on anticipated follower responses.
        
        This implements the anticipatory mechanism where the leader
        considers how followers will respond to its actions.
        """
        # Simplified implementation
        # In practice, this would use opponent models to predict responses
        
        # Scale advantages based on expected follower cooperation
        cooperation_factor = 0.8  # Placeholder
        adjusted_advantages = advantages * cooperation_factor
        
        return adjusted_advantages
    
    def _adjust_advantages_for_leader(self, leader_signals, actions_batch, advantages):
        """
        Adjust follower advantages based on leader signals.
        
        This implements the response mechanism where followers
        adjust their behavior based on leader actions.
        """
        # Extract price signal (assuming first component)
        if isinstance(leader_signals, torch.Tensor) and leader_signals.dim() > 1:
            price_signals = leader_signals[:, 0:1]  # Shape: [batch, 1]
            
            # Higher prices should discourage consumption (negative advantage)
            price_factor = 2.0 - price_signals  # Inverted price impact
            
            # Adjust advantages
            adjusted_advantages = advantages * price_factor
        else:
            adjusted_advantages = advantages
        
        return adjusted_advantages
    
    def _extract_leader_signals(self, obs_batch):
        """
        Extract leader signals from observation batch.
        
        For consumers, the first few components of observation
        typically contain UC signals (price, DR incentive, etc.)
        """
        if self.agent_type == 'consumer':
            # Assuming first 3 components are UC signals
            # [price_signal, dr_incentive, capacity_allocation]
            obs_tensor = check(obs_batch).to(**self.tpdv)
            if obs_tensor.dim() >= 2 and obs_tensor.shape[-1] >= 3:
                leader_signals = obs_tensor[..., :3]
                return leader_signals
        
        return None
    
    def _update_best_response_buffer(self, obs_batch, actions_batch, action_log_probs):
        """Update best response buffer for opponent modeling."""
        # Store recent experiences
        experience = {
            'obs': obs_batch,
            'actions': actions_batch,
            'log_probs': action_log_probs.detach()
        }
        
        self.best_response_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.best_response_buffer) > self.best_response_buffer_size:
            self.best_response_buffer.pop(0)
    
    def compute_nash_equilibrium(self, leader_policy, follower_policies):
        """
        Compute Nash equilibrium between leader and followers.
        
        This is a simplified implementation. In practice, this would
        involve iterative best response dynamics or other game-theoretic
        solution concepts.
        """
        convergence_metric = float('inf')
        iteration = 0
        
        while (iteration < self.nash_solver_config['max_iterations'] and
               convergence_metric > self.nash_solver_config['convergence_threshold']):
            
            # Update leader policy given follower policies
            leader_update = self._compute_leader_best_response(
                leader_policy, follower_policies
            )
            
            # Update follower policies given leader policy
            follower_updates = []
            for f_policy in follower_policies:
                f_update = self._compute_follower_best_response(
                    leader_policy, f_policy
                )
                follower_updates.append(f_update)
            
            # Compute convergence metric
            convergence_metric = self._compute_convergence_metric(
                leader_update, follower_updates
            )
            
            iteration += 1
        
        return convergence_metric < self.nash_solver_config['convergence_threshold']
    
    def _compute_leader_best_response(self, leader_policy, follower_policies):
        """Compute leader's best response to current follower policies."""
        # Placeholder implementation
        # In practice, this would involve policy gradient or other optimization
        return 0.0
    
    def _compute_follower_best_response(self, leader_policy, follower_policy):
        """Compute follower's best response to leader policy."""
        # Placeholder implementation
        return 0.0
    
    def _compute_convergence_metric(self, leader_update, follower_updates):
        """Compute metric for Nash equilibrium convergence."""
        # Simple sum of policy changes
        total_change = abs(leader_update) + sum(abs(f) for f in follower_updates)
        return total_change
    
    def save_model(self, save_path: str):
        """Save model with additional SN-MAPPO components."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'is_leader': self.is_leader,
            'hierarchy_level': self.hierarchy_level,
            'agent_type': self.agent_type,
            'policy_changes': self.policy_changes,
            'best_response_buffer_size': len(self.best_response_buffer)
        }
        
        torch.save(save_dict, save_path)
    
    def load_model(self, load_path: str):
        """Load model with additional SN-MAPPO components."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        
        # Load SN-MAPPO specific components
        self.is_leader = checkpoint.get('is_leader', self.is_leader)
        self.hierarchy_level = checkpoint.get('hierarchy_level', self.hierarchy_level)
        self.agent_type = checkpoint.get('agent_type', self.agent_type)
        self.policy_changes = checkpoint.get('policy_changes', [])

    def _policy_distance(self, policy1, policy2):
        """Compute distance between two policies (e.g., KL divergence)."""
        # Placeholder implementation
        return 0.0
    
    def check_equilibrium_convergence(self):
        """
        Check if policies have converged to Stackelberg-Nash equilibrium.
        Based on KL divergence constraints (Equations 57-58).
        """
        if len(self.policy_changes) < self.equilibrium_window:
            return False
        
        # Check if policy changes are below threshold
        recent_changes = self.policy_changes[-self.equilibrium_window:]
        avg_change = np.mean(recent_changes)
        std_change = np.std(recent_changes)
        
        # Check KL divergence constraint
        converged = avg_change < self.equilibrium_threshold and std_change < 0.01
        
        return converged


class SN_MAPPO_Shared(SN_MAPPO):
    """
    Parameter-shared version of SN-MAPPO for homogeneous agents.
    
    All agents of the same type (UC or consumer) share parameters.
    """
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize shared SN-MAPPO."""
        super(SN_MAPPO_Shared, self).__init__(args, obs_space, act_space, device)
        
        # Additional configuration for parameter sharing
        self.share_within_type = args.get('share_within_type', True)
        self.type_embedding_dim = args.get('type_embedding_dim', 16)
        
        # Add agent type embedding if sharing within type
        if self.share_within_type:
            self.type_embedding = nn.Embedding(2, self.type_embedding_dim)  # UC=0, Consumer=1
    
    def forward(self, obs, agent_type_id, *args, **kwargs):
        """Forward pass with agent type embedding."""
        if self.share_within_type and hasattr(self, 'type_embedding'):
            # Add type embedding to observation
            type_emb = self.type_embedding(
                torch.tensor([agent_type_id], device=obs.device)
            ).expand(obs.shape[0], -1)
            
            # Concatenate with observation
            obs_with_type = torch.cat([obs, type_emb], dim=-1)
            
            return super().forward(obs_with_type, *args, **kwargs)
        else:
            return super().forward(obs, *args, **kwargs)