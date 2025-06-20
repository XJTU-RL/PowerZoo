#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : dan_buffer.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: Extended buffer for DAN-HAPPO algorithm with neighbor observations
"""

import torch
import numpy as np
from utils.util import get_shape_from_obs_space, get_shape_from_act_space
from utils.shared_buffer import SharedReplayBuffer

class DANSharedReplayBuffer(SharedReplayBuffer):
    """Extended replay buffer for DAN-HAPPO algorithm
    
    This buffer extends the base SharedReplayBuffer to handle DAN-specific data
    such as neighbor observations and agent masks.
    """
    
    def __init__(self, args, obs_space, cent_obs_space, act_space):
        """Initialize DAN replay buffer
        Args:
            args: Arguments containing buffer configuration
            obs_space: Observation space
            cent_obs_space: Centralized observation space
            act_space: Action space
        """
        super(DANSharedReplayBuffer, self).__init__(args, obs_space, cent_obs_space, act_space)
        
        # DAN-specific parameters
        self.use_neighbor_obs = getattr(args, 'use_neighbor_obs', True)
        self.max_neighbors = getattr(args, 'max_neighbors', 5)
        
        if self.use_neighbor_obs:
            # Initialize neighbor observation buffer
            self.neighbor_obs = np.zeros(
                (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 
                 self.max_neighbors, *self.obs_shape), dtype=np.float32
            )
            
            # Initialize agent mask buffer (indicates which neighbors are valid)
            self.agent_masks = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 
                 self.max_neighbors), dtype=np.float32
            )
            
            # Initialize attention weights buffer (for analysis)
            self.attention_weights = np.zeros(
                (self.episode_length, self.n_rollout_threads, self.num_agents, 
                 self.max_neighbors), dtype=np.float32
            )
            
    def insert(self, obs, cent_obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None,
               active_masks=None, available_actions=None, neighbor_obs=None, agent_masks=None, attention_weights=None):
        """Insert data into buffer with DAN-specific data
        Args:
            obs: Observations
            cent_obs: Centralized observations
            actions: Actions
            action_log_probs: Action log probabilities
            value_preds: Value predictions
            rewards: Rewards
            masks: Masks
            bad_masks: Bad masks
            active_masks: Active masks
            available_actions: Available actions
            neighbor_obs: Neighbor observations (DAN-specific)
            agent_masks: Agent masks (DAN-specific)
            attention_weights: Attention weights (DAN-specific)
        """
        # Call parent insert method
        super().insert(obs, cent_obs, actions, action_log_probs, value_preds, rewards, masks, 
                      bad_masks, active_masks, available_actions)
        
        # Insert DAN-specific data
        if self.use_neighbor_obs:
            if neighbor_obs is not None:
                self.neighbor_obs[self.step] = neighbor_obs.copy()
            if agent_masks is not None:
                self.agent_masks[self.step] = agent_masks.copy()
            if attention_weights is not None and self.step > 0:
                self.attention_weights[self.step - 1] = attention_weights.copy()
                
    def after_update(self):
        """Reset buffer after update"""
        super().after_update()
        
        if self.use_neighbor_obs:
            # Copy last step data to first step
            self.neighbor_obs[0] = self.neighbor_obs[-1].copy()
            self.agent_masks[0] = self.agent_masks[-1].copy()
            
    def chooseinsert(self, obs, cent_obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None,
                    active_masks=None, available_actions=None, neighbor_obs=None, agent_masks=None):
        """Choose insert method for DAN buffer
        Args:
            obs: Observations
            cent_obs: Centralized observations
            actions: Actions
            action_log_probs: Action log probabilities
            value_preds: Value predictions
            rewards: Rewards
            masks: Masks
            bad_masks: Bad masks
            active_masks: Active masks
            available_actions: Available actions
            neighbor_obs: Neighbor observations (DAN-specific)
            agent_masks: Agent masks (DAN-specific)
        """
        self.insert(obs, cent_obs, actions, action_log_probs, value_preds, rewards, masks, 
                   bad_masks, active_masks, available_actions, neighbor_obs, agent_masks)
                   
    def get_neighbor_batch(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """Get neighbor observation batch for training
        Args:
            advantages: Advantage values
            num_mini_batch: Number of mini batches
            mini_batch_size: Mini batch size
        Returns:
            Generator yielding neighbor observation batches
        """
        if not self.use_neighbor_obs:
            return None
            
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of agents ({}) = {} "
                "to be greater than or equal to the number of "
                "PPO mini batches ({}).".format(n_rollout_threads, num_agents, 
                                               n_rollout_threads * num_agents, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
            
        # Flatten neighbor observations and agent masks
        neighbor_obs_batch = self.neighbor_obs[:-1].reshape(-1, self.max_neighbors, *self.obs_shape)
        agent_masks_batch = self.agent_masks[:-1].reshape(-1, self.max_neighbors)
        
        # Create random permutation
        rand = torch.randperm(batch_size * episode_length).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        for indices in sampler:
            neighbor_obs_mini_batch = neighbor_obs_batch[indices]
            agent_masks_mini_batch = agent_masks_batch[indices]
            
            yield neighbor_obs_mini_batch, agent_masks_mini_batch
            
    def feed_forward_generator_dan(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """Data generator for DAN-HAPPO training
        Args:
            advantages: Advantage values
            num_mini_batch: Number of mini batches
            mini_batch_size: Mini batch size
        Yields:
            Mini batches of training data including DAN-specific data
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of agents ({}) = {} "
                "to be greater than or equal to the number of "
                "PPO mini batches ({}).".format(n_rollout_threads, num_agents, 
                                               n_rollout_threads * num_agents, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
            
        # Flatten all data
        obs_batch = self.obs[:-1].reshape(-1, *self.obs_shape)
        cent_obs_batch = self.cent_obs[:-1].reshape(-1, *self.cent_obs_shape)
        actions_batch = self.actions.reshape(-1, self.actions.shape[-1])
        
        if self.available_actions is not None:
            available_actions_batch = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        else:
            available_actions_batch = None
            
        value_preds_batch = self.value_preds[:-1].reshape(-1, 1)
        return_batch = self.returns[:-1].reshape(-1, 1)
        masks_batch = self.masks[:-1].reshape(-1, 1)
        active_masks_batch = self.active_masks[:-1].reshape(-1, 1)
        old_action_log_probs_batch = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        
        if advantages is None:
            adv_targ = None
        else:
            adv_targ = advantages.reshape(-1, 1)
            
        # DAN-specific data
        if self.use_neighbor_obs:
            neighbor_obs_batch = self.neighbor_obs[:-1].reshape(-1, self.max_neighbors, *self.obs_shape)
            agent_masks_batch = self.agent_masks[:-1].reshape(-1, self.max_neighbors)
        else:
            neighbor_obs_batch = None
            agent_masks_batch = None
            
        # Create random permutation
        rand = torch.randperm(batch_size * episode_length).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        for indices in sampler:
            obs_mini_batch = obs_batch[indices]
            cent_obs_mini_batch = cent_obs_batch[indices]
            actions_mini_batch = actions_batch[indices]
            value_preds_mini_batch = value_preds_batch[indices]
            return_mini_batch = return_batch[indices]
            masks_mini_batch = masks_batch[indices]
            active_masks_mini_batch = active_masks_batch[indices]
            old_action_log_probs_mini_batch = old_action_log_probs_batch[indices]
            
            if available_actions_batch is not None:
                available_actions_mini_batch = available_actions_batch[indices]
            else:
                available_actions_mini_batch = None
                
            if adv_targ is not None:
                adv_targ_mini_batch = adv_targ[indices]
            else:
                adv_targ_mini_batch = None
                
            # DAN-specific mini batches
            if self.use_neighbor_obs:
                neighbor_obs_mini_batch = neighbor_obs_batch[indices]
                agent_masks_mini_batch = agent_masks_batch[indices]
            else:
                neighbor_obs_mini_batch = None
                agent_masks_mini_batch = None
                
            yield obs_mini_batch, cent_obs_mini_batch, actions_mini_batch, value_preds_mini_batch, \
                  return_mini_batch, masks_mini_batch, active_masks_mini_batch, old_action_log_probs_mini_batch, \
                  adv_targ_mini_batch, available_actions_mini_batch, neighbor_obs_mini_batch, agent_masks_mini_batch
                  
    def get_attention_statistics(self):
        """Get attention weight statistics for analysis
        Returns:
            Dictionary containing attention statistics
        """
        if not self.use_neighbor_obs or not hasattr(self, 'attention_weights'):
            return {}
            
        # Calculate attention entropy
        attention_weights = self.attention_weights[:self.step]
        if attention_weights.size > 0:
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            attention_weights_norm = attention_weights + eps
            attention_weights_norm = attention_weights_norm / np.sum(attention_weights_norm, axis=-1, keepdims=True)
            
            # Calculate entropy
            entropy = -np.sum(attention_weights_norm * np.log(attention_weights_norm + eps), axis=-1)
            avg_entropy = np.mean(entropy)
            
            # Calculate attention concentration (max attention weight)
            max_attention = np.max(attention_weights, axis=-1)
            avg_max_attention = np.mean(max_attention)
            
            return {
                'attention_entropy': avg_entropy,
                'max_attention_weight': avg_max_attention,
                'attention_std': np.std(attention_weights)
            }
        else:
            return {}
            
    def get_neighbor_statistics(self):
        """Get neighbor observation statistics
        Returns:
            Dictionary containing neighbor statistics
        """
        if not self.use_neighbor_obs:
            return {}
            
        # Calculate average number of neighbors
        agent_masks = self.agent_masks[:self.step]
        if agent_masks.size > 0:
            avg_neighbors = np.mean(np.sum(agent_masks, axis=-1))
            max_neighbors = np.max(np.sum(agent_masks, axis=-1))
            min_neighbors = np.min(np.sum(agent_masks, axis=-1))
            
            return {
                'avg_neighbors': avg_neighbors,
                'max_neighbors': max_neighbors,
                'min_neighbors': min_neighbors
            }
        else:
            return {}