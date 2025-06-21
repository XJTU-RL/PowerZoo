#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : dsr_dan_runner.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: DSR runner specifically designed for DAN-HAPPO algorithm
"""

import time
import numpy as np
import torch
from runners.on_policy_base_runner import OnPolicyBaseRunner
from utils.models_tools import update_linear_schedule

class DSRDANRunner(OnPolicyBaseRunner):
    """Runner for DSR environment with DAN-HAPPO algorithm
    
    This runner extends the base runner to handle DAN-specific features
    such as neighbor observations and enhanced action masking.
    """
    
    def __init__(self, config):
        super(DSRDANRunner, self).__init__(config)
        
        # DAN-specific parameters
        self.use_dan = getattr(config, 'use_dan', True)
        self.use_neighbor_obs = getattr(config, 'use_neighbor_obs', True)
        self.max_neighbors = getattr(config, 'max_neighbors', 5)
        self.use_enhanced_action_mask = getattr(config, 'use_enhanced_action_mask', True)
        
        # Training metrics
        self.dan_metrics = {
            'attention_entropy': [],
            'neighbor_count': [],
            'action_mask_ratio': []
        }
        
    def run(self):
        """Main training loop"""
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            # Collect rollout data
            train_infos = self.collect(episode)
            
            # Update policy
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # Save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                
            # Log training information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                
                # Log DAN-specific metrics
                if self.use_dan and train_infos:
                    self.log_dan_metrics(train_infos)
                    
                # Log training information
                self.log_train(train_infos, total_num_steps)
                
            # Evaluation
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                
    def collect(self, episode):
        """Collect rollout data with DAN-specific features
        Args:
            episode: Current episode number
        Returns:
            train_infos: Training information
        """
        self.trainer.prep_rollout()
        
        # Initialize episode data
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        
        # DAN-specific data collection
        neighbor_obs_buffer = []
        agent_mask_buffer = []
        action_mask_ratios = []
        
        for step in range(self.episode_length):
            # Sample actions
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect_step(step)
            
            # Environment step
            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
            
            # Collect DAN-specific data
            if self.use_dan:
                neighbor_obs, agent_masks = self.collect_neighbor_data(infos)
                neighbor_obs_buffer.append(neighbor_obs)
                agent_mask_buffer.append(agent_masks)
                
            # Collect action mask statistics
            if available_actions is not None:
                mask_ratios = [np.mean(mask) for mask in available_actions]
                action_mask_ratios.append(np.mean(mask_ratios))
                
            # Store data in buffer
            data = obs, share_obs, rewards, dones, infos, available_actions, \
                   values, actions, action_log_probs, rnn_states, rnn_states_critic
            
            # Add DAN-specific data
            if self.use_dan:
                data = data + (neighbor_obs_buffer[-1], agent_mask_buffer[-1])
                
            self.insert(data)
            
            # Update episode statistics
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_costs.append(info.get('episode_cost', 0))
                    episode_lengths.append(info['episode']['l'])
                    
        # Compute returns and update policy
        self.compute()
        train_infos = self.train()
        
        # Add DAN-specific metrics to training info
        if self.use_dan:
            train_infos = self.add_dan_metrics(train_infos, neighbor_obs_buffer, 
                                             agent_mask_buffer, action_mask_ratios)
            
        return train_infos
        
    def collect_step(self, step):
        """Collect data for a single step with DAN encoding
        Args:
            step: Current step number
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # Get current observations
        obs = self.buffer.obs[step]
        share_obs = self.buffer.share_obs[step]
        rnn_states = self.buffer.rnn_states[step]
        rnn_states_critic = self.buffer.rnn_states_critic[step]
        masks = self.buffer.masks[step]
        available_actions = self.buffer.available_actions[step] if self.buffer.available_actions is not None else None
        
        # Collect neighbor observations if using DAN
        neighbor_obs = None
        agent_mask = None
        if self.use_dan and self.use_neighbor_obs:
            neighbor_obs, agent_mask = self.get_neighbor_observations(step)
            
        # Get actions with DAN encoding
        with torch.no_grad():
            values, actions, action_log_probs, rnn_states, rnn_states_critic = \
                self.trainer.policy.get_actions(share_obs, obs, rnn_states, rnn_states_critic, 
                                               masks, available_actions, 
                                               neighbor_obs=neighbor_obs, agent_mask=agent_mask)
                                               
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
        
    def get_neighbor_observations(self, step):
        """Get neighbor observations for DAN
        Args:
            step: Current step number
        Returns:
            neighbor_obs: Neighbor observations
            agent_mask: Agent mask
        """
        # TODO: Implement neighbor observation collection based on environment
        # This is a placeholder implementation
        
        batch_size = self.n_rollout_threads
        num_agents = self.num_agents
        
        # Create dummy neighbor observations
        neighbor_obs = np.zeros((batch_size, num_agents, self.max_neighbors, self.obs_space.shape[0]))
        agent_mask = np.ones((batch_size, num_agents, self.max_neighbors))
        
        # In a real implementation, this would:
        # 1. Get current agent positions/IDs
        # 2. Find neighboring agents based on topology
        # 3. Collect their observations
        # 4. Create appropriate masks
        
        return neighbor_obs, agent_mask
        
    def collect_neighbor_data(self, infos):
        """Collect neighbor data from environment info
        Args:
            infos: Environment info
        Returns:
            neighbor_obs: Neighbor observations
            agent_masks: Agent masks
        """
        # Extract neighbor information from environment
        neighbor_obs = []
        agent_masks = []
        
        for info in infos:
            if 'neighbor_obs' in info:
                neighbor_obs.append(info['neighbor_obs'])
            else:
                # Create dummy neighbor observations
                dummy_obs = np.zeros((self.num_agents, self.max_neighbors, self.obs_space.shape[0]))
                neighbor_obs.append(dummy_obs)
                
            if 'agent_mask' in info:
                agent_masks.append(info['agent_mask'])
            else:
                # Create dummy agent masks
                dummy_mask = np.ones((self.num_agents, self.max_neighbors))
                agent_masks.append(dummy_mask)
                
        return np.array(neighbor_obs), np.array(agent_masks)
        
    def add_dan_metrics(self, train_infos, neighbor_obs_buffer, agent_mask_buffer, action_mask_ratios):
        """Add DAN-specific metrics to training info
        Args:
            train_infos: Original training info
            neighbor_obs_buffer: Buffer of neighbor observations
            agent_mask_buffer: Buffer of agent masks
            action_mask_ratios: Action mask ratios
        Returns:
            train_infos: Updated training info with DAN metrics
        """
        if not train_infos:
            train_infos = {}
            
        # Calculate average neighbor count
        if agent_mask_buffer:
            avg_neighbor_count = np.mean([np.sum(masks) for masks in agent_mask_buffer])
            train_infos['dan_avg_neighbor_count'] = avg_neighbor_count
            
        # Calculate average action mask ratio
        if action_mask_ratios:
            train_infos['dan_avg_action_mask_ratio'] = np.mean(action_mask_ratios)
            
        return train_infos
        
    def log_dan_metrics(self, train_infos):
        """Log DAN-specific metrics
        Args:
            train_infos: Training information containing DAN metrics
        """
        if 'dan_attention_entropy' in train_infos:
            self.dan_metrics['attention_entropy'].append(train_infos['dan_attention_entropy'])
            
        if 'dan_avg_neighbor_count' in train_infos:
            self.dan_metrics['neighbor_count'].append(train_infos['dan_avg_neighbor_count'])
            
        if 'dan_avg_action_mask_ratio' in train_infos:
            self.dan_metrics['action_mask_ratio'].append(train_infos['dan_avg_action_mask_ratio'])
            
        # Print DAN metrics
        print(f"DAN Metrics:")
        if self.dan_metrics['attention_entropy']:
            print(f"  Attention Entropy: {np.mean(self.dan_metrics['attention_entropy'][-10:]):.4f}")
        if self.dan_metrics['neighbor_count']:
            print(f"  Avg Neighbor Count: {np.mean(self.dan_metrics['neighbor_count'][-10:]):.2f}")
        if self.dan_metrics['action_mask_ratio']:
            print(f"  Action Mask Ratio: {np.mean(self.dan_metrics['action_mask_ratio'][-10:]):.4f}")
            
    def insert(self, data):
        """Insert data into buffer with DAN-specific handling
        Args:
            data: Data tuple to insert
        """
        if self.use_dan and len(data) > 11:  # Has DAN data
            obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic, \
            neighbor_obs, agent_masks = data
            
            # Store DAN-specific data
            # TODO: Extend buffer to handle neighbor observations and agent masks
            
        else:
            obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data
            
        # Call parent insert method
        super().insert(obs, share_obs, rewards, dones, infos, available_actions,
                      values, actions, action_log_probs, rnn_states, rnn_states_critic)
                      
    def save(self):
        """Save model and DAN-specific data"""
        super().save()
        
        # Save DAN metrics
        if self.use_dan:
            dan_metrics_path = str(self.save_dir) + "/dan_metrics.npy"
            np.save(dan_metrics_path, self.dan_metrics)
            
    def restore(self):
        """Restore model and DAN-specific data"""
        super().restore()
        
        # Restore DAN metrics if available
        if self.use_dan:
            dan_metrics_path = str(self.save_dir) + "/dan_metrics.npy"
            try:
                self.dan_metrics = np.load(dan_metrics_path, allow_pickle=True).item()
            except FileNotFoundError:
                print("DAN metrics file not found, starting with empty metrics.")