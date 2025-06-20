# -*- coding: utf-8 -*-
"""
Stackelberg PowerZoo Environment Wrapper

This module provides a high-level interface that integrates all Stackelberg
game components into a PowerZoo-compatible environment.
"""

import os
import copy
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.stackelberg.stackelberg_game.async_wrapper import AsyncMultiAgentWrapper

from envs.stackelberg.stackelberg_game.stackelberg_monitor import StackelbergMonitor
from envs.stackelberg.stackelberg_game.load_aggregator import IntelligentLoadAggregator

class StackelbergPowerZooEnv:
    """
    Main wrapper class for Stackelberg game environment in PowerZoo.
    
    This class provides a unified interface that:
    - Integrates with PowerZoo's training infrastructure
    - Supports multiple bus systems (13, 34, 123)
    - Implements asynchronous hierarchical decision making
    - Provides comprehensive monitoring
    """
    
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize Stackelberg PowerZoo environment.
        
        Args:
            args: Configuration dictionary from PowerZoo
        """
        self.args = copy.deepcopy(args)
        self.logger = logging.getLogger('StackelbergPowerZoo')
        
        # Extract configuration
        self._parse_config(args)
        
        # Create base environment
        self.base_env = StackelbergBaseEnv(self.env_config)
        
        # Create load aggregator
        self.load_aggregator = IntelligentLoadAggregator(
            self.base_env.circuit,
            method=self.env_config['load_aggregation']['method'],
            config=self.env_config['load_aggregation']
        )
        
        # Perform load aggregation
        n_consumer_agents = self.env_config.get(
            f'n_consumer_agents_{self.env_config["system_name"].lower()}',
            self.env_config.get('n_consumer_agents', 10)
        )
        aggregation_mapping = self.load_aggregator.aggregate_loads(n_consumer_agents)
        
        # Update base environment with aggregation
        self.base_env.load_to_agent = aggregation_mapping
        self.base_env._build_agent_to_loads()
        
        # Create async wrapper
        self.async_wrapper = AsyncMultiAgentWrapper(
            self.base_env,
            config=self.env_config.get('async_config', {})
        )
        
        # Create monitor if enabled
        self.monitor = None
        if self.env_config.get('monitoring_config', {}).get('enabled', True):
            monitor_config = self.env_config['monitoring_config']
            monitor_config['experiment_name'] = args.get('exp_name', None)
            self.monitor = StackelbergMonitor(
                log_dir=monitor_config.get('log_dir', 'logs/stackelberg'),
                experiment_name=monitor_config.get('experiment_name'),
                config=monitor_config
            )
        
        # Set environment properties for PowerZoo compatibility
        self._setup_powerzoo_compatibility()
        
        # Episode tracking
        self.current_episode = 0
        self.total_steps = 0
    
    def _parse_config(self, args: Dict[str, Any]):
        """Parse configuration from PowerZoo args."""
        # Build environment configuration
        self.env_config = {
            'system_name': args.get('env_name', '13Bus').replace('stackelberg_', ''),
            'dss_file': args.get('dss_file', 'Master_noPV_1.dss'),
            'max_episode_steps': args.get('num_steps', 24),
            'base_path': args.get('base_path', 'envs/powerzoo/systems'),
            'seed': args.get('seed', 123456),
            'worker_idx': args.get('worker_idx'),
        }
        
        # Load system-specific configuration if available
        config_file = f"stackelberg_{self.env_config['system_name'].lower()}.yaml"
        config_path = os.path.join('configs/envs_cfgs', config_file)
        
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                system_config = yaml.safe_load(f)
            
            # Merge configurations
            self.env_config.update(system_config)
        
        # Override with args
        for key in ['n_consumer_agents', 'use_load_noise', 'scale', 'use_render']:
            if key in args:
                self.env_config[key] = args[key]
    
    def _setup_powerzoo_compatibility(self):
        """Setup properties for PowerZoo compatibility."""
        # Number of agents
        self.n_agents = self.base_env.n_agents
        self.agents = list(range(self.n_agents))
        
        # Spaces
        self.observation_spaces = self.base_env.observation_spaces
        self.action_spaces = self.base_env.action_spaces
        
        # Single space versions (for homogeneous case)
        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]
        
        # Environment info
        self.env_name = self.args.get('env_name', 'stackelberg')
        self.max_episode_steps = self.env_config['max_episode_steps']
        
        # Agent types
        self.agent_types = self.async_wrapper.get_agent_types()
        
        # Other properties
        self.discrete = False  # Continuous actions
        self.episode_limit = self.max_episode_steps
    
    def reset(self, choose=None):
        """
        Reset environment.
        
        Args:
            choose: Optional load profile index
            
        Returns:
            observations: Initial observations for all agents
        """
        self.current_episode += 1
        
        # Reset through async wrapper
        observations = self.async_wrapper.reset(load_profile_idx=choose)
        
        # Log episode start
        self.logger.info(f"Episode {self.current_episode} started")
        
        return observations
    
    def step(self, actions):
        """
        Execute environment step.
        
        Args:
            actions: Dictionary or array of actions
            
        Returns:
            observations: Next observations
            rewards: Agent rewards
            dones: Done flags
            infos: Additional information
        """
        # Convert actions to dictionary if needed
        if isinstance(actions, (list, np.ndarray)):
            action_dict = {i: actions[i] for i in range(len(actions))}
        else:
            action_dict = actions
        
        # Step through async wrapper
        observations, rewards, done, infos = self.async_wrapper.step(action_dict)
        
        # Update step counter
        self.total_steps += 1
        
        # Convert outputs for PowerZoo compatibility
        if isinstance(rewards, dict):
            reward_array = np.array([rewards.get(i, 0.0) for i in range(self.n_agents)])
        else:
            reward_array = rewards
        
        # Done is same for all agents in this setup
        done_array = np.array([done] * self.n_agents)
        
        return observations, reward_array, done_array, infos
    
    def get_env_info(self):
        """Get environment information for PowerZoo."""
        env_info = {
            "n_agents": self.n_agents,
            "n_actions": self.action_spaces[0].shape[0] if hasattr(self.action_spaces[0], 'shape') else 1,
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "episode_limit": self.episode_limit,
            "agent_types": self.agent_types,
            "uc_agent_id": 0,
            "n_consumer_agents": self.n_agents - 1,
        }
        return env_info
    
    def get_obs_size(self):
        """Get observation size."""
        # Return size of first agent's observation space
        if hasattr(self.observation_spaces[0], 'shape'):
            return self.observation_spaces[0].shape[0]
        else:
            return self.observation_spaces[0].n
    
    def get_state_size(self):
        """Get global state size."""
        # Global state could be concatenation of all observations
        # plus additional system information
        total_obs_size = sum(
            space.shape[0] if hasattr(space, 'shape') else space.n
            for space in self.observation_spaces.values()
        )
        
        # Add system state dimensions
        system_state_size = 10  # Placeholder
        
        return total_obs_size + system_state_size
    
    def get_total_actions(self):
        """Get total number of actions (for discrete action spaces)."""
        if hasattr(self.action_spaces[0], 'n'):
            return self.action_spaces[0].n
        else:
            # Continuous actions
            return -1
    
    def get_stats(self):
        """Get environment statistics."""
        if hasattr(self.async_wrapper, 'monitor') and self.async_wrapper.monitor:
            return self.async_wrapper.monitor.get_summary_statistics()
        return {}
    
    def save_replay(self):
        """Save replay data."""
        if self.monitor:
            self.monitor.save_monitoring_data()
    
    def close(self):
        """Clean up environment."""
        if self.monitor:
            self.monitor.close()
        
        if hasattr(self.async_wrapper, 'close'):
            self.async_wrapper.close()
        
        self.logger.info("Environment closed")
    
    # Additional methods for PowerZoo compatibility
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.base_env, 'seed'):
            self.base_env.seed(seed)
    
    def render(self, mode='human'):
        """Render environment."""
        if hasattr(self.base_env, 'render'):
            return self.base_env.render(mode)
    
    def get_phase_info(self):
        """Get current phase information (UC decision or consumer response)."""
        return self.async_wrapper.get_phase_info()
    
    def get_monitoring_data(self):
        """Get monitoring data if available."""
        if self.monitor:
            return self.monitor.get_summary_statistics()
        return None


def make_stackelberg_env(args):
    """
    Factory function to create Stackelberg environment.
    
    This function is called by PowerZoo's environment creation logic.
    """
    return StackelbergPowerZooEnv(args)