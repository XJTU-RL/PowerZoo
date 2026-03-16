# -*- coding: utf-8 -*-
"""
Stackelberg PowerZoo Environment Wrapper

This module provides a high-level interface that integrates all Stackelberg
game components into a PowerZoo-compatible environment.

Interface:
- step() returns (local_obs, share_obs, rewards, dones, infos, avail_actions)
- reset() returns (obs, share_obs, avail_actions)
- Provides share_observation_space for MARL algorithms
- get_avail_actions() for action masking
"""

import copy
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

# Conditional gym/gymnasium import
try:
	import gymnasium as gym
	from gymnasium.spaces import Box
except ImportError:
	import gym
	from gym.spaces import Box

from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.stackelberg.stackelberg_game.async_wrapper import AsyncMultiAgentWrapper
from envs.stackelberg.stackelberg_game.stackelberg_monitor import StackelbergMonitor
from envs.stackelberg.stackelberg_game.load_aggregator import IntelligentLoadAggregator

class StackelbergVVCEnv:
    """
    Main wrapper class for Stackelberg game environment in PowerZoo.
    
    This class provides a unified interface that:
    - Integrates with PowerZoo's training infrastructure
    - Supports multiple bus systems (13, 34, 123)
    - Implements asynchronous hierarchical decision making
    - Provides comprehensive monitoring
    """
    
    def __init__(self, config_or_args, rank=None):
        """Initialize Stackelberg PowerZoo environment.

        Args:
            config_or_args: StackelbergConfig 实例或原始 env_args dict（向后兼容）
            rank: 可选 worker 索引，覆盖 config 中的 worker_idx
        """
        from envs.stackelberg.stackelberg_config import StackelbergConfig

        if isinstance(config_or_args, StackelbergConfig):
            self.config = config_or_args
            if rank is not None:
                self.config.worker_idx = rank
            self.args = {}
        else:
            self.args = copy.deepcopy(config_or_args)
            self.config = StackelbergConfig.from_env_args(self.args)
            if rank is not None:
                self.config.worker_idx = rank

        self.logger = logging.getLogger('StackelbergPowerZoo')

        # Build env_config dict from typed config (StackelbergBaseEnv still expects dict)
        self.env_config = self._build_env_config()

        # Create base environment
        self.base_env = StackelbergBaseEnv(self.env_config)

        # Create load aggregator
        load_agg_config = self.env_config.get('load_aggregation', {
            'method': 'zone',
            'max_loads_per_agent': 5,
        })
        self.load_aggregator = IntelligentLoadAggregator(
            self.base_env.circuit,
            method=load_agg_config.get('method', 'bus_proximity'),
            config=load_agg_config
        )

        # Perform load aggregation
        n_consumer_agents = self.env_config.get(
            f'n_consumer_agents_{self.config.system_name.lower()}',
            self.config.n_consumer_agents
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
        monitoring_cfg = self.env_config.get('monitoring_config', {})
        if monitoring_cfg.get('enable', True):
            monitor_config = dict(monitoring_cfg)
            monitor_config['experiment_name'] = self.config.exp_name
            self.monitor = StackelbergMonitor(
                log_dir=monitor_config.get('log_dir', 'logs/stackelberg'),
                experiment_name=monitor_config.get('experiment_name'),
                config=monitor_config
            )

        # Set environment properties for PowerZoo compatibility
        self._setup_vvc_compatibility()

        # Episode tracking
        self.current_episode = 0
        self.total_steps = 0

    def _build_env_config(self) -> dict:
        """Build env_config dict from typed StackelbergConfig.

        Translates typed config into the dict format that StackelbergBaseEnv expects.
        """
        from utils.path_utils import resolve_system_path

        cfg = self.config

        # Resolve system path
        try:
            system_dir = str(resolve_system_path(cfg.system_name))
        except FileNotFoundError:
            system_dir = None

        env_config = {
            'system_name': cfg.system_name,
            'dss_file': cfg.dss_file,
            'dss_folder': cfg.dss_folder or (system_dir if system_dir else None),
            'max_episode_steps': cfg.max_episode_steps,
            'seed': cfg.seed,
            'worker_idx': cfg.worker_idx,
            'n_consumer_agents': cfg.n_consumer_agents,
            'use_render': cfg.use_render,
            'use_load_noise': cfg.use_load_noise,
            'scale': cfg.scale,
        }

        # Pass through dict sub-configs
        for key in ['tou_config', 'tier_config', 'reward_weights',
                    'load_aggregation', 'async_config', 'monitoring_config', 'n1_security']:
            val = getattr(cfg, key, None)
            if val is not None:
                env_config[key] = val

        return env_config
    
    def _setup_vvc_compatibility(self):
        """Setup properties for PowerZoo compatibility."""
        # Number of agents
        self.n_agents = self.base_env.n_agents
        self.agents = list(range(self.n_agents))

        # Spaces - convert dict to list for PowerZoo compatibility
        self.observation_spaces = self.base_env.observation_spaces
        self.action_spaces = self.base_env.action_spaces

        # Convert to list format if dict
        if isinstance(self.observation_spaces, dict):
            self.observation_space = [self.observation_spaces[i] for i in range(self.n_agents)]
        else:
            self.observation_space = list(self.observation_spaces.values()) if hasattr(self.observation_spaces, 'values') else self.observation_spaces

        if isinstance(self.action_spaces, dict):
            self.action_space = [self.action_spaces[i] for i in range(self.n_agents)]
        else:
            self.action_space = list(self.action_spaces.values()) if hasattr(self.action_spaces, 'values') else self.action_spaces

        # Share observation space - use UC observation as global state (system-wide view)
        uc_obs_space = self.observation_spaces[0] if isinstance(self.observation_spaces, dict) else self.observation_spaces[0]
        share_obs_dim = uc_obs_space.shape[0]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # Environment info
        self.env_name = self.args.get('env_name', 'stackelberg') if self.args else 'stackelberg'
        self.max_episode_steps = self.config.max_episode_steps

        # Agent types
        self.agent_types = self.async_wrapper.get_agent_types()

        # Other properties
        self.discrete = False  # Continuous actions
        self.episode_limit = self.max_episode_steps
    
    def reset(self, choose=None):
        """
        Reset environment (PowerZoo compatible).

        Args:
            choose: Optional load profile index

        Returns:
            local_obs: List of observations for each agent
            share_obs: List of global state for each agent
            avail_actions: Available actions for each agent
        """
        self.current_episode += 1

        # Reset through async wrapper
        observations = self.async_wrapper.reset(load_profile_idx=choose)

        # Convert observations to PowerZoo format
        local_obs = self._convert_obs_to_list(observations)

        # Share observation is the UC (global) observation repeated for all agents
        uc_obs = observations.get(0, np.zeros(self.share_observation_space[0].shape))
        share_obs = [uc_obs.copy() for _ in range(self.n_agents)]

        # Get available actions
        avail_actions = self.get_avail_actions()

        # Log episode start
        self.logger.info(f"Episode {self.current_episode} started")

        return local_obs, share_obs, avail_actions
    
    def step(self, actions):
        """
        Execute environment step (PowerZoo compatible).

        This method handles the Stackelberg game's two-phase structure internally:
        1. UC (leader) makes decision
        2. Consumers (followers) respond

        Args:
            actions: List or array of actions for all agents

        Returns:
            local_obs: List of observations for each agent
            share_obs: List of global state for each agent
            rewards: List of [[reward]] for each agent
            dones: List of done flags for each agent
            infos: List of info dicts for each agent
            avail_actions: Available actions for each agent
        """
        # Convert actions to dictionary format, trimming to each agent's actual action dimension
        # NOTE: Runner's _collect_heterogeneous() pads all actions to max_action_dim (UC=5),
        # but consumers only have 3-dim action spaces. We must trim here to avoid shape mismatch
        # in stackelberg_base_env.step_consumers() np.clip().
        if isinstance(actions, (list, np.ndarray)):
            action_dict = {}
            for i in range(len(actions)):
                raw_action = np.array(actions[i]).flatten()
                actual_dim = self.action_space[i].shape[0]
                action_dict[i] = raw_action[:actual_dim]
        else:
            action_dict = actions

        # Execute Stackelberg game step (handles UC-then-Consumer internally)
        observations, rewards, done, infos = self._execute_stackelberg_step(action_dict)

        # Update step counter
        self.total_steps += 1

        # Convert observations to PowerZoo format
        local_obs = self._convert_obs_to_list(observations)

        # Share observation (UC observation as global state)
        uc_obs = observations.get(0, np.zeros(self.share_observation_space[0].shape))
        share_obs = [uc_obs.copy() for _ in range(self.n_agents)]

        # Convert rewards to HAPPO-compatible numpy format (n_agents, 1)
        if isinstance(rewards, dict):
            reward_values = [rewards.get(i, 0.0) for i in range(self.n_agents)]
        else:
            reward_values = list(rewards) if hasattr(rewards, '__iter__') else [rewards] * self.n_agents
        rewards_array = np.array([[float(r)] for r in reward_values], dtype=np.float32)

        # Done flags as numpy boolean array (n_agents,) - HAPPO requirement
        dones_array = np.array([bool(done)] * self.n_agents, dtype=bool)

        # Info list
        if isinstance(infos, dict):
            info_list = [infos.get(i, {}) for i in range(self.n_agents)]
        else:
            info_list = [infos] * self.n_agents

        # Get available actions
        avail_actions = self.get_avail_actions()

        return local_obs, share_obs, rewards_array, dones_array, info_list, avail_actions

    def _execute_stackelberg_step(self, action_dict: Dict[int, np.ndarray]) -> Tuple:
        """
        Execute complete Stackelberg game step.

        Handles the two-phase nature of Stackelberg games:
        Phase 1: UC (agent 0) acts as leader
        Phase 2: Consumers respond as followers

        Args:
            action_dict: Actions for all agents

        Returns:
            Tuple of (observations, rewards, done, infos)
        """
        # Get current phase
        phase_info = self.async_wrapper.get_phase_info()

        if phase_info['current_phase'] == 'uc_decision':
            # Phase 1: Execute UC action
            uc_action = action_dict.get(0)
            if uc_action is not None:
                _, _, _, _ = self.async_wrapper.step({0: uc_action})

            # Phase 2: Execute consumer actions
            consumer_actions = {k: v for k, v in action_dict.items() if k != 0}
            observations, rewards, done, infos = self.async_wrapper.step(consumer_actions)

        elif phase_info['current_phase'] == 'consumer_response':
            # If already in consumer phase, just execute consumer actions
            consumer_actions = {k: v for k, v in action_dict.items() if k != 0}
            observations, rewards, done, infos = self.async_wrapper.step(consumer_actions)

        else:
            # Fallback - should not reach here normally
            observations, rewards, done, infos = self.async_wrapper.step(action_dict)

        return observations, rewards, done, infos

    def _convert_obs_to_list(self, observations: Dict[int, np.ndarray]) -> List[np.ndarray]:
        """Convert observation dict to list format."""
        return [observations.get(i, np.zeros(self.observation_space[i].shape))
                for i in range(self.n_agents)]

    def get_avail_actions(self) -> List:
        """
        Get available actions for all agents.

        For continuous action spaces, returns None for each agent.
        For discrete action spaces, returns list of available actions.

        Returns:
            List of available actions for each agent
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id: int):
        """
        Get available actions for a specific agent.

        Args:
            agent_id: Agent index

        Returns:
            None for continuous actions, list for discrete actions
        """
        action_space = self.action_space[agent_id]

        # Continuous action space - all actions available
        if isinstance(action_space, Box):
            return None

        # Discrete action space - return mask
        if hasattr(action_space, 'n'):
            return [1] * action_space.n

        return None

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
        """Get global state size (consistent with share_observation_space)."""
        return self.share_observation_space[0].shape[0]
    
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
        if hasattr(self.base_env, 'seed') and callable(self.base_env.seed):
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
    """Factory function to create Stackelberg environment.

    Accepts either a StackelbergConfig or a raw dict (backward compat).
    """
    return StackelbergVVCEnv(args)