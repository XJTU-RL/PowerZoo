# -*- coding: utf-8 -*-
"""
Asynchronous Multi-Agent Wrapper for Stackelberg Game

This module implements the asynchronous execution mechanism for the
Stackelberg-Nash game framework, where UC acts first and consumers respond.
"""

import numpy as np
import gym
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from copy import deepcopy
import logging

from envs.power_envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.power_envs.stackelberg.stackelberg_game.stackelberg_monitor import StackelbergMonitor


class AsyncMultiAgentWrapper:
    """
    Wrapper for asynchronous multi-agent execution in Stackelberg game.
    
    This wrapper manages the temporal relationship between UC (leader) and
    consumers (followers), ensuring proper sequencing of actions and observations.
    """
    
    def __init__(self, 
                 env: StackelbergBaseEnv,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the async wrapper.
        
        Args:
            env: The base Stackelberg environment
            config: Configuration dictionary
        """
        self.env = env
        self.config = config or {}
        
        # Agent configuration
        self.uc_agent_id = 0
        self.consumer_agent_ids = list(range(1, env.n_consumer_agents + 1))
        self.n_agents = env.n_agents
        
        # Timing configuration
        self.consumer_delay = self.config.get('consumer_delay', 1)
        self.action_persistence = self.config.get('action_persistence', 1)
        
        # Action buffers
        self.uc_action_buffer = None
        self.uc_action_history = deque(maxlen=self.config.get('history_length', 10))
        self.consumer_action_buffers = defaultdict(lambda: None)
        self.consumer_action_history = defaultdict(
            lambda: deque(maxlen=self.config.get('history_length', 10))
        )
        
        # State tracking
        self.current_phase = 'uc_decision'  # 'uc_decision' or 'consumer_response'
        self.phase_step = 0
        self.episode_step = 0
        
        # Stackelberg game environments
        self.stackelberg_env = env  # UC-consumer interaction
        self.nash_env = None  # Consumer-consumer interaction (placeholder)
        
        # Observation caching
        self.cached_observations = {}
        self.pending_observations = {}
        
        # Reward accumulation
        self.accumulated_rewards = defaultdict(float)
        self.pending_rewards = defaultdict(float)
        
        # Info aggregation
        self.aggregated_info = defaultdict(dict)
        
        # Monitoring integration
        self.monitor = None
        if self.config.get('enable_monitoring', True):
            monitor_config = self.config.get('monitor_config', {})
            self.monitor = StackelbergMonitor(
                log_dir=monitor_config.get('log_dir', 'logs/stackelberg'),
                experiment_name=monitor_config.get('experiment_name'),
                config=monitor_config
            )
        
        # Logging
        self.logger = logging.getLogger('AsyncWrapper')
        
        # Environment state
        self._done = False
        self._episode_rewards = defaultdict(float)
    
    def reset(self, **kwargs) -> Dict[int, np.ndarray]:
        """
        Reset the environment and wrapper state.
        
        Returns:
            Initial observations for all agents
        """
        # Reset base environment
        base_obs = self.env.reset(**kwargs)
        
        # Reset wrapper state
        self.current_phase = 'uc_decision'
        self.phase_step = 0
        self.episode_step = 0
        self._done = False
        
        # Clear buffers
        self.uc_action_buffer = None
        self.uc_action_history.clear()
        self.consumer_action_buffers.clear()
        for cid in self.consumer_agent_ids:
            self.consumer_action_history[cid].clear()
        
        # Clear tracking
        self.cached_observations.clear()
        self.pending_observations.clear()
        self.accumulated_rewards.clear()
        self.pending_rewards.clear()
        self.aggregated_info.clear()
        self._episode_rewards.clear()
        
        # Cache initial observations
        self.cached_observations = deepcopy(base_obs)
        
        # Prepare observations based on current phase
        observations = self._prepare_phase_observations(base_obs)
        
        return observations
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], bool, Dict[int, Dict[str, Any]]
    ]:
        """
        Execute one step of the asynchronous multi-agent environment.
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            observations: Next observations for active agents
            rewards: Rewards for active agents
            done: Whether episode is finished
            infos: Additional information
        """
        if self.current_phase == 'uc_decision':
            return self._step_uc_phase(actions)
        else:  # consumer_response
            return self._step_consumer_phase(actions)
    
    def _step_uc_phase(self, actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], bool, Dict[int, Dict[str, Any]]
    ]:
        """Handle UC decision phase."""
        # Validate UC action is provided
        if self.uc_agent_id not in actions:
            raise ValueError(f"UC action (agent {self.uc_agent_id}) not provided in UC phase")
        
        uc_action = actions[self.uc_agent_id]
        
        # Execute UC action
        uc_reward, uc_info = self.env.step_uc(uc_action)
        
        # Store UC action
        self.uc_action_buffer = uc_action.copy()
        self.uc_action_history.append(uc_action.copy())
        
        # Store UC reward (partial)
        self.pending_rewards[self.uc_agent_id] = uc_reward
        self.aggregated_info[self.uc_agent_id] = uc_info
        
        # Transition to consumer phase
        self.current_phase = 'consumer_response'
        self.phase_step += 1
        
        # Get observations for consumers
        all_observations = self.env.get_observations()
        consumer_observations = {
            aid: all_observations[aid] 
            for aid in self.consumer_agent_ids
        }
        
        # Prepare return values (only consumers get observations)
        rewards = {}  # No rewards yet in this phase
        infos = {aid: {'phase': 'consumer_response'} for aid in self.consumer_agent_ids}
        
        # Cache observations
        self.cached_observations = deepcopy(all_observations)
        
        return consumer_observations, rewards, False, infos
    
    def _step_consumer_phase(self, actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], bool, Dict[int, Dict[str, Any]]
    ]:
        """Handle consumer response phase."""
        # Validate consumer actions
        consumer_actions = {}
        for cid in self.consumer_agent_ids:
            if cid in actions:
                consumer_actions[cid] = actions[cid]
                self.consumer_action_buffers[cid] = actions[cid].copy()
                self.consumer_action_history[cid].append(actions[cid].copy())
            else:
                # Use default or previous action if not provided
                if self.consumer_action_buffers[cid] is not None:
                    consumer_actions[cid] = self.consumer_action_buffers[cid]
                else:
                    # Default action (no change)
                    consumer_actions[cid] = np.zeros(
                        self.env.action_spaces[cid].shape,
                        dtype=np.float32
                    )
        
        # Execute consumer actions in environment
        all_rewards, all_infos, done = self.env.step_consumers(consumer_actions)
        
        # Accumulate rewards
        for aid, reward in all_rewards.items():
            self.accumulated_rewards[aid] += reward
            self._episode_rewards[aid] += reward
        
        # Add pending UC reward
        if self.uc_agent_id in self.pending_rewards:
            self.accumulated_rewards[self.uc_agent_id] += self.pending_rewards[self.uc_agent_id]
            self._episode_rewards[self.uc_agent_id] += self.pending_rewards[self.uc_agent_id]
            self.pending_rewards.clear()
        
        # Update info
        for aid, info in all_infos.items():
            self.aggregated_info[aid].update(info)
        
        # Log to monitor if enabled
        if self.monitor and self.current_phase == 'consumer_response':
            combined_actions = {self.uc_agent_id: self.uc_action_buffer}
            combined_actions.update(consumer_actions)
            
            self.monitor.log_step(
                rewards=all_rewards,
                observations=self.cached_observations,
                actions=combined_actions,
                infos=all_infos,
                system_state=self.env.system_state
            )
        
        # Increment episode step
        self.episode_step += 1
        self._done = done
        
        # Transition back to UC phase if not done
        if not done:
            self.current_phase = 'uc_decision'
            self.phase_step += 1
            
            # Get next observations
            next_observations = self.env.get_observations()
            
            # Prepare UC observation for next round
            uc_observations = {
                self.uc_agent_id: next_observations[self.uc_agent_id]
            }
            
            # Return accumulated rewards and reset
            rewards = dict(self.accumulated_rewards)
            self.accumulated_rewards.clear()
            
            infos = deepcopy(self.aggregated_info)
            for aid in infos:
                infos[aid]['phase'] = 'uc_decision'
            
            return uc_observations, rewards, done, infos
        else:
            # Episode finished
            if self.monitor:
                self.monitor.log_episode_end()
            
            # Return final rewards
            rewards = dict(self.accumulated_rewards)
            infos = deepcopy(self.aggregated_info)
            
            # Add episode summary to info
            for aid in infos:
                infos[aid]['episode_reward'] = self._episode_rewards[aid]
                infos[aid]['episode_complete'] = True
            
            return {}, rewards, done, infos
    
    def _prepare_phase_observations(self, 
                                   base_observations: Dict[int, np.ndarray]
                                   ) -> Dict[int, np.ndarray]:
        """Prepare observations based on current phase."""
        if self.current_phase == 'uc_decision':
            # Only UC gets observation
            return {self.uc_agent_id: base_observations[self.uc_agent_id]}
        else:
            # Only consumers get observations
            return {
                aid: base_observations[aid]
                for aid in self.consumer_agent_ids
            }
    
    def get_agent_types(self) -> Dict[int, str]:
        """Get agent type mapping."""
        agent_types = {self.uc_agent_id: 'uc'}
        for cid in self.consumer_agent_ids:
            agent_types[cid] = 'consumer'
        return agent_types
    
    def get_active_agents(self) -> List[int]:
        """Get list of agents that should act in current phase."""
        if self.current_phase == 'uc_decision':
            return [self.uc_agent_id]
        else:
            return self.consumer_agent_ids
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get current phase information."""
        return {
            'current_phase': self.current_phase,
            'phase_step': self.phase_step,
            'episode_step': self.episode_step,
            'active_agents': self.get_active_agents()
        }
    
    def close(self):
        """Clean up resources."""
        if self.monitor:
            self.monitor.close()
        self.env.close() if hasattr(self.env, 'close') else None
    
    # Gym compatibility methods
    @property
    def observation_space(self):
        """Get observation spaces (delegated to base env)."""
        return self.env.observation_spaces
    
    @property
    def action_space(self):
        """Get action spaces (delegated to base env)."""
        return self.env.action_spaces
    
    @property
    def n_agents(self):
        """Get number of agents."""
        return self.env.n_agents
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)


class AsyncMultiAgentWrapperV2(AsyncMultiAgentWrapper):
    """
    Enhanced version with additional features:
    - Multi-level time delays
    - Action prediction for missing agents
    - Partial observability handling
    """
    
    def __init__(self, 
                 env: StackelbergBaseEnv,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced wrapper."""
        super().__init__(env, config)
        
        # Enhanced timing configuration
        self.consumer_delays = self.config.get('consumer_delays', {})
        if not self.consumer_delays:
            # Default: all consumers have same delay
            for cid in self.consumer_agent_ids:
                self.consumer_delays[cid] = self.consumer_delay
        
        # Action prediction
        self.enable_action_prediction = self.config.get('enable_action_prediction', False)
        self.action_predictors = {}
        if self.enable_action_prediction:
            self._init_action_predictors()
        
        # Partial observability
        self.partial_observability = self.config.get('partial_observability', False)
        self.observation_noise = self.config.get('observation_noise', 0.0)
    
    def _init_action_predictors(self):
        """Initialize action prediction models."""
        # Simple moving average predictors for now
        for aid in range(self.n_agents):
            self.action_predictors[aid] = {
                'history': deque(maxlen=10),
                'prediction': None
            }
    
    def _predict_action(self, agent_id: int) -> Optional[np.ndarray]:
        """Predict action for agent based on history."""
        if not self.enable_action_prediction:
            return None
        
        predictor = self.action_predictors[agent_id]
        if len(predictor['history']) < 3:
            return None
        
        # Simple moving average
        recent_actions = list(predictor['history'])
        predicted_action = np.mean(recent_actions, axis=0)
        
        return predicted_action
    
    def _add_observation_noise(self, 
                              observations: Dict[int, np.ndarray]
                              ) -> Dict[int, np.ndarray]:
        """Add noise to observations for partial observability."""
        if not self.partial_observability or self.observation_noise <= 0:
            return observations
        
        noisy_obs = {}
        for aid, obs in observations.items():
            noise = np.random.normal(0, self.observation_noise, obs.shape)
            noisy_obs[aid] = obs + noise
        
        return noisy_obs