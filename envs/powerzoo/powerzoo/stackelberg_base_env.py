# -*- coding: utf-8 -*-
"""
Stackelberg Game Base Environment for Power System Demand Response

This module implements a bi-level non-cooperative game-theoretic framework
for demand response in distribution networks, based on Stackelberg-Nash equilibrium.

Key Features:
- Hierarchical decision-making: UC (leader) and consumers (followers)
- Asynchronous action execution with temporal relationships
- Support for 13Bus, 34Bus, and 123Bus systems
- Comprehensive monitoring and logging
- No inheritance from existing Env class for maximum flexibility
"""

import os
import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import json

from envs.powerzoo.powerzoo.circuit import Circuits
from envs.powerzoo.powerzoo.loadprofile import LoadProfile


class StackelbergBaseEnv:
    """
    Base environment for Stackelberg game-theoretic demand response.
    
    This environment models the interaction between a Utility Company (UC) as the leader
    and multiple consumers as followers in a Stackelberg-Nash game framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Stackelberg base environment.
        
        Args:
            config: Configuration dictionary containing:
                - system_name: '13Bus', '34Bus', or '123Bus'
                - dss_file: OpenDSS file name
                - max_episode_steps: Episode length (default: 24)
                - n_consumer_agents: Number of consumer agents
                - monitoring_config: Monitoring settings
                - reward_weights: Reward function weights
                - ...
        """
        self.config = config
        self.system_name = config['system_name']
        self.dss_file = config['dss_file']
        self.max_episode_steps = config.get('max_episode_steps', 24)
        
        # System paths
        self.dss_folder_path = os.path.join(
            config.get('base_path', 'envs/powerzoo/systems'),
            self.system_name
        )
        
        # Agent configuration
        self.n_uc_agents = 1  # Single UC agent
        self.n_consumer_agents = self._get_consumer_count(config)
        self.n_agents = self.n_uc_agents + self.n_consumer_agents
        
        # Initialize circuit and load profile
        self._init_circuit(config)
        self._init_load_profile(config)
        
        # Initialize agent structures
        self._init_agents()
        
        # Initialize spaces
        self._init_action_spaces()
        self._init_observation_spaces()
        
        # Initialize monitoring
        self._init_monitoring(config.get('monitoring_config', {}))
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # Asynchronous execution support
        self.uc_action_buffer = None
        self.consumer_actions_buffer = {}
        self.action_history = deque(maxlen=config.get('history_length', 5))
        
        # Stackelberg game parameters
        self.stackelberg_config = config.get('stackelberg_config', {})
        self.price_bounds = self.stackelberg_config.get('price_bounds', (0.5, 2.0))
        self.dr_incentive_bounds = self.stackelberg_config.get('dr_incentive_bounds', (0.0, 0.5))
        
        # System state tracking
        self.system_state = {}
        self.agent_states = {}
        
    def _get_consumer_count(self, config: Dict[str, Any]) -> int:
        """Determine number of consumer agents based on system size."""
        consumer_counts = {
            '13Bus': config.get('n_consumer_agents_13bus', 5),
            '34Bus': config.get('n_consumer_agents_34bus', 10),
            '123Bus': config.get('n_consumer_agents_123bus', 20),
            '8500Node': config.get('n_consumer_agents_8500node', 50)
        }
        return consumer_counts.get(self.system_name, 10)
    
    def _init_circuit(self, config: Dict[str, Any]):
        """Initialize the power system circuit."""
        self.circuit = Circuits(
            os.path.join(self.dss_folder_path, self.dss_file),
            RB_act_num=(
                config.get('reg_act_num', 33),
                config.get('bat_act_num', 33)
            ),
            dss_act=config.get('dss_act', False)
        )
        
        # Get system information
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.n_buses = len(self.all_bus_names)
        
        # Build network topology
        self.topology = self.circuit.topology
        
    def _init_load_profile(self, config: Dict[str, Any]):
        """Initialize load profiles for the system."""
        self.load_profile = LoadProfile(
            self.max_episode_steps,
            self.dss_folder_path,
            self.dss_file,
            config.get('use_load_noise', True),
            worker_idx=config.get('worker_idx', None)
        )
        
        self.num_profiles = self.load_profile.gen_loadprofile(
            use_noise=config.get('use_load_noise', True),
            scale=config.get('scale', 1.0)
        )
        
    def _init_agents(self):
        """Initialize agent structures."""
        # UC agent (index 0)
        self.uc_agent_id = 0
        
        # Consumer agents (indices 1 to n_consumer_agents)
        self.consumer_agent_ids = list(range(1, self.n_consumer_agents + 1))
        
        # Agent type mapping
        self.agent_types = {self.uc_agent_id: 'uc'}
        for agent_id in self.consumer_agent_ids:
            self.agent_types[agent_id] = 'consumer'
        
        # Initialize load aggregation mapping
        self._init_load_aggregation()
        
    def _init_load_aggregation(self):
        """Initialize load-to-agent mapping for aggregation."""
        self.load_to_agent = {}
        self.agent_to_loads = defaultdict(list)
        
        # Get all loads in the system
        all_loads = list(self.circuit.loads.keys())
        n_loads = len(all_loads)
        
        if n_loads == 0:
            return
        
        # Simple round-robin assignment for now
        # TODO: Implement zone/priority/graph-based aggregation
        for i, load_name in enumerate(all_loads):
            agent_id = (i % self.n_consumer_agents) + 1  # Start from 1
            self.load_to_agent[load_name] = agent_id
            self.agent_to_loads[agent_id].append(load_name)
    
    def _init_action_spaces(self):
        """Initialize action spaces for UC and consumers."""
        self.action_spaces = {}
        
        # UC action space: [price_signal, dr_incentive, capacity_allocation]
        uc_action_dim = 3  # Simplified for now
        self.action_spaces[self.uc_agent_id] = gym.spaces.Box(
            low=np.array([self.price_bounds[0], self.dr_incentive_bounds[0], 0.0]),
            high=np.array([self.price_bounds[1], self.dr_incentive_bounds[1], 1.0]),
            dtype=np.float32
        )
        
        # Consumer action spaces: [load_adjustment, der_output, storage_action]
        consumer_action_dim = 3  # Simplified for now
        for agent_id in self.consumer_agent_ids:
            self.action_spaces[agent_id] = gym.spaces.Box(
                low=np.array([-1.0, 0.0, -1.0]),  # load_adj, der_out, storage
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            )
    
    def _init_observation_spaces(self):
        """Initialize observation spaces for UC and consumers."""
        self.observation_spaces = {}
        
        # UC observation: system-wide information
        uc_obs_dim = 10 + self.n_buses * 2  # System stats + bus voltages/powers
        self.observation_spaces[self.uc_agent_id] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(uc_obs_dim,),
            dtype=np.float32
        )
        
        # Consumer observations: local information + UC signals
        consumer_obs_dim = 15  # Local state + price signals + comfort
        for agent_id in self.consumer_agent_ids:
            self.observation_spaces[agent_id] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(consumer_obs_dim,),
                dtype=np.float32
            )
    
    def _init_monitoring(self, monitoring_config: Dict[str, Any]):
        """Initialize monitoring system."""
        self.monitoring = {
            'enabled': monitoring_config.get('enabled', True),
            'metrics': monitoring_config.get('metrics', [
                'power_loss', 'voltage_violation', 'carbon_emission',
                'nash_gap', 'social_welfare', 'convergence'
            ]),
            'log_interval': monitoring_config.get('log_interval', 10),
            'save_interval': monitoring_config.get('save_interval', 100),
            'history': defaultdict(list),
            'episode_stats': defaultdict(list)
        }
        
        # Initialize metric tracking
        for metric in self.monitoring['metrics']:
            self.monitoring['history'][metric] = []
    
    def reset(self, load_profile_idx: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Reset the environment for a new episode.
        
        Args:
            load_profile_idx: Index of load profile to use
            
        Returns:
            Initial observations for all agents
        """
        # Reset time
        self.current_step = 0
        self.episode_count += 1
        
        # Choose load profile
        if load_profile_idx is None:
            load_profile_idx = np.random.randint(0, self.num_profiles)
        self.load_profile.choose_loadprofile(load_profile_idx, 
                                           self.config.get('use_load_noise', True))
        self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)
        
        # Reset circuit
        self.circuit.reset()
        
        # Clear buffers
        self.uc_action_buffer = None
        self.consumer_actions_buffer.clear()
        self.action_history.clear()
        
        # Get initial system state
        self._update_system_state()
        
        # Get initial observations
        observations = self._get_observations()
        
        # Reset monitoring for new episode
        if self.monitoring['enabled']:
            self._reset_episode_monitoring()
        
        return observations
    
    def step_uc(self, uc_action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Execute UC (leader) action.
        
        Args:
            uc_action: UC's action (price signal, DR incentive, etc.)
            
        Returns:
            uc_reward: Immediate reward for UC
            uc_info: Additional information
        """
        # Validate action
        uc_action = np.clip(uc_action, 
                           self.action_spaces[self.uc_agent_id].low,
                           self.action_spaces[self.uc_agent_id].high)
        
        # Store UC action in buffer
        self.uc_action_buffer = uc_action
        
        # Calculate immediate UC reward (partial, before consumer response)
        uc_reward = self._calculate_uc_immediate_reward(uc_action)
        
        # Prepare info
        uc_info = {
            'price_signal': uc_action[0],
            'dr_incentive': uc_action[1],
            'capacity_allocation': uc_action[2],
            'step': self.current_step
        }
        
        return uc_reward, uc_info
    
    def step_consumers(self, consumer_actions: Dict[int, np.ndarray]) -> Tuple[
        Dict[int, float], Dict[int, Dict[str, Any]], bool
    ]:
        """
        Execute consumer (follower) actions after observing UC action.
        
        Args:
            consumer_actions: Dictionary mapping agent_id to action
            
        Returns:
            rewards: Rewards for all agents (UC + consumers)
            infos: Information dictionaries for all agents
            done: Whether episode is finished
        """
        # Validate consumer actions
        for agent_id, action in consumer_actions.items():
            if agent_id in self.consumer_agent_ids:
                consumer_actions[agent_id] = np.clip(
                    action,
                    self.action_spaces[agent_id].low,
                    self.action_spaces[agent_id].high
                )
        
        # Store consumer actions
        self.consumer_actions_buffer = consumer_actions
        
        # Execute combined actions in the circuit
        self._execute_circuit_actions()
        
        # Update system state
        self._update_system_state()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Prepare info
        infos = self._prepare_step_info()
        
        # Update monitoring
        if self.monitoring['enabled']:
            self._update_monitoring(rewards, infos)
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        
        # Store action history
        self.action_history.append({
            'uc_action': self.uc_action_buffer.copy(),
            'consumer_actions': {k: v.copy() for k, v in consumer_actions.items()},
            'step': self.current_step - 1
        })
        
        return rewards, infos, done
    
    def get_observations(self) -> Dict[int, np.ndarray]:
        """Get current observations for all agents."""
        return self._get_observations()
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Internal method to compute observations."""
        observations = {}
        
        # UC observation
        observations[self.uc_agent_id] = self._get_uc_observation()
        
        # Consumer observations
        for agent_id in self.consumer_agent_ids:
            observations[agent_id] = self._get_consumer_observation(agent_id)
        
        return observations
    
    def _get_uc_observation(self) -> np.ndarray:
        """Get observation for UC agent."""
        obs_components = []
        
        # System-wide metrics
        obs_components.extend([
            self.system_state.get('total_load', 0.0),
            self.system_state.get('total_generation', 0.0),
            self.system_state.get('power_loss_ratio', 0.0),
            self.system_state.get('min_voltage', 0.95),
            self.system_state.get('max_voltage', 1.05),
            self.system_state.get('voltage_violations', 0.0),
            self.current_step / self.max_episode_steps,  # Time progress
            np.sin(2 * np.pi * self.current_step / 24),  # Time of day encoding
            np.cos(2 * np.pi * self.current_step / 24),
            self.system_state.get('carbon_intensity', 0.5)
        ])
        
        # Bus voltages (simplified - min/max per bus)
        bus_voltages = self.system_state.get('bus_voltages', {})
        for bus_name in sorted(self.all_bus_names):
            if bus_name in bus_voltages:
                voltages = bus_voltages[bus_name]
                obs_components.extend([min(voltages), max(voltages)])
            else:
                obs_components.extend([1.0, 1.0])
        
        return np.array(obs_components, dtype=np.float32)
    
    def _get_consumer_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a consumer agent."""
        obs_components = []
        
        # UC signals (if available)
        if self.uc_action_buffer is not None:
            obs_components.extend(self.uc_action_buffer)
        else:
            obs_components.extend([1.0, 0.0, 0.5])  # Default values
        
        # Local state information
        agent_loads = self.agent_to_loads.get(agent_id, [])
        if agent_loads:
            # Aggregate load information
            total_load = sum(self.circuit.loads[load].feature[1] 
                           for load in agent_loads if load in self.circuit.loads)
            avg_voltage = np.mean([
                np.mean(self.system_state.get('bus_voltages', {}).get(
                    self.circuit.loads[load].bus1, [1.0]
                ))
                for load in agent_loads if load in self.circuit.loads
            ])
        else:
            total_load = 0.0
            avg_voltage = 1.0
        
        obs_components.extend([
            total_load / 1000.0,  # Normalize to MW
            avg_voltage,
            self.current_step / self.max_episode_steps,
            np.sin(2 * np.pi * self.current_step / 24),
            np.cos(2 * np.pi * self.current_step / 24),
        ])
        
        # Historical actions (last 2 steps)
        if len(self.action_history) >= 1:
            last_actions = self.action_history[-1].get('consumer_actions', {})
            if agent_id in last_actions:
                obs_components.extend(last_actions[agent_id])
            else:
                obs_components.extend([0.0, 0.0, 0.0])
        else:
            obs_components.extend([0.0, 0.0, 0.0])
        
        if len(self.action_history) >= 2:
            last_actions = self.action_history[-2].get('consumer_actions', {})
            if agent_id in last_actions:
                obs_components.extend(last_actions[agent_id])
            else:
                obs_components.extend([0.0, 0.0, 0.0])
        else:
            obs_components.extend([0.0, 0.0, 0.0])
        
        # Pad to fixed size
        obs_array = np.array(obs_components, dtype=np.float32)
        if len(obs_array) < 15:
            obs_array = np.pad(obs_array, (0, 15 - len(obs_array)), 'constant')
        
        return obs_array[:15]
    
    def _execute_circuit_actions(self):
        """Execute the combined UC and consumer actions in the circuit."""
        # This is a simplified implementation
        # In practice, this would translate agent actions to circuit control actions
        
        # Example: Adjust loads based on consumer actions
        for agent_id, action in self.consumer_actions_buffer.items():
            if agent_id in self.consumer_agent_ids:
                load_adjustment = action[0]  # First component is load adjustment
                
                # Apply to associated loads
                for load_name in self.agent_to_loads.get(agent_id, []):
                    if load_name in self.circuit.loads:
                        # Simplified load adjustment
                        # In practice, this would be more sophisticated
                        pass
        
        # Solve circuit
        self.circuit.dss.ActiveCircuit.Solution.Solve()
    
    def _update_system_state(self):
        """Update system state after circuit solution."""
        # Get bus voltages
        bus_voltages = {}
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            # Extract phase voltages (skip angle values)
            bus_voltages[bus_name] = [
                bus_voltages[bus_name][i] 
                for i in range(len(bus_voltages[bus_name])) 
                if i % 2 == 0
            ]
        
        # Calculate system metrics
        total_loss = self.circuit.total_loss()[0]
        total_power = self.circuit.total_power()[0]
        power_loss_ratio = -total_loss / total_power if total_power != 0 else 0
        
        # Find voltage violations
        voltage_violations = 0
        min_voltage = float('inf')
        max_voltage = float('-inf')
        
        for voltages in bus_voltages.values():
            for v in voltages:
                if v < 0.95 or v > 1.05:
                    voltage_violations += 1
                min_voltage = min(min_voltage, v)
                max_voltage = max(max_voltage, v)
        
        # Update system state
        self.system_state = {
            'bus_voltages': bus_voltages,
            'total_loss': total_loss,
            'total_power': abs(total_power),
            'power_loss_ratio': power_loss_ratio,
            'voltage_violations': voltage_violations,
            'min_voltage': min_voltage,
            'max_voltage': max_voltage,
            'total_load': sum(load.feature[1] for load in self.circuit.loads.values()),
            'total_generation': abs(total_power),
            'carbon_intensity': 0.5  # Placeholder - would calculate based on generation mix
        }
    
    def _calculate_uc_immediate_reward(self, uc_action: np.ndarray) -> float:
        """Calculate immediate reward for UC before consumer response."""
        # Penalty for extreme pricing
        price_penalty = 0.0
        if uc_action[0] < 0.7 or uc_action[0] > 1.5:
            price_penalty = -0.1 * abs(uc_action[0] - 1.0)
        
        # Cost of DR incentives
        dr_cost = -uc_action[1] * 10.0  # Scaled cost
        
        return price_penalty + dr_cost
    
    def _calculate_rewards(self) -> Dict[int, float]:
        """Calculate rewards for all agents."""
        rewards = {}
        
        # UC reward
        rewards[self.uc_agent_id] = self._calculate_uc_reward()
        
        # Consumer rewards
        for agent_id in self.consumer_agent_ids:
            rewards[agent_id] = self._calculate_consumer_reward(agent_id)
        
        return rewards
    
    def _calculate_uc_reward(self) -> float:
        """Calculate UC reward based on system state."""
        # Power loss penalty
        power_loss_reward = -self.system_state['power_loss_ratio'] * 10.0
        
        # Voltage violation penalty
        voltage_reward = -self.system_state['voltage_violations'] * 0.5
        
        # Revenue from electricity sales (simplified)
        if self.uc_action_buffer is not None:
            price = self.uc_action_buffer[0]
            revenue = price * self.system_state['total_load'] / 1000.0 * 0.01
        else:
            revenue = 0.0
        
        # Carbon emission penalty (simplified)
        carbon_penalty = -self.system_state['carbon_intensity'] * 2.0
        
        return power_loss_reward + voltage_reward + revenue + carbon_penalty
    
    def _calculate_consumer_reward(self, agent_id: int) -> float:
        """Calculate consumer reward."""
        if agent_id not in self.consumer_actions_buffer:
            return 0.0
        
        action = self.consumer_actions_buffer[agent_id]
        
        # Cost of electricity (based on UC price)
        if self.uc_action_buffer is not None:
            price = self.uc_action_buffer[0]
            dr_incentive = self.uc_action_buffer[1]
        else:
            price = 1.0
            dr_incentive = 0.0
        
        # Load adjustment cost (comfort penalty)
        load_adjustment = action[0]
        comfort_penalty = -abs(load_adjustment) * 2.0
        
        # Economic benefit from DR participation
        dr_benefit = dr_incentive * abs(load_adjustment) * 10.0
        
        # Electricity cost
        agent_loads = self.agent_to_loads.get(agent_id, [])
        total_load = sum(
            self.circuit.loads[load].feature[1] 
            for load in agent_loads 
            if load in self.circuit.loads
        ) / 1000.0  # Convert to MW
        
        electricity_cost = -price * total_load * (1 + load_adjustment)
        
        return comfort_penalty + dr_benefit + electricity_cost
    
    def _prepare_step_info(self) -> Dict[int, Dict[str, Any]]:
        """Prepare information dictionaries for all agents."""
        infos = {}
        
        # UC info
        infos[self.uc_agent_id] = {
            'power_loss_ratio': self.system_state['power_loss_ratio'],
            'voltage_violations': self.system_state['voltage_violations'],
            'total_load': self.system_state['total_load'],
            'carbon_intensity': self.system_state['carbon_intensity'],
            'step': self.current_step
        }
        
        # Consumer info
        for agent_id in self.consumer_agent_ids:
            agent_loads = self.agent_to_loads.get(agent_id, [])
            infos[agent_id] = {
                'n_loads': len(agent_loads),
                'total_load': sum(
                    self.circuit.loads[load].feature[1]
                    for load in agent_loads
                    if load in self.circuit.loads
                ),
                'step': self.current_step
            }
        
        return infos
    
    def _reset_episode_monitoring(self):
        """Reset monitoring for new episode."""
        # Store episode statistics
        if self.episode_count > 1:
            for metric, values in self.monitoring['history'].items():
                if values:
                    self.monitoring['episode_stats'][f'{metric}_mean'].append(np.mean(values))
                    self.monitoring['episode_stats'][f'{metric}_std'].append(np.std(values))
                    self.monitoring['episode_stats'][f'{metric}_min'].append(np.min(values))
                    self.monitoring['episode_stats'][f'{metric}_max'].append(np.max(values))
        
        # Clear history for new episode
        for metric in self.monitoring['metrics']:
            self.monitoring['history'][metric].clear()
    
    def _update_monitoring(self, rewards: Dict[int, float], infos: Dict[int, Dict[str, Any]]):
        """Update monitoring metrics."""
        # Power loss
        if 'power_loss' in self.monitoring['metrics']:
            self.monitoring['history']['power_loss'].append(
                self.system_state['power_loss_ratio']
            )
        
        # Voltage violations
        if 'voltage_violation' in self.monitoring['metrics']:
            self.monitoring['history']['voltage_violation'].append(
                self.system_state['voltage_violations']
            )
        
        # Carbon emission (placeholder)
        if 'carbon_emission' in self.monitoring['metrics']:
            self.monitoring['history']['carbon_emission'].append(
                self.system_state['carbon_intensity'] * self.system_state['total_generation']
            )
        
        # Nash gap (simplified - difference between UC and average consumer reward)
        if 'nash_gap' in self.monitoring['metrics']:
            uc_reward = rewards[self.uc_agent_id]
            avg_consumer_reward = np.mean([
                rewards[aid] for aid in self.consumer_agent_ids
            ])
            nash_gap = abs(uc_reward - avg_consumer_reward)
            self.monitoring['history']['nash_gap'].append(nash_gap)
        
        # Social welfare (total rewards)
        if 'social_welfare' in self.monitoring['metrics']:
            social_welfare = sum(rewards.values())
            self.monitoring['history']['social_welfare'].append(social_welfare)
        
        # Log if needed
        if self.current_step % self.monitoring['log_interval'] == 0:
            self._log_monitoring_metrics()
    
    def _log_monitoring_metrics(self):
        """Log current monitoring metrics."""
        log_str = f"Episode {self.episode_count}, Step {self.current_step}: "
        for metric in self.monitoring['metrics']:
            if metric in self.monitoring['history'] and self.monitoring['history'][metric]:
                recent_value = self.monitoring['history'][metric][-1]
                log_str += f"{metric}={recent_value:.4f}, "
        print(log_str.rstrip(', '))
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring metrics."""
        summary = {
            'episode': self.episode_count,
            'current_metrics': {},
            'episode_stats': dict(self.monitoring['episode_stats'])
        }
        
        for metric in self.monitoring['metrics']:
            if metric in self.monitoring['history'] and self.monitoring['history'][metric]:
                summary['current_metrics'][metric] = {
                    'current': self.monitoring['history'][metric][-1],
                    'mean': np.mean(self.monitoring['history'][metric]),
                    'std': np.std(self.monitoring['history'][metric]),
                    'min': np.min(self.monitoring['history'][metric]),
                    'max': np.max(self.monitoring['history'][metric])
                }
        
        return summary
    
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to file."""
        monitoring_data = {
            'config': self.config,
            'episode_count': self.episode_count,
            'monitoring_history': dict(self.monitoring['history']),
            'episode_stats': dict(self.monitoring['episode_stats']),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(monitoring_data, f, indent=2, default=str)
    
    def render(self, mode: str = 'human'):
        """Render the environment (placeholder for visualization)."""
        if mode == 'human':
            # Could implement network visualization here
            pass
        elif mode == 'rgb_array':
            # Return image array for video recording
            pass
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self