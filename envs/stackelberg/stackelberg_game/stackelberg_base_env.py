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
- PowerZoo/MARL framework compatible interface

Compatibility:
- gym/gymnasium API compatible (reset returns (obs, info), step returns 5-tuple)
- PowerZoo MARL interface (share_observation_space, get_avail_actions)
- HAPPO algorithm compatible (agent_types, heterogeneous spaces)
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from datetime import datetime
import json

# Conditional gym/gymnasium import for compatibility
try:
	import gymnasium as gym
	from gymnasium.spaces import Box, Discrete
except ImportError:
	import gym
	from gym.spaces import Box, Discrete

from envs.powerzoo.powerzoo.circuit import Circuits
from envs.powerzoo.powerzoo.loadprofile import LoadProfile
from envs.stackelberg.stackelberg_game.circuit_adapter import StackelbergCircuitAdapter

logger = logging.getLogger('StackelbergBaseEnv')


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
        self.seed = config.get('seed', None)
        self.system_name = config['system_name']
        self.dss_file = config['dss_file']
        self.max_episode_steps = config.get('max_episode_steps', 24)
        
        base_path = config.get('base_path', 'envs/powerzoo/systems')
        self.dss_folder_path = os.path.join(base_path, self.system_name)
        config['dss_file_abs_path'] = os.path.abspath(os.path.join(self.dss_folder_path, self.dss_file))

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

        # NOTE: PER (Prioritized Experience Replay) has been removed from environment.
        # PER is an algorithm-level feature and should be implemented in the algorithm/buffer.

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
        
        # ESS (Energy Storage System) parameters
        self.ess_config = config.get('ess_config', {})
        self.ess_capacity = self.ess_config.get('capacity', 100.0)  # MWh
        self.ess_max_power = self.ess_config.get('max_power', 20.0)  # MW
        self.ess_efficiency_charge = self.ess_config.get('eta_c', 0.95)
        self.ess_efficiency_discharge = self.ess_config.get('eta_o', 0.95)
        self.ess_decay_rate = self.ess_config.get('eta_s', 0.99)
        self.ess_soc = 0.5  # Initial State of Charge (50%)
        
        # DER generation profile
        self.der_profile = config.get('der_profile', None)
        
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
        """
        Initialize circuit from DSS file.

        Raises:
            FileNotFoundError: If DSS file doesn't exist
            RuntimeError: If circuit initialization fails
        """
        dss_path = config['dss_file_abs_path']

        if not os.path.exists(dss_path):
            raise FileNotFoundError(f"DSS file not found: {dss_path}")

        try:
            self.circuit = Circuits(
                dss_file=dss_path,
                dss_act=config.get('dss_act', False)
            )

            # Create circuit adapter for action translation
            self.circuit_adapter = StackelbergCircuitAdapter(
                self.circuit,
                config=config.get('circuit_adapter_config', {})
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize circuit from {dss_path}: {e}")

        # Get system information
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.n_buses = len(self.all_bus_names)

        # Build network topology
        self.topology = self.circuit.topology

        logger.info(f"Circuit initialized: {self.n_buses} buses, {len(self.circuit.loads)} loads")
        
    def _init_load_profile(self, config: Dict[str, Any]):
        """Initialize load profile."""
        abs_path = config['dss_file_abs_path']
        assert os.path.exists(abs_path), f"DSS file not found at: {abs_path}"
        with open(abs_path, 'r') as f:
            dss_content = f.read()

        self.load_profile = LoadProfile(
            self.max_episode_steps,
            self.dss_folder_path,
            dss_content,
            dss_path=abs_path,
            use_noise=config.get('use_load_noise', True),
            worker_idx=config.get('worker_idx', None)
        )
        self.all_load_profiles = self.load_profile.get_loadprofile(0)
        
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
    
    def _build_agent_to_loads(self):
        """Build reverse mapping from agents to loads."""
        self.agent_to_loads.clear()
        
        for load_name, agent_id in self.load_to_agent.items():
            self.agent_to_loads[agent_id].append(load_name)
    
    def _init_action_spaces(self):
        """Initialize action spaces for UC and consumers."""
        self.action_spaces = {}
        
        # UC action space based on paper: [p_b, p_s, p_a, p_c, p_d, p_e, p_o, p_m]
        # For practical implementation, we use a simplified version:
        # [price_multiplier, dr_incentive, dr_target, ess_charge, ess_discharge]
        uc_action_dim = 5
        self.action_spaces[self.uc_agent_id] = gym.spaces.Box(
            low=np.array([
                self.price_bounds[0],          # price_multiplier
                self.dr_incentive_bounds[0],   # dr_incentive
                0.0,                          # dr_target
                0.0,                          # ess_charge
                0.0                           # ess_discharge
            ]),
            high=np.array([
                self.price_bounds[1],          # price_multiplier
                self.dr_incentive_bounds[1],   # dr_incentive
                1.0,                          # dr_target (fraction of total load)
                1.0,                          # ess_charge
                1.0                           # ess_discharge
            ]),
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
            load_profile_idx = np.random.randint(0, self.load_profile.num_profiles)
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
        """
        Get observation for a consumer agent.

        Args:
            agent_id: Consumer agent ID

        Returns:
            Observation array of shape (15,)
        """
        obs_components = []

        # UC signals (if available) - must match UC action space dimension (5)
        if self.uc_action_buffer is not None:
            obs_components.extend(self.uc_action_buffer)
        else:
            # Default values: [price_mult=1.0, dr_incentive=0.0, dr_target=0.5, ess_charge=0.0, ess_discharge=0.0]
            obs_components.extend([1.0, 0.0, 0.5, 0.0, 0.0])
        
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
        """
        Execute the combined UC and consumer actions in the circuit.

        Uses the circuit adapter to translate agent actions to DSS controls.
        """
        # Apply UC actions (ESS control, etc.)
        if self.uc_action_buffer is not None:
            self.circuit_adapter.apply_uc_actions(self.uc_action_buffer)

        # Apply consumer actions (load adjustments)
        if self.consumer_actions_buffer:
            self.circuit_adapter.apply_consumer_actions(
                self.consumer_actions_buffer,
                self.load_to_agent
            )

        # Solve circuit with updated settings
        try:
            self.circuit.dss.ActiveCircuit.Solution.Solve()

            # Check for convergence
            if not self.circuit.dss.ActiveCircuit.Solution.Converged:
                logger.warning("Circuit solution did not converge")

        except Exception as e:
            logger.error(f"Circuit solve failed: {e}")
    
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
        
        # Update ESS state if UC actions available
        if self.uc_action_buffer is not None and len(self.uc_action_buffer) >= 5:
            self._update_ess_state(self.uc_action_buffer[3], self.uc_action_buffer[4])
        
        # Get DER generation
        der_generation = self._get_der_generation()
        
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
            'der_generation': der_generation,
            'ess_soc': self.ess_soc,
            'carbon_intensity': self._calculate_carbon_intensity(der_generation, abs(total_power))
        }
    
    def _update_ess_state(self, charge_action: float, discharge_action: float):
        """
        Update ESS state based on UC actions.
        Implements Equation 19 from the paper.
        """
        # Ensure charge and discharge don't happen simultaneously
        if charge_action > 0 and discharge_action > 0:
            # Prioritize discharge
            charge_action = 0.0
        
        # Calculate actual power considering constraints
        charge_power = min(charge_action * self.ess_max_power, 
                          (1.0 - self.ess_soc) * self.ess_capacity)
        discharge_power = min(discharge_action * self.ess_max_power,
                            self.ess_soc * self.ess_capacity)
        
        # Update SOC based on Equation 19
        # S_t = (1 - η_s) * S_{t-1} + η_c * p_c - p_o / η_o
        delta_t = 1.0  # 1 hour time step
        
        self.ess_soc = (self.ess_decay_rate * self.ess_soc + 
                       self.ess_efficiency_charge * charge_power * delta_t / self.ess_capacity -
                       discharge_power * delta_t / (self.ess_efficiency_discharge * self.ess_capacity))
        
        # Constrain SOC between 0 and 1
        self.ess_soc = np.clip(self.ess_soc, 0.0, 1.0)
    
    def _get_der_generation(self) -> float:
        """Get current DER generation based on time step."""
        if self.der_profile is not None:
            # Use provided profile
            return self.der_profile[self.current_step % len(self.der_profile)]
        else:
            # Simple solar generation model
            hour = self.current_step % 24
            if 6 <= hour <= 18:  # Daylight hours
                # Peak at noon
                solar_gen = 50.0 * np.sin(np.pi * (hour - 6) / 12)  # MW
            else:
                solar_gen = 0.0
            return solar_gen * 1000  # Convert to kW
    
    def _calculate_carbon_intensity(self, der_generation: float, total_generation: float) -> float:
        """Calculate carbon intensity based on generation mix."""
        if total_generation <= 0:
            return 0.5
        
        # DER (renewable) has zero carbon
        renewable_fraction = min(der_generation / total_generation, 1.0)
        
        # Grid carbon intensity (kg CO2/MWh)
        grid_carbon_intensity = 0.5  # Default value
        
        # Weighted average
        return grid_carbon_intensity * (1 - renewable_fraction)
    
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
        """
        Calculate UC reward based on paper equations (5-10).
        UC utility = C_t^s + C_t^m + C_t^g + C_t^r
        """
        if self.uc_action_buffer is None:
            return 0.0
        
        # Extract UC actions
        # uc_action = [price_multiplier, dr_incentive, dr_target, ess_charge, ess_discharge]
        price_multiplier = self.uc_action_buffer[0]  # Price signal
        dr_incentive = self.uc_action_buffer[1]      # DR incentive
        dr_target = self.uc_action_buffer[2] if len(self.uc_action_buffer) > 2 else 0.5
        ess_charge = self.uc_action_buffer[3] if len(self.uc_action_buffer) > 3 else 0.0
        ess_discharge = self.uc_action_buffer[4] if len(self.uc_action_buffer) > 4 else 0.0
        
        # Get time of day for TUTT pricing
        hour = self.current_step % 24
        
        # C_t^s: Revenue from selling electricity (Equation 7)
        # Using Time-of-Use Tiered Tariff (TUTT)
        C_t_s = self._calculate_electricity_revenue(price_multiplier, hour)
        
        # C_t^m: Cost of purchasing from power grid (Equation 8)
        # Including uncertainty epsilon
        base_market_price = self._get_market_price(hour)
        uncertainty = np.random.normal(0, 0.03)  # 3% std as per paper
        market_price = base_market_price * (1 + np.clip(uncertainty, -0.03, 0.03))
        total_purchase = self.system_state.get('total_power', 0) / 1000.0  # MW
        C_t_m = -market_price * total_purchase
        
        # C_t_g: DER absorption profit (Equation 9)
        # T_d(p_g) = T_1^d + T_2^d * p_g (linear DER tariff)
        der_generation = self.system_state.get('der_generation', 0) / 1000.0  # MW
        der_absorbed = der_generation * 0.9  # Assume 90% absorption
        der_curtailed = der_generation * 0.1
        T_1_d = 0.05  # Base DER tariff ($/kWh)
        T_2_d = -0.001  # DER tariff slope
        T_a = 0.02  # Curtailment cost
        
        der_tariff = T_1_d + T_2_d * der_generation
        C_t_g = der_tariff * der_absorbed - T_a * der_curtailed
        
        # C_t_r: DR flexibility service cost (Equation 10)
        # Quadratic subsidy structure
        total_dr_response = self._calculate_total_dr_response()
        dr_target_mw = dr_target * self.system_state.get('total_load', 0) / 1000.0  # MW target
        T_s = 0.01  # DR subsidy rate
        T_r = dr_incentive  # DR flexibility purchase price
        
        if dr_target_mw > 0:
            C_t_r = T_s * (total_dr_response ** 2) / dr_target_mw - T_r * total_dr_response
        else:
            C_t_r = -T_r * total_dr_response
        
        # Total UC utility
        total_utility = C_t_s + C_t_m + C_t_g + C_t_r
        
        # Add power quality penalties (not in paper but important for system)
        voltage_penalty = -self.system_state['voltage_violations'] * 0.1
        loss_penalty = -self.system_state['power_loss_ratio'] * 5.0
        
        return total_utility + voltage_penalty + loss_penalty
    
    def _calculate_electricity_revenue(self, price_multiplier: float, hour: int) -> float:
        """
        Calculate revenue from selling electricity with TUTT pricing.
        Based on Equation 7 in the paper.
        """
        # Time-of-Use periods (based on paper Table II)
        peak_hours = [20, 21, 22]  # 20:00-22:00
        high_hours = list(range(9, 16))  # 09:00-15:00
        
        # Base tariffs ($/kWh) - from paper Table II
        if hour in peak_hours:
            base_tariff = 0.078  # Peak period
        elif hour in high_hours:
            base_tariff = 0.068  # High period
        else:
            base_tariff = 0.048  # Flat period
        
        # Apply price multiplier
        actual_tariff = base_tariff * price_multiplier
        
        # Calculate total consumer consumption
        total_consumption = 0.0
        for agent_id in self.consumer_agent_ids:
            agent_loads = self.agent_to_loads.get(agent_id, [])
            for load_name in agent_loads:
                if load_name in self.circuit.loads:
                    total_consumption += self.circuit.loads[load_name].feature[1]
        
        total_consumption = total_consumption / 1000.0  # Convert to MW
        
        # Apply tiered pricing based on cumulative consumption
        # Tier 1: 0-28.8 GWh, Tier 2: 28.8-48 GWh, Tier 3: >48 GWh
        cumulative_consumption = self.agent_states.get('cumulative_consumption', 0.0)
        
        if cumulative_consumption < 28800:  # Tier 1
            tier_factor = 1.0
        elif cumulative_consumption < 48000:  # Tier 2
            tier_factor = 0.9
        else:  # Tier 3
            tier_factor = 0.8
        
        revenue = actual_tariff * total_consumption * tier_factor
        
        # Update cumulative consumption
        self.agent_states['cumulative_consumption'] = cumulative_consumption + total_consumption
        
        return revenue
    
    def _get_market_price(self, hour: int) -> float:
        """Get base market price for electricity purchase."""
        # Simplified market price model based on time of day
        if hour in [20, 21, 22]:  # Peak
            return 0.12
        elif hour in range(9, 16):  # High
            return 0.09
        else:  # Flat
            return 0.06
    
    def _calculate_total_dr_response(self) -> float:
        """Calculate total demand response from all consumers."""
        total_dr = 0.0
        for agent_id in self.consumer_agent_ids:
            if agent_id in self.consumer_actions_buffer:
                action = self.consumer_actions_buffer[agent_id]
                load_adjustment = action[0]  # First component is load adjustment
                
                # Get agent's base load
                agent_loads = self.agent_to_loads.get(agent_id, [])
                base_load = sum(
                    self.circuit.loads[load].feature[1]
                    for load in agent_loads
                    if load in self.circuit.loads
                ) / 1000.0  # MW
                
                dr_amount = abs(load_adjustment) * base_load
                total_dr += dr_amount
        
        return total_dr
    
    def _calculate_consumer_reward(self, agent_id: int) -> float:
        """
        Calculate consumer reward based on paper equations (21-25).
        Consumer utility = C_i,t^p + C_i,t^d + C_i,t^l
        Note: We minimize cost, so return negative of utility
        """
        if agent_id not in self.consumer_actions_buffer:
            return 0.0
        
        action = self.consumer_actions_buffer[agent_id]
        
        # Extract consumer actions
        # action = [load_adjustment, der_output, storage_action]
        load_adjustment = action[0]  # Delta p_i,t^l + Delta p_i,t^s
        der_consumption = action[1] if len(action) > 1 else 0.0  # p_i,t^r
        storage_action = action[2] if len(action) > 2 else 0.0
        
        # Get UC signals
        if self.uc_action_buffer is not None:
            price_multiplier = self.uc_action_buffer[0]
            dr_incentive = self.uc_action_buffer[1]
        else:
            price_multiplier = 1.0
            dr_incentive = 0.0
        
        # Get agent's base load
        agent_loads = self.agent_to_loads.get(agent_id, [])
        base_load = sum(
            self.circuit.loads[load].feature[1]
            for load in agent_loads
            if load in self.circuit.loads
        ) / 1000.0  # MW
        
        # Actual consumption after adjustment
        actual_load = base_load * (1 + load_adjustment)
        
        # C_i,t^p: Cost of purchasing electricity from UC (Equation 23)
        hour = self.current_step % 24
        electricity_tariff = self._get_consumer_tariff(price_multiplier, hour, agent_id)
        C_i_t_p = electricity_tariff * actual_load
        
        # C_i,t^d: DER consumption reward (Equation 24)
        # T_t^d(p_g) = T_1^d + T_2^d * p_g
        der_generation = self.system_state.get('der_generation', 0) / 1000.0  # MW
        T_1_d = 0.05  # Base DER tariff
        T_2_d = -0.001  # DER tariff slope
        der_tariff = T_1_d + T_2_d * der_generation
        
        # Consumer can consume DER up to their share
        max_der_share = der_generation / self.n_consumer_agents
        actual_der_consumption = min(der_consumption * base_load, max_der_share)
        C_i_t_d = -der_tariff * actual_der_consumption  # Negative because it's a benefit
        
        # C_i,t^l: Load scheduling utility with comfort penalty (Equation 25)
        # Quadratic comfort penalty for deviating from base load
        T_i_c = 0.01  # Comfort tariff ($/kW)
        dr_amount = abs(load_adjustment) * base_load
        
        if actual_load > 0:
            comfort_penalty = T_i_c * (dr_amount ** 2) / actual_load
        else:
            comfort_penalty = T_i_c * (dr_amount ** 2)
        
        # DR participation incentive
        T_r = dr_incentive  # DR flexibility purchase price
        dr_benefit = T_r * dr_amount
        
        C_i_t_l = comfort_penalty - dr_benefit
        
        # Total consumer cost (we return negative for RL reward)
        total_cost = C_i_t_p + C_i_t_d + C_i_t_l
        
        # Add voltage quality bonus (not in paper but helps convergence)
        avg_voltage = self._get_agent_avg_voltage(agent_id)
        voltage_bonus = 0.0
        if 0.95 <= avg_voltage <= 1.05:
            voltage_bonus = 0.1
        
        return -total_cost + voltage_bonus
    
    def _get_consumer_tariff(self, price_multiplier: float, hour: int, agent_id: int) -> float:
        """
        Get consumer electricity tariff with TUTT pricing.
        Based on time-of-use and cumulative consumption tiers.
        """
        # Time-of-Use base tariffs
        if hour in [20, 21, 22]:  # Peak
            base_tariff = 0.078
        elif hour in range(9, 16):  # High
            base_tariff = 0.068
        else:  # Flat
            base_tariff = 0.048
        
        # Apply price multiplier from UC
        tariff = base_tariff * price_multiplier
        
        # Get agent's cumulative consumption for tiered pricing
        agent_cumulative = self.agent_states.get(f'consumer_{agent_id}_cumulative', 0.0)
        
        # Apply tier factors based on monthly consumption
        if agent_cumulative < 2880:  # Tier 1: < 2.88 GWh/month
            tier_factor = 1.0
        elif agent_cumulative < 4800:  # Tier 2: 2.88-4.8 GWh/month
            tier_factor = 0.85
        else:  # Tier 3: > 4.8 GWh/month
            tier_factor = 0.75
        
        return tariff * tier_factor
    
    def _get_agent_avg_voltage(self, agent_id: int) -> float:
        """Get average voltage for agent's loads."""
        agent_loads = self.agent_to_loads.get(agent_id, [])
        if not agent_loads:
            return 1.0
        
        voltages = []
        for load_name in agent_loads:
            if load_name in self.circuit.loads:
                bus = self.circuit.loads[load_name].bus1
                bus_voltages = self.system_state.get('bus_voltages', {}).get(bus, [1.0])
                voltages.extend(bus_voltages)
        
        return np.mean(voltages) if voltages else 1.0
    
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

    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            self.config['seed'] = seed
            logger.info(f"Environment seed set to {seed}")

    def close(self):
        """Clean up environment resources."""
        logger.info("Closing StackelbergBaseEnv")