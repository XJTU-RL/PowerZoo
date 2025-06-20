# -*- coding: utf-8 -*-
"""
Environment Factory for Stackelberg Game

This module provides factory functions to create Stackelberg game environments
with proper configuration and setup.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.stackelberg.stackelberg_game.async_wrapper import AsyncMultiAgentWrapper


def load_stackelberg_config(env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration for Stackelberg environment.
    
    Args:
        env_name: Name of the environment (e.g., 'stackelberg_13bus')
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent.parent.parent.parent / 'configs' / 'envs_cfgs' / f'{env_name}.yaml'
    
    if not os.path.exists(config_path):
        # Try without 'stackelberg_' prefix
        alt_name = env_name.replace('stackelberg_', '')
        config_path = Path(__file__).parent.parent.parent.parent / 'configs' / 'envs_cfgs' / f'stackelberg_{alt_name}.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Create default config
        config = create_default_config(env_name)
    
    return config


def create_default_config(env_name: str) -> Dict[str, Any]:
    """
    Create default configuration for Stackelberg environment.
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Default configuration dictionary
    """
    # Extract system name
    if '13bus' in env_name.lower():
        system_name = '13Bus'
        dss_file = 'IEEE13Nodeckt_daily.dss'
        n_consumers = 8
    elif '34bus' in env_name.lower():
        system_name = '34Bus'
        dss_file = 'ieee34Mod1_daily.dss'
        n_consumers = 12
    elif '123bus' in env_name.lower():
        system_name = '123Bus'
        dss_file = 'IEEE123Master_daily.dss'
        n_consumers = 20
    else:
        system_name = '13Bus'
        dss_file = 'IEEE13Nodeckt_daily.dss'
        n_consumers = 8
    
    config = {
        'env_name': env_name,
        'system_name': system_name,
        'dss_file': dss_file,
        'max_episode_steps': 24,
        'n_consumer_agents': n_consumers,
        
        # UC action space
        'uc_action_space': {
            'price_signal': {'low': 0.5, 'high': 2.0},
            'dr_incentive': {'low': 0.0, 'high': 0.5},
            'capacity_allocation': {'low': 0.0, 'high': 1.0},
            'ess_charge': {'low': -1.0, 'high': 1.0},
            'der_curtailment': {'low': 0.0, 'high': 1.0}
        },
        
        # Consumer action space
        'consumer_action_space': {
            'load_adjustment': {'low': -0.3, 'high': 0.1},
            'der_output': {'low': 0.0, 'high': 1.0}
        },
        
        # Reward weights
        'reward_weights': {
            'uc': {
                'electricity_revenue': 1.0,
                'market_cost': 1.0,
                'der_profit': 0.8,
                'dr_cost': 0.6,
                'system_loss': 0.5,
                'voltage_violation': 2.0,
                'carbon_reduction': 0.3
            },
            'consumer': {
                'electricity_cost': 1.0,
                'comfort_loss': 0.8,
                'dr_revenue': 1.2,
                'voltage_quality': 0.3
            }
        },
        
        # ESS configuration
        'ess_config': {
            'total_capacity': 2.0,
            'initial_soc': 0.5,
            'efficiency_charge': 0.95,
            'efficiency_discharge': 0.95,
            'self_discharge_rate': 0.001,
            'min_soc': 0.2,
            'max_soc': 0.9,
            'max_power': 0.5
        },
        
        # DER configuration
        'der_config': {
            'total_capacity': 3.0,
            'availability_profile': 'solar',
            'forecast_error_std': 0.1,
            'curtailment_cost': 0.02
        },
        
        # Market configuration
        'market_config': {
            'base_price': 0.10,
            'peak_multiplier': 2.0,
            'valley_multiplier': 0.5,
            'market_volatility': 0.03
        },
        
        # Time-of-use configuration
        'tou_config': {
            'peak_hours': [8, 9, 10, 11, 17, 18, 19, 20],
            'valley_hours': [0, 1, 2, 3, 4, 5, 23],
            'normal_hours': [6, 7, 12, 13, 14, 15, 16, 21, 22]
        },
        
        # Monitoring configuration
        'monitoring_config': {
            'enable': True,
            'log_dir': f'logs/{env_name}',
            'save_interval': 100,
            'plot_interval': 50
        },
        
        # Async configuration
        'async_config': {
            'consumer_delay': 1,
            'action_persistence': 1,
            'enable_action_prediction': False
        },
        
        # Other settings
        'seed': 42,
        'debug': False,
        'verbose': True
    }
    
    return config


def make_stackelberg_env(env_name: str, 
                        config: Optional[Dict[str, Any]] = None,
                        use_async_wrapper: bool = True,
                        **kwargs) -> Any:
    """
    Create a Stackelberg game environment.
    
    Args:
        env_name: Name of the environment
        config: Optional configuration dictionary
        use_async_wrapper: Whether to use async wrapper
        **kwargs: Additional keyword arguments
        
    Returns:
        Environment instance (wrapped or unwrapped)
    """
    # Load configuration
    if config is None:
        config = load_stackelberg_config(env_name)
    
    # Update config with kwargs
    config.update(kwargs)
    
    # Create base environment
    base_env = StackelbergBaseEnv(config)
    
    # Optionally wrap with async wrapper
    if use_async_wrapper:
        async_config = config.get('async_config', {})
        env = AsyncMultiAgentWrapper(base_env, async_config)
    else:
        env = base_env
    
    return env


def make_parallel_stackelberg_envs(env_name: str,
                                  n_envs: int,
                                  config: Optional[Dict[str, Any]] = None,
                                  use_async_wrapper: bool = True,
                                  **kwargs) -> list:
    """
    Create multiple parallel Stackelberg environments.
    
    Args:
        env_name: Name of the environment
        n_envs: Number of parallel environments
        config: Optional configuration dictionary
        use_async_wrapper: Whether to use async wrapper
        **kwargs: Additional keyword arguments
        
    Returns:
        List of environment instances
    """
    envs = []
    
    for i in range(n_envs):
        # Create config for this instance
        env_config = load_stackelberg_config(env_name) if config is None else config.copy()
        
        # Update with instance-specific settings
        env_config['seed'] = env_config.get('seed', 42) + i
        env_config['monitoring_config']['experiment_name'] = f"{env_name}_env{i}"
        
        # Update with kwargs
        env_config.update(kwargs)
        
        # Create environment
        env = make_stackelberg_env(env_name, env_config, use_async_wrapper)
        envs.append(env)
    
    return envs


# Convenience functions for specific environments
def make_stackelberg_13bus(**kwargs):
    """Create Stackelberg 13Bus environment."""
    return make_stackelberg_env('stackelberg_13bus', **kwargs)


def make_stackelberg_34bus(**kwargs):
    """Create Stackelberg 34Bus environment."""
    return make_stackelberg_env('stackelberg_34bus', **kwargs)


def make_stackelberg_123bus(**kwargs):
    """Create Stackelberg 123Bus environment."""
    return make_stackelberg_env('stackelberg_123bus', **kwargs)