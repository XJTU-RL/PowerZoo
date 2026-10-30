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


def load_stackelberg_config(env_name, config_path=None):
    from utils.config_pipeline import ConfigLoader
    return ConfigLoader().load('happo', env_name, env_config=config_path).env_args


def create_default_config(env_name):
    return load_stackelberg_config(env_name)


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