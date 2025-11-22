# -*- coding: utf-8 -*-
"""
Stackelberg Game Environment Package

This package implements a bi-level non-cooperative Stackelberg-Nash game framework
for demand response in power distribution networks.

Key Components:
- StackelbergBaseEnv: Core environment implementation
- StackelbergPowerZooEnv: PowerZoo-compatible wrapper
- AsyncMultiAgentWrapper: Asynchronous multi-agent execution
- StackelbergMonitor: Monitoring and logging
- IntelligentLoadAggregator: Load aggregation strategies
- StackelbergCircuitAdapter: Circuit control interface
"""

from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.stackelberg.stackelberg_game.async_wrapper import AsyncMultiAgentWrapper
from envs.stackelberg.stackelberg_game.stackelberg_monitor import StackelbergMonitor
from envs.stackelberg.stackelberg_game.load_aggregator import IntelligentLoadAggregator
from envs.stackelberg.stackelberg_game.circuit_adapter import StackelbergCircuitAdapter
from envs.stackelberg.stackelberg_game.env_factory import (
    make_stackelberg_env,
    make_stackelberg_13bus,
    make_stackelberg_34bus,
    make_stackelberg_123bus,
    load_stackelberg_config
)
from envs.stackelberg.stackelberg_powerzoo_env import StackelbergPowerZooEnv

__all__ = [
    # Core environments
    'StackelbergBaseEnv',
    'StackelbergPowerZooEnv',
    # Wrappers and utilities
    'AsyncMultiAgentWrapper',
    'StackelbergMonitor',
    'IntelligentLoadAggregator',
    'StackelbergCircuitAdapter',
    # Factory functions
    'make_stackelberg_env',
    'make_stackelberg_13bus',
    'make_stackelberg_34bus',
    'make_stackelberg_123bus',
    'load_stackelberg_config',
]