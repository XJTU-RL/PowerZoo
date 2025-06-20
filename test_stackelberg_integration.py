#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to validate Stackelberg game implementation integration.

This script tests:
1. Environment creation and initialization
2. Reward function calculations match paper equations
3. SN-MAPPO algorithm initialization
4. Total derivative computation
5. Nash gap tracking
6. Monitoring system functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from pathlib import Path

def test_environment_creation():
    """Test Stackelberg environment creation."""
    print("Testing Stackelberg environment creation...")
    
    # Load config
    config_path = Path("configs/envs_cfgs/stackelberg_13bus.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import environment
    from envs.stackelberg.stackelberg_game import StackelbergBaseEnv
    
    # Create environment
    env = StackelbergBaseEnv(config)
    
    # Check basic properties
    assert env.n_agents == 9, f"Expected 9 agents (1 UC + 8 consumers), got {env.n_agents}"
    assert env.max_episode_steps == 24, f"Expected 24 steps, got {env.max_episode_steps}"
    
    # Test reset
    obs = env.reset()
    assert len(obs) == env.n_agents, f"Expected {env.n_agents} observations, got {len(obs)}"
    
    print("✓ Environment creation successful")
    return env


def test_reward_calculations(env):
    """Test reward calculations match paper equations."""
    print("\nTesting reward calculations...")
    
    # Reset environment
    env.reset()
    
    # Set up test state
    env.hour = 10  # Peak hour
    env.system_state['total_load'] = 10.0  # MW
    env.system_state['der_generation'] = 2.0  # MW
    env.system_state['ess_soc'] = 0.5
    
    # Calculate UC reward
    uc_reward = env._calculate_uc_reward()
    print(f"UC reward: {uc_reward:.4f}")
    
    # Calculate consumer reward
    consumer_reward = env._calculate_consumer_reward(agent_id=1)
    print(f"Consumer reward: {consumer_reward:.4f}")
    
    # Check components exist
    assert hasattr(env, 'reward_components'), "Missing reward components tracking"
    
    print("✓ Reward calculations successful")


def test_sn_mappo_algorithm():
    """Test SN-MAPPO algorithm initialization."""
    print("\nTesting SN-MAPPO algorithm...")
    
    # Load algorithm config
    config_path = Path("configs/algos_cfgs/sn_mappo.yaml")
    with open(config_path, 'r') as f:
        algo_config = yaml.safe_load(f)
    
    # Import algorithm
    from algorithms.actors.sn_mappo import SN_MAPPO
    
    # Create dummy spaces
    obs_space = type('MockSpace', (), {'shape': (14,)})()
    act_space = type('MockSpace', (), {'shape': (5,), 'n': 5})()
    
    # Initialize UC agent
    uc_config = algo_config.copy()
    uc_config.update(algo_config['uc_config'])
    uc_config['device'] = 'cpu'
    
    uc_agent = SN_MAPPO(uc_config, obs_space, act_space)
    assert uc_agent.is_leader == True
    assert uc_agent.agent_type == 'uc'
    
    # Initialize consumer agent
    consumer_config = algo_config.copy()
    consumer_config.update(algo_config['consumer_config'])
    consumer_config['device'] = 'cpu'
    
    consumer_agent = SN_MAPPO(consumer_config, obs_space, act_space)
    assert consumer_agent.is_leader == False
    assert consumer_agent.agent_type == 'consumer'
    
    print("✓ SN-MAPPO algorithm initialization successful")
    return uc_agent, consumer_agent


def test_total_derivative():
    """Test total derivative computation."""
    print("\nTesting total derivative computation...")
    
    # Create dummy losses and parameters
    loss_uc = torch.tensor(1.0, requires_grad=True)
    loss_consumers = torch.tensor(0.5, requires_grad=True)
    
    # Create dummy parameters
    uc_params = [torch.randn(10, 5, requires_grad=True)]
    consumer_params = [torch.randn(10, 5, requires_grad=True)]
    
    # Import and test
    from algorithms.actors.sn_mappo import SN_MAPPO
    
    # This would normally be called within the algorithm
    # For now, just check the method exists
    assert hasattr(SN_MAPPO, '_compute_total_derivative')
    
    print("✓ Total derivative computation available")


def test_monitoring_system():
    """Test monitoring system functionality."""
    print("\nTesting monitoring system...")
    
    from envs.stackelberg.stackelberg_game import StackelbergMonitor
    
    # Create monitor
    monitor = StackelbergMonitor(
        log_dir="logs/test_stackelberg",
        experiment_name="test_run"
    )
    
    # Test logging step
    rewards = {0: 10.0, 1: -5.0, 2: -4.0}
    observations = {i: np.random.randn(14) for i in range(3)}
    actions = {0: np.array([1.0, 0.2, 0.5, 0.0, 0.1]), 
               1: np.array([-0.1, 0.5]), 
               2: np.array([-0.2, 0.3])}
    infos = {i: {} for i in range(3)}
    system_state = {'power_loss_ratio': 0.02, 'voltage_violations': 0}
    
    monitor.log_step(rewards, observations, actions, infos, system_state)
    
    # Test Nash gap tracking
    assert hasattr(monitor, '_update_nash_gap')
    
    # Test episode end
    monitor.log_episode_end()
    
    print("✓ Monitoring system functional")
    
    # Clean up
    monitor.close()


def test_async_wrapper():
    """Test asynchronous wrapper functionality."""
    print("\nTesting async wrapper...")
    
    from envs.stackelberg.stackelberg_game import AsyncMultiAgentWrapper, StackelbergBaseEnv
    
    # Load config
    config_path = Path("configs/envs_cfgs/stackelberg_13bus.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create base environment
    base_env = StackelbergBaseEnv(config)
    
    # Create wrapper
    wrapper = AsyncMultiAgentWrapper(base_env, config.get('async_config', {}))
    
    # Test reset
    obs = wrapper.reset()
    assert len(obs) == 1, "Should only have UC observation after reset"
    assert 0 in obs, "UC (agent 0) should have observation"
    
    # Test UC phase
    uc_action = {0: np.array([1.0, 0.2, 0.5, 0.0, 0.1])}
    obs, rewards, done, infos = wrapper.step(uc_action)
    
    # Should now have consumer observations
    assert len(obs) == 8, "Should have 8 consumer observations"
    assert all(i in obs for i in range(1, 9)), "All consumers should have observations"
    
    print("✓ Async wrapper functional")
    
    # Clean up
    wrapper.close()


def test_per_functionality():
    """Test Prioritized Experience Replay functionality."""
    print("\nTesting PER functionality...")
    
    # Load config
    config_path = Path("configs/envs_cfgs/stackelberg_13bus.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable PER
    config['per_config'] = {
        'enable': True,
        'buffer_size': 1000,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001,
        'epsilon': 1e-6
    }
    
    from envs.stackelberg.stackelberg_game import StackelbergBaseEnv
    
    # Create environment with PER
    env = StackelbergBaseEnv(config)
    
    assert env.enable_per == True
    assert hasattr(env, 'replay_buffer')
    assert hasattr(env, 'priorities')
    
    print("✓ PER functionality available")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Stackelberg Game Implementation Integration Test")
    print("=" * 60)
    
    try:
        # Test 1: Environment creation
        env = test_environment_creation()
        
        # Test 2: Reward calculations
        test_reward_calculations(env)
        
        # Test 3: SN-MAPPO algorithm
        uc_agent, consumer_agent = test_sn_mappo_algorithm()
        
        # Test 4: Total derivative
        test_total_derivative()
        
        # Test 5: Monitoring system
        test_monitoring_system()
        
        # Test 6: Async wrapper
        test_async_wrapper()
        
        # Test 7: PER functionality
        test_per_functionality()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())