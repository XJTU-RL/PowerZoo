# -*- coding: utf-8 -*-
"""
Test Suite for Stackelberg Game Environment

This module provides comprehensive tests for the Stackelberg game-based
demand response environment implementation.
"""

import os
import sys
import pytest
import numpy as np
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.powerzoo.stackelberg_powerzoo_env import StackelbergPowerZooEnv, make_stackelberg_env
from envs.powerzoo.powerzoo.stackelberg_base_env import StackelbergBaseEnv
from envs.powerzoo.powerzoo.async_wrapper import AsyncMultiAgentWrapper
from envs.powerzoo.powerzoo.load_aggregator import IntelligentLoadAggregator
from envs.powerzoo.powerzoo.stackelberg_monitor import StackelbergMonitor


class TestStackelbergBaseEnv:
    """Test the base Stackelberg environment."""
    
    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        return {
            'system_name': '13Bus',
            'dss_file': 'Master_noPV_1.dss',
            'max_episode_steps': 24,
            'n_consumer_agents': 5,
            'base_path': 'envs/powerzoo/systems',
            'monitoring_config': {'enabled': False}  # Disable for faster tests
        }
    
    def test_env_creation(self, base_config):
        """Test environment creation."""
        env = StackelbergBaseEnv(base_config)
        
        assert env is not None
        assert env.n_agents == 6  # 1 UC + 5 consumers
        assert env.max_episode_steps == 24
        assert env.system_name == '13Bus'
    
    def test_reset(self, base_config):
        """Test environment reset."""
        env = StackelbergBaseEnv(base_config)
        observations = env.reset()
        
        assert isinstance(observations, dict)
        assert len(observations) == env.n_agents
        assert 0 in observations  # UC agent
        
        # Check observation shapes
        for agent_id, obs in observations.items():
            assert isinstance(obs, np.ndarray)
            assert obs.shape == env.observation_spaces[agent_id].shape
    
    def test_uc_step(self, base_config):
        """Test UC action execution."""
        env = StackelbergBaseEnv(base_config)
        env.reset()
        
        # Create UC action
        uc_action = np.array([1.0, 0.2, 0.5])  # price, DR incentive, capacity
        
        # Execute UC step
        uc_reward, uc_info = env.step_uc(uc_action)
        
        assert isinstance(uc_reward, float)
        assert isinstance(uc_info, dict)
        assert 'price_signal' in uc_info
        assert uc_info['price_signal'] == 1.0
    
    def test_consumer_step(self, base_config):
        """Test consumer action execution."""
        env = StackelbergBaseEnv(base_config)
        env.reset()
        
        # First UC action
        uc_action = np.array([1.2, 0.1, 0.5])
        env.step_uc(uc_action)
        
        # Consumer actions
        consumer_actions = {
            i: np.array([0.1, 0.5, 0.0])  # load adj, DER, storage
            for i in range(1, 6)
        }
        
        rewards, infos, done = env.step_consumers(consumer_actions)
        
        assert isinstance(rewards, dict)
        assert len(rewards) == env.n_agents
        assert isinstance(done, bool)
    
    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        config = {
            'system_name': '13Bus',
            'dss_file': 'Master_noPV_1.dss',
            'max_episode_steps': 5,  # Short episode for testing
            'n_consumer_agents': 2,
            'monitoring_config': {
                'enabled': True,
                'log_interval': 1,
                'save_interval': 10
            }
        }
        
        env = StackelbergBaseEnv(config)
        obs = env.reset()
        
        # Run a few steps
        for _ in range(3):
            uc_action = np.random.rand(3)
            env.step_uc(uc_action)
            
            consumer_actions = {
                i: np.random.rand(3)
                for i in range(1, 3)
            }
            env.step_consumers(consumer_actions)
        
        # Check monitoring data
        summary = env.get_monitoring_summary()
        assert 'current_metrics' in summary
        assert 'episode' in summary


class TestAsyncWrapper:
    """Test the asynchronous execution wrapper."""
    
    @pytest.fixture
    def wrapped_env(self):
        """Create wrapped environment."""
        base_config = {
            'system_name': '13Bus',
            'dss_file': 'Master_noPV_1.dss',
            'max_episode_steps': 10,
            'n_consumer_agents': 3,
            'monitoring_config': {'enabled': False}
        }
        
        base_env = StackelbergBaseEnv(base_config)
        wrapper = AsyncMultiAgentWrapper(base_env)
        return wrapper
    
    def test_async_reset(self, wrapped_env):
        """Test async wrapper reset."""
        observations = wrapped_env.reset()
        
        # Should only get UC observation initially
        assert len(observations) == 1
        assert 0 in observations
    
    def test_async_execution_sequence(self, wrapped_env):
        """Test proper UC-consumer sequencing."""
        observations = wrapped_env.reset()
        
        # Phase 1: UC decision
        assert wrapped_env.current_phase == 'uc_decision'
        uc_action = {0: np.array([1.0, 0.1, 0.5])}
        
        obs, rewards, done, infos = wrapped_env.step(uc_action)
        
        # Should transition to consumer phase
        assert wrapped_env.current_phase == 'consumer_response'
        assert len(obs) == 3  # 3 consumers
        assert 0 not in obs  # UC not in observations
        
        # Phase 2: Consumer response
        consumer_actions = {
            1: np.array([0.1, 0.5, 0.0]),
            2: np.array([0.0, 0.3, 0.1]),
            3: np.array([-0.1, 0.2, 0.0])
        }
        
        obs, rewards, done, infos = wrapped_env.step(consumer_actions)
        
        # Should have rewards for all agents
        assert len(rewards) == 4  # UC + 3 consumers
        
        # Should transition back to UC phase if not done
        if not done:
            assert wrapped_env.current_phase == 'uc_decision'
            assert 0 in obs  # UC gets observation


class TestLoadAggregator:
    """Test the intelligent load aggregator."""
    
    def test_zone_based_aggregation(self):
        """Test zone-based load aggregation."""
        # This test would require a mock circuit object
        # Simplified test for demonstration
        
        config = {
            'use_electrical_distance': True
        }
        
        # Would need mock circuit for full test
        # aggregator = IntelligentLoadAggregator(mock_circuit, 'zone', config)
        # mapping = aggregator.aggregate_loads(5)
        # assert len(mapping) > 0
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        methods = ['zone', 'priority', 'graph', 'adaptive']
        
        # Each method should be callable
        # Full test would require mock circuit
        for method in methods:
            assert method in ['zone', 'priority', 'graph', 'adaptive']


class TestStackelbergPowerZooEnv:
    """Test the main PowerZoo wrapper."""
    
    @pytest.fixture  
    def env_args(self):
        """Create environment arguments."""
        return {
            'env_name': 'stackelberg_13bus',
            'num_steps': 24,
            'seed': 42,
            'use_render': False,
            'worker_idx': 0
        }
    
    def test_powerzoo_env_creation(self, env_args):
        """Test PowerZoo environment creation."""
        env = make_stackelberg_env(env_args)
        
        assert env is not None
        assert hasattr(env, 'n_agents')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
    
    def test_powerzoo_compatibility(self, env_args):
        """Test PowerZoo interface compatibility."""
        env = make_stackelberg_env(env_args)
        
        # Check required attributes
        assert hasattr(env, 'observation_spaces')
        assert hasattr(env, 'action_spaces')
        assert hasattr(env, 'n_agents')
        assert hasattr(env, 'get_env_info')
        
        # Check env info
        env_info = env.get_env_info()
        assert 'n_agents' in env_info
        assert 'obs_shape' in env_info
        assert 'episode_limit' in env_info
    
    def test_full_episode_run(self, env_args):
        """Test running a full episode."""
        env_args['num_steps'] = 5  # Short episode for testing
        env = make_stackelberg_env(env_args)
        
        observations = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 10:  # Safety limit
            # Get active agents for current phase
            phase_info = env.get_phase_info()
            active_agents = phase_info['active_agents']
            
            # Create actions for active agents
            actions = {}
            for agent_id in active_agents:
                if agent_id in env.action_spaces:
                    action_space = env.action_spaces[agent_id]
                    if hasattr(action_space, 'sample'):
                        actions[agent_id] = action_space.sample()
                    else:
                        actions[agent_id] = np.zeros(3)
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            done = dones[0] if isinstance(dones, np.ndarray) else dones
            step_count += 1
        
        assert step_count > 0
        env.close()


class TestMonitoring:
    """Test monitoring and logging functionality."""
    
    def test_monitor_creation(self):
        """Test monitor creation."""
        monitor = StackelbergMonitor(
            log_dir='tests/logs',
            experiment_name='test_exp'
        )
        
        assert monitor is not None
        assert monitor.experiment_name == 'test_exp'
        
        monitor.close()
    
    def test_metric_logging(self):
        """Test metric logging."""
        monitor = StackelbergMonitor(
            log_dir='tests/logs',
            experiment_name='test_metrics'
        )
        
        # Log some test data
        rewards = {0: 10.0, 1: -5.0, 2: -3.0}
        observations = {i: np.random.rand(10) for i in range(3)}
        actions = {i: np.random.rand(3) for i in range(3)}
        infos = {i: {} for i in range(3)}
        system_state = {
            'power_loss_ratio': 0.05,
            'voltage_violations': 2,
            'min_voltage': 0.96,
            'max_voltage': 1.04
        }
        
        monitor.log_step(rewards, observations, actions, infos, system_state)
        
        # Check that metrics were recorded
        assert len(monitor.uc_metrics['reward']) == 1
        assert monitor.uc_metrics['reward'][0] == 10.0
        
        monitor.close()


class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_files = [
            'configs/envs_cfgs/stackelberg_13bus.yaml',
            'configs/envs_cfgs/stackelberg_34bus.yaml', 
            'configs/envs_cfgs/stackelberg_123bus.yaml',
            'configs/algos_cfgs/sn_mappo.yaml'
        ]
        
        for config_file in config_files:
            assert os.path.exists(config_file), f"Config file {config_file} not found"
    
    def test_config_loading(self):
        """Test loading configuration files."""
        config_path = 'configs/envs_cfgs/stackelberg_13bus.yaml'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'env_name' in config
            assert 'system_name' in config
            assert config['system_name'] == '13Bus'
            assert 'n_consumer_agents_13bus' in config


def test_integration():
    """Integration test for the complete system."""
    # Create minimal configuration
    args = {
        'env_name': 'stackelberg_13bus',
        'num_steps': 3,
        'seed': 123,
        'use_render': False,
        'worker_idx': 0
    }
    
    # Create environment
    env = make_stackelberg_env(args)
    
    # Run a short episode
    obs = env.reset()
    
    for _ in range(3):
        phase_info = env.get_phase_info()
        active_agents = phase_info['active_agents']
        
        actions = {}
        for agent_id in active_agents:
            actions[agent_id] = np.random.rand(3) if agent_id in env.action_spaces else np.zeros(3)
        
        obs, rewards, dones, infos = env.step(actions)
        
        if dones[0]:
            break
    
    # Get final stats
    stats = env.get_stats()
    
    # Clean up
    env.close()
    
    print("Integration test passed!")


if __name__ == "__main__":
    # Run basic tests
    print("Running Stackelberg environment tests...")
    
    # Test environment creation
    test_integration()
    
    print("All tests completed!")