#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for using the Stackelberg game environment.

This script demonstrates:
1. Creating a Stackelberg environment
2. Running a simple episode with random actions
3. Using the monitoring system
4. Training with SN-MAPPO algorithm
"""

import numpy as np
import torch
import yaml
from pathlib import Path

# Import Stackelberg environment components
from envs.stackelberg.stackelberg_game import (
    StackelbergBaseEnv, 
    AsyncMultiAgentWrapper,
    StackelbergMonitor
)
from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env

# Import SN-MAPPO algorithm
from algorithms.actors.sn_mappo import SN_MAPPO


def run_random_episode():
    """Run a single episode with random actions."""
    print("=" * 60)
    print("Running Random Episode in Stackelberg Environment")
    print("=" * 60)
    
    # Create environment using factory
    env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=True)
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    print(f"UC observation shape: {obs[0].shape if 0 in obs else 'N/A'}")
    
    episode_rewards = {i: 0.0 for i in range(env.n_agents)}
    done = False
    step = 0
    
    while not done:
        # Get active agents for current phase
        active_agents = env.get_active_agents()
        
        # Generate random actions for active agents
        actions = {}
        for agent_id in active_agents:
            if agent_id == 0:  # UC agent
                # UC has 5-dimensional action
                actions[agent_id] = np.random.uniform(-1, 1, size=5)
                actions[agent_id][0] = np.random.uniform(0.5, 2.0)  # Price signal
                actions[agent_id][1] = np.random.uniform(0.0, 0.5)  # DR incentive
                actions[agent_id][2] = np.random.uniform(0.0, 1.0)  # Capacity
                actions[agent_id][4] = np.random.uniform(0.0, 1.0)  # DER curtailment
            else:  # Consumer agent
                # Consumers have 2-dimensional action
                actions[agent_id] = np.array([
                    np.random.uniform(-0.3, 0.1),  # Load adjustment
                    np.random.uniform(0.0, 1.0)    # DER output
                ])
        
        # Step environment
        obs, rewards, done, infos = env.step(actions)
        
        # Accumulate rewards
        for agent_id, reward in rewards.items():
            episode_rewards[agent_id] += reward
        
        # Print phase info
        phase_info = env.get_phase_info()
        print(f"Step {step}: Phase={phase_info['current_phase']}, "
              f"Active agents={phase_info['active_agents']}, "
              f"Rewards={list(rewards.values())}")
        
        step += 1
    
    # Print episode summary
    print("\nEpisode Summary:")
    print(f"Total steps: {step}")
    print(f"UC total reward: {episode_rewards[0]:.2f}")
    consumer_rewards = [episode_rewards[i] for i in range(1, env.n_agents)]
    print(f"Average consumer reward: {np.mean(consumer_rewards):.2f}")
    print(f"Social welfare: {sum(episode_rewards.values()):.2f}")
    
    # Close environment
    env.close()


def demonstrate_monitoring():
    """Demonstrate the monitoring system."""
    print("\n" + "=" * 60)
    print("Demonstrating Monitoring System")
    print("=" * 60)
    
    # Create environment with monitoring
    config = {
        'monitoring_config': {
            'enable': True,
            'log_dir': 'logs/stackelberg_demo',
            'save_interval': 10,
            'plot_interval': 5,
            'experiment_name': 'demo_run'
        }
    }
    
    env = make_stackelberg_env('stackelberg_13bus', config=config)
    
    # Run multiple episodes
    n_episodes = 5
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # Random actions
            active_agents = env.get_active_agents()
            actions = {}
            
            for agent_id in active_agents:
                if agent_id == 0:
                    actions[agent_id] = np.array([1.0, 0.2, 0.5, 0.0, 0.1])
                else:
                    actions[agent_id] = np.array([-0.1, 0.5])
            
            obs, rewards, done, infos = env.step(actions)
        
        print(f"Episode {episode + 1} completed")
    
    print("\nMonitoring data saved to logs/stackelberg_demo/")
    env.close()


def demonstrate_sn_mappo_training():
    """Demonstrate SN-MAPPO algorithm setup."""
    print("\n" + "=" * 60)
    print("Demonstrating SN-MAPPO Training Setup")
    print("=" * 60)
    
    # Load algorithm config
    algo_config_path = Path("configs/algos_cfgs/sn_mappo.yaml")
    with open(algo_config_path, 'r') as f:
        algo_config = yaml.safe_load(f)
    
    # Create observation and action spaces
    obs_space = type('MockSpace', (), {'shape': (14,)})()
    uc_act_space = type('MockSpace', (), {'shape': (5,), 'n': 5})()
    consumer_act_space = type('MockSpace', (), {'shape': (2,), 'n': 2})()
    
    # Create UC agent
    uc_config = algo_config.copy()
    uc_config.update(algo_config['uc_config'])
    uc_config['device'] = 'cpu'
    
    uc_agent = SN_MAPPO(uc_config, obs_space, uc_act_space)
    print(f"UC Agent created: is_leader={uc_agent.is_leader}, "
          f"agent_type={uc_agent.agent_type}")
    
    # Create consumer agents
    consumer_agents = []
    consumer_config = algo_config.copy()
    consumer_config.update(algo_config['consumer_config'])
    consumer_config['device'] = 'cpu'
    
    for i in range(8):  # 8 consumers for 13Bus
        agent = SN_MAPPO(consumer_config, obs_space, consumer_act_space)
        consumer_agents.append(agent)
    
    print(f"Created {len(consumer_agents)} consumer agents")
    
    # Demonstrate policy evaluation
    dummy_obs = torch.randn(1, 1, 14)  # [batch, agents, obs_dim]
    dummy_rnn_states = torch.zeros(1, 1, 128)
    dummy_masks = torch.ones(1, 1)
    
    # UC action
    uc_action, _, _ = uc_agent.get_actions(
        dummy_obs, dummy_rnn_states, dummy_masks
    )
    print(f"UC action shape: {uc_action.shape}")
    
    # Consumer action
    consumer_action, _, _ = consumer_agents[0].get_actions(
        dummy_obs, dummy_rnn_states, dummy_masks
    )
    print(f"Consumer action shape: {consumer_action.shape}")
    
    # Check Nash equilibrium convergence
    converged = uc_agent.check_equilibrium_convergence()
    print(f"Nash equilibrium converged: {converged}")


def demonstrate_env_factory():
    """Demonstrate environment factory usage."""
    print("\n" + "=" * 60)
    print("Demonstrating Environment Factory")
    print("=" * 60)
    
    # Create single environment
    env = make_stackelberg_env('stackelberg_13bus')
    print(f"Created environment with {env.n_agents} agents")
    
    # Create parallel environments
    from envs.stackelberg.stackelberg_game.env_factory import make_parallel_stackelberg_envs
    
    parallel_envs = make_parallel_stackelberg_envs(
        'stackelberg_13bus',
        n_envs=4,
        use_async_wrapper=True
    )
    print(f"Created {len(parallel_envs)} parallel environments")
    
    # Clean up
    env.close()
    for penv in parallel_envs:
        penv.close()


def main():
    """Run all demonstrations."""
    # 1. Run random episode
    run_random_episode()
    
    # 2. Demonstrate monitoring
    demonstrate_monitoring()
    
    # 3. Demonstrate SN-MAPPO setup
    demonstrate_sn_mappo_training()
    
    # 4. Demonstrate environment factory
    demonstrate_env_factory()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()