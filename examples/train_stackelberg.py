#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for training Stackelberg-Nash MAPPO on PowerZoo

This script demonstrates how to train agents using the Stackelberg game
framework for demand response in power systems.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from runners import RUNNER_REGISTRY


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Stackelberg-Nash MAPPO on PowerZoo"
    )
    
    # Algorithm and environment
    parser.add_argument(
        "--algo", 
        type=str, 
        default="sn_mappo",
        help="Algorithm name"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default="stackelberg_13bus",
        choices=["stackelberg_13bus", "stackelberg_34bus", "stackelberg_123bus"],
        help="Environment name"
    )
    
    # Experiment settings
    parser.add_argument(
        "--exp_name",
        type=str,
        default="stackelberg_dr",
        help="Experiment name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed"
    )
    
    # Training settings
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10000000,
        help="Number of environment steps"
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    
    # Device settings
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use CUDA"
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device ID"
    )
    
    # Logging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="Log interval in episodes"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Save interval in episodes"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (overrides other arguments)"
    )
    
    return parser.parse_args()


def load_config(args):
    """Load configuration from files and command line."""
    config = {}
    
    # Load algorithm config
    algo_config_path = f"configs/algos_cfgs/{args.algo}.yaml"
    if os.path.exists(algo_config_path):
        with open(algo_config_path, 'r') as f:
            algo_config = yaml.safe_load(f)
            config.update(algo_config)
    
    # Load environment config
    env_config_path = f"configs/envs_cfgs/{args.env}.yaml"
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
            config.update(env_config)
    
    # Override with command line arguments
    cmd_args = vars(args)
    for key, value in cmd_args.items():
        if value is not None:
            config[key] = value
    
    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.json'):
                file_config = json.load(f)
            else:
                file_config = yaml.safe_load(f)
            config.update(file_config)
    
    return config


def setup_stackelberg_config(config):
    """Setup Stackelberg-specific configuration."""
    # Ensure we have the right environment setup
    if 'stackelberg' not in config.get('env', ''):
        config['env'] = f"stackelberg_{config.get('system_name', '13bus').lower()}"
    
    # Set algorithm-specific parameters
    config['algo'] = 'sn_mappo'
    
    # Agent configuration based on system
    system_name = config.get('system_name', '13Bus')
    if system_name == '13Bus':
        config['n_agents'] = 6  # 1 UC + 5 consumers
    elif system_name == '34Bus':
        config['n_agents'] = 11  # 1 UC + 10 consumers
    elif system_name == '123Bus':
        config['n_agents'] = 21  # 1 UC + 20 consumers
    
    # Set agent-specific configurations
    config['agent_configs'] = {
        0: config.get('uc_config', {
            'is_leader': True,
            'hierarchy_level': 0,
            'agent_type': 'uc'
        })
    }
    
    # Consumer configurations
    for i in range(1, config['n_agents']):
        config['agent_configs'][i] = config.get('consumer_config', {
            'is_leader': False,
            'hierarchy_level': 1,
            'agent_type': 'consumer'
        })
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Setup Stackelberg-specific configuration
    config = setup_stackelberg_config(config)
    
    # Print configuration
    print("=" * 50)
    print("Stackelberg-Nash MAPPO Training")
    print("=" * 50)
    print(f"Algorithm: {config['algo']}")
    print(f"Environment: {config['env']}")
    print(f"System: {config.get('system_name', '13Bus')}")
    print(f"Agents: {config['n_agents']} (1 UC + {config['n_agents']-1} consumers)")
    print(f"Episodes: {config.get('num_env_steps', 10000000) // config.get('max_episode_steps', 24)}")
    print(f"Seed: {config['seed']}")
    print("=" * 50)
    
    # Get runner class
    runner_class = RUNNER_REGISTRY.get(config['algo'])
    if runner_class is None:
        raise ValueError(f"Unknown algorithm: {config['algo']}")
    
    # Create and run trainer
    print("\nInitializing trainer...")
    runner = runner_class(config)
    
    print("Starting training...")
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        print("\nCleaning up...")
        runner.close()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()