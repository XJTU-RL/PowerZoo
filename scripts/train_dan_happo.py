#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : train_dan_happo.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: Training script for DAN-HAPPO algorithm on DSR environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import wandb
from configs.dan_happo_config import get_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.dsr.dsr_env_optimized import DSREnvOptimized
from utils.dan_buffer import DANSharedReplayBuffer

def make_train_env(all_args):
    """Create training environment
    Args:
        all_args: Configuration arguments
    Returns:
        env: Training environment
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "DSR":
                if all_args.scenario_name == "dsr_optimized":
                    env = DSREnvOptimized(all_args)
                else:
                    # Fallback to original DSR environment
                    from envs.dsr.dsr_env import DSREnv
                    env = DSREnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    """Create evaluation environment
    Args:
        all_args: Configuration arguments
    Returns:
        env: Evaluation environment
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "DSR":
                if all_args.scenario_name == "dsr_optimized":
                    env = DSREnvOptimized(all_args)
                else:
                    from envs.dsr.dsr_env import DSREnv
                    env = DSREnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    """Parse additional arguments
    Args:
        args: Existing arguments
        parser: Argument parser
    Returns:
        args: Updated arguments
    """
    parser.add_argument('--scenario_name', type=str, default='dsr_optimized', 
                       help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=10, 
                       help="Number of agents")
    parser.add_argument('--use_wandb', action='store_true', default=False, 
                       help="Use wandb for logging")
    parser.add_argument('--user_name', type=str, default='dan_happo_user', 
                       help="Wandb user name")
    parser.add_argument('--wandb_name', type=str, default='dan_happo_dsr', 
                       help="Wandb experiment name")
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    """Main training function
    Args:
        args: Command line arguments
    """
    # Get configuration
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # Set device
    if all_args.cuda and torch.cuda.is_available():
        print("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # Set random seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    # Create environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    
    # Create save directory
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # Initialize wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                        project=all_args.wandb_name,
                        entity=all_args.user_name,
                        notes=socket.gethostname(),
                        name=str(all_args.algorithm_name) + "_" +
                             str(all_args.experiment_name) +
                             "_seed" + str(all_args.seed),
                        group=all_args.scenario_name,
                        dir=str(run_dir),
                        job_type="training",
                        reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    
    # Set run directory
    setattr(all_args, 'run_dir', run_dir)
    
    # Get observation and action spaces
    obs_space = envs.observation_space[0]
    share_obs_space = envs.share_observation_space[0] if hasattr(envs, 'share_observation_space') else obs_space
    act_space = envs.action_space[0]
    
    print(f"Observation space: {obs_space}")
    print(f"Share observation space: {share_obs_space}")
    print(f"Action space: {act_space}")
    print(f"Number of agents: {num_agents}")
    
    # Create DAN-specific buffer
    from common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
    from common.buffers.on_policy_critic_buffer import OnPolicyCriticBuffer
    
    # Initialize actor buffers for each agent
    actor_buffers = []
    for agent_id in range(num_agents):
        ac_buffer = DANSharedReplayBuffer(all_args, obs_space, act_space)
        actor_buffers.append(ac_buffer)
    
    # Initialize critic buffer
    critic_buffer = OnPolicyCriticBuffer(all_args, share_obs_space)
    
    # Get algorithm from registered algorithms
    from algorithms import ALGO_REGISTRY
    
    # Initialize algorithm
    if all_args.algorithm_name in ALGO_REGISTRY:
        AlgoClass = ALGO_REGISTRY[all_args.algorithm_name]
        algorithm = AlgoClass(all_args, num_agents, obs_space, share_obs_space, act_space, device)
    else:
        raise NotImplementedError(f"Algorithm {all_args.algorithm_name} not found in registry")
    
    # Get runner from registered runners  
    from runners import RUNNER_REGISTRY
    
    if all_args.algorithm_name in RUNNER_REGISTRY:
        RunnerClass = RUNNER_REGISTRY[all_args.algorithm_name]
        runner = RunnerClass(all_args, envs, eval_envs, num_agents, actor_buffers, critic_buffer, algorithm)
    else:
        raise NotImplementedError(f"Runner for {all_args.algorithm_name} not found in registry")
    
    # Start training
    print("Starting DAN-HAPPO training...")
    print(f"Algorithm: {all_args.algorithm_name}")
    print(f"Environment: {all_args.env_name} - {all_args.scenario_name}")
    print(f"Use DAN: {all_args.use_dan}")
    print(f"DAN Hidden Dim: {all_args.dan_hidden_dim}")
    print(f"DAN Attention Heads: {all_args.dan_num_heads}")
    print(f"Enhanced Action Mask: {all_args.use_enhanced_action_mask}")
    print(f"Optimized Reward Function: {all_args.use_progressive_overload_penalty}")
    print(f"Training steps: {all_args.num_env_steps}")
    print(f"Episode length: {all_args.episode_length}")
    print(f"Number of rollout threads: {all_args.n_rollout_threads}")
    print(f"Device: {device}")
    print("-" * 50)
    
    runner.run()
    
    # Close environments
    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    
    # Close wandb
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()
    
    print("Training completed!")

if __name__ == "__main__":
    import socket
    main(sys.argv[1:])