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
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from configs.dan_happo_config import get_config
from envs.dsr.dsr_env import DSREnv  # Unified environment
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from algorithms.actors.dan_happo import DAN_HAPPO
from runner.shared.dsr_dan_runner import DSRDANRunner
from utils.dan_buffer import DANSharedReplayBuffer

def make_train_env(all_args):
    """Create training environment
    Args:
        all_args: Configuration arguments
    Returns:
        Environment function
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "DSR":
                # Create args dictionary for unified environment
                env_args = {
                    'env_name': 'dsr',
                    'use_dan': True,  # Enable DAN features
                    'case_path': getattr(all_args, 'case_path', None),
                    'max_episode_steps': all_args.episode_length,
                    'use_enhanced_action_mask': all_args.use_enhanced_action_mask,
                    'severe_overload_threshold': all_args.severe_overload_threshold,
                    'terminate_on_severe_overload': all_args.terminate_on_severe_overload,
                    'use_progressive_penalty': all_args.progressive_overload_penalty,
                    'overload_penalty_levels': all_args.overload_penalty_levels,
                    'overload_penalty_weights': all_args.overload_penalty_weights
                }
                env = DSREnv(env_args, rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return get_env_fn

def make_eval_env(all_args):
    """Create evaluation environment
    Args:
        all_args: Configuration arguments
    Returns:
        Environment function
    """
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "DSR":
                # Create args dictionary for unified environment
                env_args = {
                    'env_name': 'dsr',
                    'use_dan': True,  # Enable DAN features
                    'case_path': getattr(all_args, 'case_path', None),
                    'max_episode_steps': all_args.episode_length,
                    'use_enhanced_action_mask': all_args.use_enhanced_action_mask,
                    'severe_overload_threshold': all_args.severe_overload_threshold,
                    'terminate_on_severe_overload': all_args.terminate_on_severe_overload,
                    'use_progressive_penalty': all_args.progressive_overload_penalty,
                    'overload_penalty_levels': all_args.overload_penalty_levels,
                    'overload_penalty_weights': all_args.overload_penalty_weights
                }
                env = DSREnv(env_args, rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return get_env_fn



def main(args):
    """Main training function
    Args:
        args: Command line arguments
    """
    # Get configuration parser
    parser = get_config()
    
    # Parse command line arguments
    all_args = parser.parse_args(args)
    
    # Set device
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    # Set random seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    # Set process title
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name)
    )
    
    # Create directories
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    # Initialize wandb
    if all_args.use_wandb:
        try:
            import wandb
            run = wandb.init(
                config=all_args,
                project=all_args.env_name,
                entity=all_args.user_name,
                notes=socket.gethostname(),
                name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
                group=all_args.scenario_name,
                dir=str(run_dir),
                job_type="training",
                reinit=True
            )
        except ImportError:
            print("Warning: wandb not installed, disabling wandb logging")
            all_args.use_wandb = False
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
    
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    
    # Create environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    # Create environments
    if all_args.n_rollout_threads == 1:
        envs = ShareDummyVecEnv([envs(0)])
    else:
        envs = ShareSubprocVecEnv([envs(i) for i in range(all_args.n_rollout_threads)])
    
    if all_args.use_eval:
        if all_args.n_eval_rollout_threads == 1:
            eval_envs = ShareDummyVecEnv([eval_envs(0)])
        else:
            eval_envs = ShareSubprocVecEnv([eval_envs(i) for i in range(all_args.n_eval_rollout_threads)])
    
    config["envs"] = envs
    config["eval_envs"] = eval_envs
    
    # Initialize DAN-HAPPO policy
    policy = DAN_HAPPO(
        all_args,
        envs.observation_space[0],
        envs.share_observation_space[0],
        envs.action_space[0],
        device=device
    )
    
    config["policy"] = policy
    
    # Create DAN shared replay buffer
    buffer = DANSharedReplayBuffer(
        all_args,
        envs.observation_space[0],
        envs.share_observation_space[0],
        envs.action_space[0]
    )
    
    config["buffer"] = buffer
    
    # Create DAN runner
    runner = DSRDANRunner(config)
    
    # Start training
    runner.run()
    
    # Post-process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])