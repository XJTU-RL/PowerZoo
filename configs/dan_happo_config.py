#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : dan_happo_config.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: Configuration for DAN-HAPPO algorithm
"""

import argparse

def get_base_config():
    """Get base configuration
    Returns:
        args: Base configuration arguments
    """
    parser = argparse.ArgumentParser(description='DAN-HAPPO Configuration')
    
    # Basic training parameters
    parser.add_argument("--algorithm_name", type=str, default='dan_happo', help="algorithm name")
    parser.add_argument("--experiment_name", type=str, default="test", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1, help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1, help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=int(10e6), help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='dan_happo_user', help="[for wandb usage], to specify user's name for simply collecting training data.")
    
    # Network parameters
    parser.add_argument("--hidden_size", type=int, default=64, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false', default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")
    
    # Recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true', default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false', default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10, help="Time length of chunks used to train a recurrent_policy")
    
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="critic learning rate (default: 5e-4)")
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    
    # PPO parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks", action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    
    # Run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true', default=False, help='use a linear schedule on the learning rate')
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")
    
    # Eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")
    
    # Logging parameters
    parser.add_argument("--use_wandb", action='store_true', default=False, help="by default, do not use wandb. If set, use wandb for logging.")
    parser.add_argument("--user_name", type=str, default='zxd_xjtu', help="wandb user name")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb project name")
    
    # Render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")
    
    # Environment parameters
    parser.add_argument("--env_name", type=str, default='DSR', help="specify the name of environment")
    parser.add_argument("--episode_length", type=int, default=200, help="Max length for any episode")
    
    # HAPPO special parameters
    parser.add_argument("--actor_num_mini_batch", type=int, default=1, help='number of batches for actor update')
    parser.add_argument("--use_centralized_V", action='store_false', default=True, help="Whether to use centralized V function")
    parser.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--use_common_layer", action='store_false', default=True, help="Whether to use common layer")
    parser.add_argument("--use_action_attention", action='store_true', default=False, help="Whether to use action attention")
    parser.add_argument("--action_aggregation", type=str, default='prod', help="action aggregation method")
    
    return parser

def get_dan_happo_config():
    """Get DAN-HAPPO specific configuration
    Returns:
        args: Configuration arguments
    """
    # Get base configuration parser
    parser = get_base_config()
    
    # Add DAN-HAPPO specific arguments
    parser.add_argument('--scenario_name', type=str, default='DSR', help="Scenario name")
    parser.add_argument('--case_path', type=str, default='./envs/cases/13Bus/IEEE13Nodeckt.dss', help="Path to DSS case file")
    parser.add_argument('--num_agents', type=int, default=4, help="Number of agents")
    parser.add_argument('--use_dan', action='store_true', default=True, help="Whether to use DAN architecture")
    parser.add_argument('--use_neighbor_obs', action='store_true', default=True, help="Whether to use neighbor observations")
    parser.add_argument('--use_enhanced_action_mask', action='store_true', default=True, help="Whether to use enhanced action masking")
    parser.add_argument('--progressive_overload_penalty', action='store_true', default=True, help="Whether to use progressive overload penalty")
    
    # Parse arguments and get base args object
    args = parser.parse_args([])
    
    # DAN-HAPPO specific parameters
    
    # === DAN Architecture Parameters ===
    args.use_dan = True  # Enable DAN architecture
    args.dan_hidden_dim = 128  # Hidden dimension for DAN
    args.dan_num_heads = 4  # Number of attention heads
    args.dan_use_attention = True  # Use attention mechanism
    args.dan_dropout = 0.1  # Dropout rate for DAN
    args.dan_layer_norm = True  # Use layer normalization
    
    # === DAN Observation Processing ===
    args.env_obs_ratio = 0.6  # Ratio of environmental observations
    args.use_neighbor_obs = True  # Use neighboring agent observations
    args.max_neighbors = 5  # Maximum number of neighbors to consider
    
    # === DAN Training Parameters ===
    args.dan_lr = args.lr  # DAN learning rate (same as main lr)
    args.dan_weight_decay = 1e-5  # Weight decay for DAN
    args.dan_grad_norm_max_norm = 10.0  # Gradient clipping for DAN
    
    # === Enhanced Environment Parameters ===
    args.use_optimized_env = True  # Use optimized DSR environment
    args.use_enhanced_action_mask = True  # Use enhanced action masking
    
    # === Improved Reward Function Parameters ===
    # Increase overload penalty weights
    args.reward_overload = 5.0  # Increased from 1.0
    args.reward_severe_overload = 20.0  # New severe overload penalty
    args.severe_overload_threshold = 1.5  # Threshold for severe overload (150%)
    
    # Progressive overload penalties
    args.use_progressive_overload_penalty = True
    args.overload_penalty_levels = [1.1, 1.3, 1.5, 2.0]  # Overload ratio thresholds
    args.overload_penalty_weights = [2.0, 5.0, 10.0, 25.0]  # Corresponding penalty weights
    
    # Termination conditions
    args.terminate_on_severe_overload = True
    args.max_consecutive_overloads = 5  # Max consecutive steps with overloads
    
    # === HAPPO Parameters (inherited but can be modified) ===
    args.algorithm_name = "dan_happo"
    args.use_centralized_V = True
    args.use_obs_instead_of_state = False
    args.use_popart = True
    args.use_valuenorm = False
    args.use_feature_normalization = True
    args.use_orthogonal = True
    
    # === Training Parameters ===
    args.num_mini_batch = 1
    args.ppo_epoch = 15
    args.use_clipped_value_loss = True
    args.clip_param = 0.2
    args.entropy_coef = 0.01
    args.value_loss_coef = 1
    args.use_max_grad_norm = True
    args.max_grad_norm = 10.0
    
    # === Network Architecture ===
    args.hidden_size = 128  # Should match dan_hidden_dim for consistency
    args.layer_N = 2
    args.use_ReLU = True
    args.use_common_layer = True
    
    # === Logging and Evaluation ===
    args.use_wandb = False  # Disable wandb by default
    args.log_interval = 10
    args.eval_interval = 25
    args.save_interval = 100
    
    # === Experiment Parameters ===
    args.experiment_name = "dan_happo_dsr"
    args.seed = 1
    args.n_training_threads = 1
    args.n_rollout_threads = 8
    args.num_env_steps = 2e6
    args.episode_length = 200
    
    # === Environment Specific ===
    args.env_name = "DSR"
    args.scenario_name = "dsr_optimized"  # Use optimized environment
    
    return args

def get_config():
    """Get configuration for DAN-HAPPO
    Returns:
        parser: ArgumentParser object
    """
    # Get base configuration parser
    parser = get_base_config()
    
    # Add DAN-HAPPO specific arguments
    parser.add_argument('--scenario_name', type=str, default='DSR', help="Scenario name")
    parser.add_argument('--case_path', type=str, default='./envs/cases/13Bus/IEEE13Nodeckt.dss', help="Path to DSS case file")
    parser.add_argument('--num_agents', type=int, default=4, help="Number of agents")
    parser.add_argument('--use_dan', action='store_true', default=True, help="Whether to use DAN architecture")
    parser.add_argument('--use_neighbor_obs', action='store_true', default=True, help="Whether to use neighbor observations")
    parser.add_argument('--use_enhanced_action_mask', action='store_true', default=True, help="Whether to use enhanced action masking")
    parser.add_argument('--progressive_overload_penalty', action='store_true', default=True, help="Whether to use progressive overload penalty")
    
    return parser

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("DAN-HAPPO Configuration:")
    print(f"  Algorithm: {config.algorithm_name}")
    print(f"  Use DAN: {config.use_dan}")
    print(f"  DAN Hidden Dim: {config.dan_hidden_dim}")
    print(f"  DAN Attention Heads: {config.dan_num_heads}")
    print(f"  Enhanced Action Mask: {config.use_enhanced_action_mask}")
    print(f"  Overload Penalty: {config.reward_overload}")
    print(f"  Severe Overload Penalty: {config.reward_severe_overload}")
    print(f"  Progressive Penalties: {config.use_progressive_overload_penalty}")