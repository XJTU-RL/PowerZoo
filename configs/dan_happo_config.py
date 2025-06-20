#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : dan_happo_config.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: Configuration file for DAN-HAPPO algorithm
"""

import argparse
from configs.config import get_config as get_base_config

def get_dan_happo_config():
    """Get DAN-HAPPO specific configuration
    Returns:
        args: Configuration arguments
    """
    # Get base configuration
    args = get_base_config()
    
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
    """Main configuration function
    Returns:
        args: Configuration arguments
    """
    return get_dan_happo_config()

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