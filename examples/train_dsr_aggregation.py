#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练DSR环境的示例脚本 - 展示负荷聚合功能
Example script for training DSR environment with load aggregation
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSR environment with load aggregation")
    
    # 环境参数
    parser.add_argument("--env_name", type=str, default="dsr", help="Environment name")
    parser.add_argument("--algorithm_name", type=str, default="mappo", help="Algorithm name")
    parser.add_argument("--experiment_name", type=str, default="dsr_aggregation", help="Experiment name")
    
    # DSR特定参数
    parser.add_argument("--system_name", type=str, default="123Bus", 
                       choices=["13Bus", "34Bus", "123Bus", "8500-Node"],
                       help="Power system to use")
    parser.add_argument("--use_load_aggregation", action="store_true", 
                       help="Use load aggregation for agents")
    parser.add_argument("--n_load_agents", type=int, default=None,
                       help="Number of load agents (None for automatic)")
    parser.add_argument("--load_aggregation_method", type=str, default="zone",
                       choices=["zone", "priority", "random"],
                       help="Load aggregation method")
    
    # 训练参数
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cuda", action="store_false", help="Use GPU")
    parser.add_argument("--cuda_deterministic", action="store_false", help="Use deterministic CUDA")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of training threads")
    parser.add_argument("--n_rollout_threads", type=int, default=3, help="Number of rollout threads")
    parser.add_argument("--num_mini_batch", type=int, default=1, help="Number of mini batches")
    parser.add_argument("--episode_length", type=int, default=15, help="Episode length")
    parser.add_argument("--num_env_steps", type=int, default=10000000, help="Number of environment steps")
    
    # 算法参数
    parser.add_argument("--share_policy", action="store_true", help="Share policy among agents")
    parser.add_argument("--use_centralized_V", action="store_false", help="Use centralized value function")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--layer_N", type=int, default=2, help="Number of layers")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--ppo_epoch", type=int, default=5, help="PPO epochs")
    
    # 评估参数
    parser.add_argument("--use_eval", action="store_true", help="Use evaluation")
    parser.add_argument("--eval_interval", type=int, default=25, help="Evaluation interval")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1, help="Number of eval threads")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--save_interval", type=int, default=25, help="Save interval")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 根据系统规模自动设置聚合参数
    if args.system_name == "13Bus":
        # 小系统不需要聚合
        args.use_load_aggregation = False
        print("13Bus系统: 自动禁用负荷聚合（系统较小）")
    elif args.system_name == "8500-Node" and not args.use_load_aggregation:
        # 大系统强制使用聚合
        args.use_load_aggregation = True
        if args.n_load_agents is None:
            args.n_load_agents = 50
        print(f"8500-Node系统: 自动启用负荷聚合，使用{args.n_load_agents}个负荷智能体")
    
    # 导入训练脚本
    from train import main as train_main
    
    # 设置环境参数
    env_args = {
        'system_name': args.system_name,
        'use_load_aggregation': args.use_load_aggregation,
        'n_load_agents': args.n_load_agents,
        'load_aggregation_method': args.load_aggregation_method,
        'max_episode_steps': args.episode_length,
    }
    
    # 更新参数
    for key, value in env_args.items():
        setattr(args, key, value)
    
    # 显示配置信息
    print("=" * 80)
    print("DSR环境训练配置")
    print("=" * 80)
    print(f"系统: {args.system_name}")
    print(f"使用负荷聚合: {args.use_load_aggregation}")
    if args.use_load_aggregation:
        print(f"负荷智能体数量: {args.n_load_agents if args.n_load_agents else '自动'}")
        print(f"聚合方法: {args.load_aggregation_method}")
    print(f"算法: {args.algorithm_name}")
    print(f"并行环境数: {args.n_rollout_threads}")
    print(f"训练步数: {args.num_env_steps}")
    print("=" * 80)
    
    # 运行训练
    train_main(args)


if __name__ == "__main__":
    main()