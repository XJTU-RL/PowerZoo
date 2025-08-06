#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用增强训练脚本
展示如何在不同环境中使用增强的TensorBoard回调类
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, DQN, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# 导入增强回调类
from examples.single_agent.powerzoo_llm.enhanced_tensorboard_callback import (
    EnhancedTensorBoardCallback, 
    create_enhanced_callback
)

# 导入PowerZoo环境（如果可用）
try:
    from envs.power_envs.powerzoo_llm.env import SingleAgentPowerZooEnv
    from envs.power_envs.powerzoo_llm.single_agent.single_agent_training_config import SingleAgentConfig
    POWERZOO_AVAILABLE = True
except ImportError:
    POWERZOO_AVAILABLE = False
    print("Warning: PowerZoo environment not available")


def load_config_from_yaml(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_powerzoo_environment(config_path: str):
    """
    创建PowerZoo环境
    
    Args:
        config_path: 配置文件路径
    """
    if not POWERZOO_AVAILABLE:
        raise ImportError("PowerZoo environment is not available")
    
    # 加载配置
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        config_dict = load_config_from_yaml(config_path)
        print(f"从YAML文件加载配置: {config_path}")
        
        # 创建SingleAgentConfig，传入完整的配置字典
        # SingleAgentConfig会自动处理配置结构
        config = SingleAgentConfig()
        
        # 手动设置一些关键参数（如果配置文件中有的话）
        if 'train' in config_dict:
            train_config = config_dict['train']
            if 'episode_length' in train_config:
                config.episode_length = train_config['episode_length']
            if 'n_rollout_threads' in train_config:
                config.n_rollout_threads = train_config['n_rollout_threads']
        
        # 设置环境相关参数
        config.config_dict = config_dict  # 保存完整配置以备后用
    else:
        config = SingleAgentConfig(config_path)
    
    # 创建环境
    env = SingleAgentPowerZooEnv(config)
    env = Monitor(env)
    
    return env, config


def create_gym_environment(env_name: str = "CartPole-v1"):
    """
    创建Gym环境
    """
    env = gym.make(env_name)
    env = Monitor(env)
    return env


def get_algorithm_class(algo_name: str):
    """
    根据算法名称获取算法类
    """
    algorithms = {
        'ppo': PPO,
        'dqn': DQN,
        'sac': SAC,
        'a2c': A2C
    }
    
    if algo_name.lower() not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: {list(algorithms.keys())}")
    
    return algorithms[algo_name.lower()]


def train_with_enhanced_callback(args):
    """
    使用增强回调进行训练
    """
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.algorithm}_{args.env_type}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training with enhanced callback")
    print(f"Environment type: {args.env_type}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Log directory: {log_dir}")
    
    # 创建环境
    if args.env_type == "powerzoo":
        if not args.config:
            print("错误: PowerZoo环境需要指定配置文件")
            return
        env, config = create_powerzoo_environment(args.config)
        print(f"PowerZoo environment created with config: {args.config}")
    elif args.env_type == "gym":
        env = create_gym_environment(args.gym_env)
        config = None
        print(f"Gym environment created: {args.gym_env}")
    else:
        raise ValueError(f"Unsupported environment type: {args.env_type}")
    
    # 创建增强回调
    callback = create_enhanced_callback(
        log_dir=log_dir,
        env_type=args.env_type,
        enable_powerzoo_logging=(args.env_type == "powerzoo"),
        save_freq=args.save_freq,
        name_prefix=f"{args.algorithm}_{args.env_type}",
        verbose=args.verbose
    )
    
    # 获取算法类
    AlgorithmClass = get_algorithm_class(args.algorithm)
    
    # 创建模型
    if args.env_type == "powerzoo" and config:
        # 使用PowerZoo配置
        model_kwargs = {
            'learning_rate': config.algo_args.get('lr', 3e-4),
            'n_steps': config.algo_args.get('n_steps', 2048),
            'batch_size': config.algo_args.get('batch_size', 64),
            'n_epochs': config.algo_args.get('n_epochs', 10),
            'gamma': config.algo_args.get('gamma', 0.99),
            'verbose': args.verbose
        }
        
        # 过滤不适用于当前算法的参数
        if args.algorithm.lower() == 'dqn':
            model_kwargs = {
                'learning_rate': model_kwargs['learning_rate'],
                'gamma': model_kwargs['gamma'],
                'verbose': model_kwargs['verbose']
            }
        elif args.algorithm.lower() == 'sac':
            model_kwargs = {
                'learning_rate': model_kwargs['learning_rate'],
                'gamma': model_kwargs['gamma'],
                'verbose': model_kwargs['verbose']
            }
    else:
        # 使用默认配置
        model_kwargs = {'verbose': args.verbose}
    
    model = AlgorithmClass('MlpPolicy', env, **model_kwargs)
    
    print(f"Model created: {AlgorithmClass.__name__}")
    print(f"Training for {args.total_timesteps} timesteps")
    
    # 开始训练
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    print("Training completed successfully!")
    print(f"Logs and models saved to: {log_dir}")
    print(f"To view TensorBoard logs, run:")
    print(f"tensorboard --logdir {log_dir} --port 6006 --host 0.0.0.0")
    
    return model, log_dir


def main():
    parser = argparse.ArgumentParser(description="通用增强训练脚本")
    
    # 环境参数
    parser.add_argument('--env-type', type=str, choices=['powerzoo', 'gym'], 
                       default='gym', help='环境类型')
    parser.add_argument('--config', type=str, help='PowerZoo配置文件路径（支持相对路径和绝对路径）')
    parser.add_argument('--gym-env', type=str, default='CartPole-v1', 
                       help='Gym环境名称')
    
    # 算法参数
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn', 'sac', 'a2c'],
                       default='ppo', help='强化学习算法')
    parser.add_argument('--total-timesteps', type=int, default=10000, 
                       help='总训练步数')
    
    # 日志参数
    parser.add_argument('--log-dir', type=str, default='./logs/enhanced_training',
                       help='日志保存目录')
    parser.add_argument('--save-freq', type=int, default=1000, 
                       help='模型保存频率')
    parser.add_argument('--verbose', type=int, default=1, help='详细程度')
    
    args = parser.parse_args()
    
    # 处理配置文件路径
    if args.config and args.env_type == 'powerzoo':
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # 如果是相对路径，尝试在标准配置目录中查找
            standard_config_dir = project_root / 'configs' / 'single_agent_cfgs'
            potential_config = standard_config_dir / config_path
            if potential_config.exists():
                args.config = str(potential_config)
                print(f"使用标准配置目录中的配置文件: {args.config}")
            else:
                # 相对于项目根目录
                potential_config = project_root / config_path
                if potential_config.exists():
                    args.config = str(potential_config)
                    print(f"使用项目根目录相对路径的配置文件: {args.config}")
                else:
                    print(f"警告: 配置文件未找到: {config_path}")
        else:
            print(f"使用绝对路径配置文件: {args.config}")
    
    # 验证参数
    if args.env_type == 'powerzoo':
        if not args.config:
            print("Error: PowerZoo environment requires --config parameter")
            return
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return
        if not POWERZOO_AVAILABLE:
            print("Error: PowerZoo environment is not available")
            return
    
    try:
        model, log_dir = train_with_enhanced_callback(args)
        print("\n=== Training Summary ===")
        print(f"Environment: {args.env_type}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Total timesteps: {args.total_timesteps}")
        print(f"Log directory: {log_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()