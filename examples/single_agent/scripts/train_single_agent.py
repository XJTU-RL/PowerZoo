#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单智能体强化学习训练主程序

本程序提供了完整的单智能体强化学习训练流程，包括：
- 命令行参数解析
- 配置管理和验证
- 环境创建和包装
- 模型初始化和训练
- 评估和监控
- 模型保存和加载

支持的算法: PPO, DQN, SAC, A2C
支持的环境: PowerZoo单智能体环境

使用示例:
    # 基础训练
    python train_single_agent.py --algo ppo --env powerzoo_single --exp_name test_ppo
    
    # 自定义参数训练
    python train_single_agent.py --algo dqn --total_timesteps 50000 --seed 42
    
    # 从检查点继续训练
    python train_single_agent.py --algo ppo --model_path ./models/checkpoint.zip
"""

import argparse
import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 导入自定义模块
from envs.power_envs.powerzoo_llm.single_agent.single_agent_training_config import SingleAgentTrainingConfig, get_config
from utils.single_agent_tools import (
    create_single_agent_env,
    create_model,
    create_callbacks,
    train_model,
    evaluate_model,
    save_model_and_config,
    load_model_and_config,
    setup_logging,
    print_training_info
)

# 设置日志
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="单智能体强化学习训练程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础参数
    parser.add_argument(
        "--algo", "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "sac", "a2c", "ddpg", "td3", "her"],
        help="强化学习算法"
    )
    
    parser.add_argument(
        "--env", "--environment",
        type=str,
        default="powerzoo_single",
        help="环境名称"
    )
    
    parser.add_argument(
        "--exp_name", "--experiment_name",
        type=str,
        default=None,
        help="实验名称（默认自动生成）"
    )
    
    # 训练参数
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="总训练步数"
    )
    
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="并行环境数量"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto, cpu, cuda)"
    )
    
    # 评估参数
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="评估频率（训练步数）"
    )
    
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="每次评估的回合数"
    )
    
    # 保存参数
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10000,
        help="模型保存频率（训练步数）"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="日志保存目录"
    )
    
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./models",
        help="模型保存目录"
    )
    
    # 模型加载
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="预训练模型路径"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="配置文件路径"
    )
    
    # 环境参数
    parser.add_argument(
        "--circuit_name",
        type=str,
        default="13Bus",
        choices=["13Bus", "34Bus", "123Bus", "8500Node"],
        help="电力系统电路名称"
    )
    
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=24,
        help="每个回合的最大步数"
    )
    
    parser.add_argument(
        "--action_space_type",
        type=str,
        default="auto",
        choices=["auto", "discrete", "continuous"],
        help="动作空间类型 (auto: 根据算法自动选择, discrete: 离散, continuous: 连续)"
    )
    
    # 其他参数
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="详细程度 (0: 无输出, 1: 基本信息, 2: 详细信息)"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="禁用TensorBoard日志"
    )
    
    parser.add_argument(
        "--normalize_env",
        action="store_true",
        help="标准化环境观测和奖励"
    )
    
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅进行评估，不训练"
    )
    
    return parser.parse_args()


def create_experiment_name(args) -> str:
    """创建实验名称"""
    if args.exp_name:
        return args.exp_name
    
    # 自动生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.algo}_{args.env}_{args.circuit_name}_{timestamp}"
    
    if args.seed is not None:
        exp_name += f"_seed{args.seed}"
    
    return exp_name


def setup_experiment_directories(config: SingleAgentTrainingConfig):
    """设置实验目录
    
    将所有实验结果保存到 /home/zhengxiaodong/exps/PowerZoo/results 文件夹下
    """
    # 设置results根目录
    results_root = "/home/zhengxiaodong/exps/PowerZoo/results"
    
    # 创建实验特定的目录
    exp_log_dir = os.path.join(results_root, "logs", config.experiment_name)
    exp_model_dir = os.path.join(results_root, "models", config.experiment_name)
    
    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(exp_model_dir, exist_ok=True)
    
    # 更新配置中的路径
    config.log_dir = exp_log_dir
    config.model_save_dir = exp_model_dir
    
    logger.info(f"实验结果将保存到: {results_root}")
    logger.info(f"日志目录: {exp_log_dir}")
    logger.info(f"模型目录: {exp_model_dir}")
    
    return exp_log_dir, exp_model_dir


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建实验名称
        exp_name = create_experiment_name(args)
        
        # 创建训练配置
        config = SingleAgentTrainingConfig(
            algorithm=args.algo,
            environment=args.env,
            experiment_name=exp_name,
            total_timesteps=args.total_timesteps,
            device=args.device,
            seed=args.seed,
            log_dir=args.log_dir,
            model_save_dir=args.model_save_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_freq=args.save_freq,
            verbose=args.verbose,
            tensorboard_log=not args.no_tensorboard
        )
        
        # 确定动作空间类型
        if args.action_space_type == "auto":
            # 根据算法自动选择动作空间类型
            continuous_algos = ["ddpg", "td3", "her"]
            action_space_type = "continuous" if args.algo.lower() in continuous_algos else "discrete"
            logger.info(f"自动选择动作空间类型: {action_space_type} (算法: {args.algo})")
        else:
            action_space_type = args.action_space_type
            logger.info(f"使用指定的动作空间类型: {action_space_type}")
        
        # 更新配置中的动作空间类型
        config.action_space_type = action_space_type
        
        # 更新单智能体环境配置
        if config.single_agent_env_config:
            config.single_agent_env_config.circuit_name = args.circuit_name
            config.single_agent_env_config.max_episode_steps = args.max_episode_steps
            if args.seed is not None:
                config.single_agent_env_config.seed = args.seed
        
        # 设置实验目录
        exp_log_dir, exp_model_dir = setup_experiment_directories(config)
        
        # 设置日志
        log_file = os.path.join(exp_log_dir, "training.log")
        setup_logging(args.log_level, log_file)
        
        # 打印训练信息
        print_training_info(config)
        
        # 创建环境
        logger.info("创建训练环境...")
        monitor_dir = os.path.join(exp_log_dir, "monitor")
        train_env = create_single_agent_env(
            config=config,
            n_envs=args.n_envs,
            seed=args.seed,
            monitor_dir=monitor_dir,
            normalize_env=args.normalize_env
        )
        
        # 创建评估环境
        logger.info("创建评估环境...")
        eval_monitor_dir = os.path.join(exp_log_dir, "eval_monitor")
        eval_env = create_single_agent_env(
            config=config,
            n_envs=1,
            seed=args.seed + 1000 if args.seed else None,
            monitor_dir=eval_monitor_dir,
            normalize_env=args.normalize_env
        )
        
        # 创建或加载模型
        logger.info("初始化模型...")
        if args.model_path:
            model, loaded_config = load_model_and_config(
                args.model_path,
                args.config_path,
                env=train_env
            )
            if loaded_config:
                logger.info("使用加载的配置更新当前配置")
                # 可以选择性地更新某些配置
        else:
            model = create_model(
                algorithm=args.algo,
                env=train_env,
                config=config
            )
        
        # 仅评估模式
        if args.eval_only:
            logger.info("进入评估模式...")
            eval_results = evaluate_model(
                model=model,
                eval_env=eval_env,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True
            )
            
            # 保存评估结果
            import json
            eval_results_path = os.path.join(exp_log_dir, "eval_results.json")
            with open(eval_results_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"评估结果已保存到: {eval_results_path}")
            return 0
        
        # 创建回调函数
        logger.info("设置训练回调...")
        best_model_path = os.path.join(exp_model_dir, "best_model")
        checkpoint_path = os.path.join(exp_model_dir, "checkpoints")
        callbacks = create_callbacks(
            config=config,
            eval_env=eval_env,
            best_model_save_path=best_model_path,
            checkpoint_save_path=checkpoint_path
        )
        
        # 开始训练
        logger.info("开始训练...")
        trained_model = train_model(
            model=model,
            config=config,
            callbacks=callbacks
        )
        
        # 最终评估
        logger.info("进行最终评估...")
        final_eval_results = evaluate_model(
            model=trained_model,
            eval_env=eval_env,
            n_eval_episodes=args.n_eval_episodes * 2,  # 更多回合的最终评估
            deterministic=True,
            return_episode_rewards=True
        )
        
        # 保存最终模型和配置
        logger.info("保存最终模型和配置...")
        save_model_and_config(
            model=trained_model,
            config=config,
            save_dir=exp_model_dir,
            model_name="final_model"
        )
        
        # 保存最终评估结果
        import json
        final_eval_path = os.path.join(exp_log_dir, "final_eval_results.json")
        with open(final_eval_path, 'w', encoding='utf-8') as f:
            json.dump(final_eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"最终评估结果已保存到: {final_eval_path}")
        
        # 训练完成
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"实验名称: {config.experiment_name}")
        logger.info(f"最终平均奖励: {final_eval_results['mean_reward']:.2f} ± {final_eval_results['std_reward']:.2f}")
        logger.info(f"日志目录: {exp_log_dir}")
        logger.info(f"模型目录: {exp_model_dir}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        return 1
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        logger.error(f"错误详情:\n{traceback.format_exc()}")
        return 1
        
    finally:
        # 清理资源
        try:
            if 'train_env' in locals():
                train_env.close()
            if 'eval_env' in locals():
                eval_env.close()
        except:
            pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)