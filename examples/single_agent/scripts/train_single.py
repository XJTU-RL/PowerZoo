# -*- coding: utf-8 -*-
"""
@File      : train_single.py
@Time      : 2025-01-XX XX:XX
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 单智能体强化学习训练入口文件。
此文件基于Stable Baselines3框架，支持PPO、DQN、SAC、A2C等主流单智能体算法。
主要功能：
- 解析命令行参数和配置文件
- 创建PowerZoo单智能体环境
- 初始化和训练RL算法
- 模型保存和评估
- 支持TensorBoard日志记录
"""

import argparse
import os
import sys
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# 将项目根目录添加到系统路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Stable Baselines3 imports
from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

# HER (Hindsight Experience Replay) import
try:
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.her import HER
    HER_AVAILABLE = True
except ImportError:
    HER_AVAILABLE = False
    print("Warning: HER is not available. Please install stable-baselines3[extra] for HER support.")

# PowerZoo imports
from envs.power_envs.powerzoo_llm.single_agent.single_agent_env import SingleAgentPowerZooEnv
from utils.tensorboard_callback import EnhancedTensorBoardCallback
from envs.power_envs.powerzoo_llm.model_utils.model_manager import ModelManager

# 算法映射字典
ALGORITHM_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
    "sac": SAC,
    "a2c": A2C,
    "ddpg": DDPG,
    "td3": TD3,
}

# HER算法需要特殊处理
if HER_AVAILABLE:
    ALGORITHM_REGISTRY["her"] = HER

# 支持的策略类型
POLICY_REGISTRY = {
    "ppo": "MlpPolicy",
    "dqn": "MlpPolicy",
    "sac": "MlpPolicy",
    "a2c": "MlpPolicy",
    "ddpg": "MlpPolicy",
    "td3": "MlpPolicy",
    "her": "MlpPolicy",  # HER使用底层算法的策略
}

def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_environment(env_config: Dict[str, Any], seed: Optional[int] = None):
    """创建PowerZoo单智能体环境"""
    from envs.power_envs.powerzoo_llm.single_agent.single_agent_config import SingleAgentConfig
    
    # 创建配置对象
    config = SingleAgentConfig(
        circuit_name=env_config.get('env_name', '13Bus'),
        seed=seed or env_config.get('seed', 123456),
        max_episode_steps=env_config.get('num_steps', 24),
        log_level=env_config.get('log_level', 'INFO')
    )
    
    # 创建环境
    env = SingleAgentPowerZooEnv(config=config)
    
    # 使用Monitor包装环境以记录统计信息
    env = Monitor(env)
    
    return env

def setup_callbacks(algo_config: Dict[str, Any], env_config: Dict[str, Any], 
                   log_dir: str, eval_env, algorithm_name: str = "unknown") -> list:
    """设置训练回调函数"""
    callbacks = []
    
    # 增强TensorBoard回调（集成系统日志和PowerZoo LLM日志）
    tensorboard_callback = EnhancedTensorBoardCallback(
        log_dir=os.path.join(log_dir, 'tensorboard'),
        log_freq=algo_config.get('train', {}).get('log_interval', 100),
        save_freq=algo_config.get('train', {}).get('save_interval', 10000),
        model_save_path=os.path.join(log_dir, 'models'),
        verbose=1,
        algorithm_name=algorithm_name,
        enable_system_logging=True,
        enable_powerzoo_logging=True
    )
    callbacks.append(tensorboard_callback)
    
    # 评估回调
    if algo_config.get('eval', {}).get('use_eval', True):
        eval_freq = algo_config.get('train', {}).get('eval_interval', 1000)
        n_eval_episodes = algo_config.get('eval', {}).get('eval_episodes', 5)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, 'best_model'),
            log_path=os.path.join(log_dir, 'eval'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=algo_config.get('eval', {}).get('deterministic_eval', True),
            render=False
        )
        callbacks.append(eval_callback)
    
    # 检查点回调
    save_freq = algo_config.get('train', {}).get('save_interval', 10000)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='model'
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks

def create_model(algo_name: str, env, algo_config: Dict[str, Any], 
                log_dir: str, device: str = "auto"):
    """创建RL模型"""
    algorithm_class = ALGORITHM_REGISTRY[algo_name]
    policy_type = POLICY_REGISTRY[algo_name]
    
    # 获取算法特定配置
    algo_params = algo_config.get(algo_name, {})
    
    # HER算法需要特殊处理
    if algo_name == "her" and HER_AVAILABLE:
        # HER需要包装其他算法
        base_algo = algo_params.get('base_algorithm', 'ddpg')
        base_algo_class = ALGORITHM_REGISTRY.get(base_algo, DDPG)
        
        # 创建基础算法的参数
        base_params = algo_config.get(base_algo, {})
        base_model_params = {
            'policy': policy_type,
            'env': env,
            'verbose': 1,
            'device': device,
            'seed': algo_config.get('seed', 1),
            **base_params
        }
        
        # 移除不属于模型初始化的参数
        excluded_keys = ['total_timesteps', 'save_interval', 'base_algorithm']
        for key in excluded_keys:
            base_model_params.pop(key, None)
            algo_params.pop(key, None)
        
        print(f"创建HER模型，基础算法: {base_algo.upper()}，参数: {base_model_params}")
        
        try:
            # 创建HER模型
            model = HER(
                policy=policy_type,
                env=env,
                model_class=base_algo_class,
                verbose=1,
                tensorboard_log=log_dir,
                **algo_params
            )
            return model
        except Exception as e:
            print(f"创建HER模型时出错: {e}")
            print("使用默认HER参数重试...")
            model = HER(
                policy=policy_type,
                env=env,
                model_class=DDPG,
                verbose=1,
                tensorboard_log=log_dir
            )
            return model
    
    # 普通算法处理
    # 设置TensorBoard日志目录
    tensorboard_log_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # 通用参数
    common_params = {
        'policy': policy_type,
        'env': env,
        'verbose': 1,
        'tensorboard_log': tensorboard_log_dir,
        'device': device,
        'seed': algo_config.get('seed', 1)
    }
    
    # 合并算法特定参数
    model_params = {**common_params, **algo_params}
    
    # 移除不属于模型初始化的参数
    excluded_keys = ['total_timesteps', 'save_interval']
    for key in excluded_keys:
        model_params.pop(key, None)
    
    print(f"创建{algo_name.upper()}模型，参数: {model_params}")
    
    try:
        model = algorithm_class(**model_params)
        return model
    except Exception as e:
        print(f"创建模型时出错: {e}")
        # 使用默认参数重试
        print("使用默认参数重试...")
        model = algorithm_class(
            policy=policy_type,
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
        return model

def train_model(model, algo_config: Dict[str, Any], callbacks: list):
    """训练模型"""
    total_timesteps = algo_config.get('train', {}).get('total_timesteps', 100000)
    
    print(f"开始训练，总步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    return model

def evaluate_model(model, eval_env, n_eval_episodes: int = 10):
    """评估模型性能"""
    print(f"评估模型性能，评估episode数: {n_eval_episodes}")
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 算法选择
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=list(ALGORITHM_REGISTRY.keys()),
        help="选择单智能体算法: ppo, dqn, sac, a2c, ddpg, td3, her"
    )
    
    # 环境选择
    parser.add_argument(
        "--env",
        type=str,
        default="powerzoo_single",
        help="环境配置名称"
    )
    
    # 实验名称
    parser.add_argument(
        "--exp_name",
        type=str,
        default="single_agent_test",
        help="实验名称"
    )
    
    # 配置文件路径
    parser.add_argument(
        "--algo_config",
        type=str,
        default="",
        help="算法配置文件路径"
    )
    
    parser.add_argument(
        "--env_config",
        type=str,
        default="",
        help="环境配置文件路径"
    )
    
    # 设备选择
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备: auto, cpu, cuda"
    )
    
    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="随机种子"
    )
    
    # 总训练步数
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="总训练步数"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 加载配置文件
    if args.algo_config:
        algo_config = load_config(args.algo_config)
    else:
        algo_config_path = f"configs/single_agent_cfgs/{args.algo}.yaml"
        algo_config = load_config(algo_config_path)
    
    if args.env_config:
        env_config = load_config(args.env_config)
    else:
        env_config_path = f"configs/envs_cfgs/{args.env}.yaml"
        env_config = load_config(env_config_path)
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/single_agent/{args.algo}_{args.env}_{args.exp_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"日志目录: {log_dir}")
    
    # 保存配置文件
    with open(os.path.join(log_dir, 'algo_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(algo_config, f, default_flow_style=False, allow_unicode=True)
    
    with open(os.path.join(log_dir, 'env_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(env_config, f, default_flow_style=False, allow_unicode=True)
    
    try:
        # 创建环境
        print("创建训练环境...")
        train_env = create_environment(env_config, args.seed)
        
        print("创建评估环境...")
        eval_env = create_environment(env_config, args.seed + 1000)
        
        # 初始化模型管理器
        model_manager = ModelManager(
            base_dir=os.path.join(log_dir, 'model_manager'),
            max_versions=5,
            auto_backup=True
        )
        
        # 设置回调函数
        callbacks = setup_callbacks(algo_config, env_config, log_dir, eval_env, args.algo)
        
        # 创建模型
        print(f"创建{args.algo.upper()}模型...")
        model = create_model(args.algo, train_env, algo_config, log_dir, args.device)
        
        # 配置自定义日志记录器
        custom_log_dir = os.path.join(log_dir, 'sb3_logs')
        os.makedirs(custom_log_dir, exist_ok=True)
        new_logger = configure(custom_log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # 训练模型
        if args.total_timesteps:
            algo_config.setdefault('train', {})['total_timesteps'] = args.total_timesteps
        print("开始训练...")
        model = train_model(model, algo_config, callbacks)
        
        # 最终评估
        print("进行最终评估...")
        mean_reward, std_reward = evaluate_model(model, eval_env, 10)
        
        # 准备性能指标
        performance_metrics = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'total_timesteps': algo_config.get('train', {}).get('total_timesteps', 100000)
        }
        
        # 使用模型管理器保存最终模型
        model_name = f"{args.algo}_{args.env}_{args.exp_name}"
        model_manager.save_model(
            model=model,
            model_name=model_name,
            training_config=algo_config,
            environment_config=env_config,
            performance_metrics=performance_metrics,
            description=f"Final trained model for {args.exp_name}",
            tags=[args.algo, args.env, 'final_model']
        )
        
        # 传统方式也保存一份（兼容性）
        final_model_path = os.path.join(log_dir, 'final_model')
        model.save(final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        
        # 保存评估结果
        eval_results = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'algorithm': args.algo,
            'environment': args.env,
            'experiment_name': args.exp_name,
            'total_timesteps': algo_config.get('train', {}).get('total_timesteps', 100000)
        }
        
        with open(os.path.join(log_dir, 'eval_results.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(eval_results, f, default_flow_style=False, allow_unicode=True)
        
        print("训练完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理资源
        try:
            train_env.close()
            eval_env.close()
        except:
            pass
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)