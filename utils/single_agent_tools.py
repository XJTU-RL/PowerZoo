# -*- coding: utf-8 -*-
"""
单智能体训练工具模块

本模块提供了单智能体强化学习训练的核心工具函数，包括：
- 环境创建和包装
- 模型初始化和配置
- 训练回调函数设置
- 评估和监控工具
- 模型保存和加载
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path

# Stable Baselines3 imports
from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3

# 尝试导入HER，如果不可用则跳过
try:
    from stable_baselines3 import HER
    HER_AVAILABLE = True
except ImportError:
    HER = None
    HER_AVAILABLE = False
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, 
    StopTrainingOnRewardThreshold, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

# PowerZoo imports
from envs.smartgrid.single_agent.single_agent_env import SingleAgentPowerZooEnv
from envs.smartgrid.single_agent.single_agent_config import SingleAgentConfig
from envs.smartgrid.single_agent.single_agent_training_config import SingleAgentTrainingConfig

logger = logging.getLogger(__name__)

# 算法注册表
ALGORITHM_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
    "sac": SAC,
    "a2c": A2C,
    "ddpg": DDPG,
    "td3": TD3,
}

# 如果HER可用，添加到注册表
if HER_AVAILABLE:
    ALGORITHM_REGISTRY["her"] = HER

# 策略注册表
POLICY_REGISTRY = {
    "ppo": "MlpPolicy",
    "dqn": "MlpPolicy",
    "sac": "MlpPolicy",
    "a2c": "MlpPolicy",
    "ddpg": "MlpPolicy",
    "td3": "MlpPolicy",
}

# 如果HER可用，添加到策略注册表
if HER_AVAILABLE:
    POLICY_REGISTRY["her"] = "MlpPolicy"


class SingleAgentTrainingLogger(BaseCallback):
    """单智能体训练日志回调"""
    
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 记录回合奖励和长度
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        # 定期记录统计信息
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])  # 最近100个回合的平均奖励
                mean_length = np.mean(self.episode_lengths[-100:])  # 最近100个回合的平均长度
                
                self.logger.record("train/mean_episode_reward", mean_reward)
                self.logger.record("train/mean_episode_length", mean_length)
                self.logger.record("train/num_episodes", len(self.episode_rewards))
                
                if self.verbose >= 1:
                    logger.info(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.2f}")
        
        return True


def create_single_agent_env(
    config: SingleAgentTrainingConfig,
    env_id: Optional[str] = None,
    n_envs: int = 1,
    seed: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    normalize_env: bool = False
):
    """创建单智能体环境
    
    Args:
        config: 训练配置
        env_id: 环境ID（可选）
        n_envs: 并行环境数量
        seed: 随机种子
        monitor_dir: Monitor日志目录
        normalize_env: 是否标准化环境
    
    Returns:
        创建的环境
    """
    
    def _make_env(rank: int = 0):
        """创建单个环境的工厂函数"""
        def _init():
            # 创建环境
            env = SingleAgentPowerZooEnv(
                config=config.single_agent_env_config,
                action_space_type=config.action_space_type
            )
            
            # 设置随机种子
            if seed is not None:
                env.seed(seed + rank)
                set_random_seed(seed + rank)
            
            # 添加Monitor包装
            if monitor_dir:
                os.makedirs(monitor_dir, exist_ok=True)
                monitor_path = os.path.join(monitor_dir, f"env_{rank}")
                env = Monitor(env, monitor_path)
            
            return env
        return _init
    
    # 创建向量化环境
    if n_envs == 1:
        env = DummyVecEnv([_make_env(0)])
    else:
        env = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    
    # 环境标准化
    if normalize_env:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    logger.info(f"创建了 {n_envs} 个并行环境")
    return env


def create_model(
    algorithm: str,
    env,
    config: SingleAgentTrainingConfig,
    model_path: Optional[str] = None
):
    """创建RL模型
    
    Args:
        algorithm: 算法名称
        env: 环境
        config: 训练配置
        model_path: 预训练模型路径（可选）
    
    Returns:
        创建的模型
    """
    
    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(f"不支持的算法: {algorithm}。支持的算法: {list(ALGORITHM_REGISTRY.keys())}")
    
    # 获取算法类和策略
    algorithm_class = ALGORITHM_REGISTRY[algorithm]
    policy = POLICY_REGISTRY[algorithm]
    
    # 获取模型参数
    model_kwargs = config.get_sb3_model_kwargs()
    
    # 设置日志记录器
    if config.tensorboard_log:
        tensorboard_log = os.path.join(config.log_dir, "tensorboard")
        os.makedirs(tensorboard_log, exist_ok=True)
        model_kwargs["tensorboard_log"] = tensorboard_log
    
    # 创建模型
    if model_path and os.path.exists(model_path):
        logger.info(f"从 {model_path} 加载预训练模型")
        model = algorithm_class.load(model_path, env=env, **model_kwargs)
    else:
        logger.info(f"创建新的 {algorithm.upper()} 模型")
        model = algorithm_class(policy, env, **model_kwargs)
    
    # 设置自定义日志记录器
    if config.log_dir:
        log_path = os.path.join(config.log_dir, "sb3_logs")
        os.makedirs(log_path, exist_ok=True)
        new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
    
    logger.info(f"成功创建 {algorithm.upper()} 模型")
    return model


def create_callbacks(
    config: SingleAgentTrainingConfig,
    eval_env,
    best_model_save_path: Optional[str] = None,
    checkpoint_save_path: Optional[str] = None
) -> CallbackList:
    """创建训练回调函数
    
    Args:
        config: 训练配置
        eval_env: 评估环境
        best_model_save_path: 最佳模型保存路径
        checkpoint_save_path: 检查点保存路径
    
    Returns:
        回调函数列表
    """
    
    callbacks = []
    
    # 训练日志回调
    training_logger = SingleAgentTrainingLogger(
        log_freq=config.algo_config.get("train", {}).get("log_interval", 100),
        verbose=config.verbose
    )
    callbacks.append(training_logger)
    
    # 评估回调
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path or os.path.join(config.model_save_dir, "best_model"),
            log_path=os.path.join(config.log_dir, "eval_logs"),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=config.verbose
        )
        callbacks.append(eval_callback)
    
    # 检查点保存回调
    if checkpoint_save_path or config.save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=config.save_freq,
            save_path=checkpoint_save_path or os.path.join(config.model_save_dir, "checkpoints"),
            name_prefix="model_checkpoint",
            verbose=config.verbose
        )
        callbacks.append(checkpoint_callback)
    
    # 早停回调（可选）
    reward_threshold = config.algo_config.get("train", {}).get("reward_threshold")
    if reward_threshold:
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=config.verbose
        )
        callbacks.append(stop_callback)
    
    return CallbackList(callbacks)


def train_model(
    model,
    config: SingleAgentTrainingConfig,
    callbacks: Optional[CallbackList] = None
):
    """训练模型
    
    Args:
        model: RL模型
        config: 训练配置
        callbacks: 回调函数列表
    
    Returns:
        训练后的模型
    """
    
    # 获取训练参数
    training_kwargs = config.get_training_kwargs()
    total_timesteps = training_kwargs["total_timesteps"]
    
    logger.info(f"开始训练，总步数: {total_timesteps}")
    
    try:
        # 开始训练
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=training_kwargs.get("log_interval", 10),
            reset_num_timesteps=True
        )
        
        logger.info("训练完成")
        
        # 保存最终模型
        final_model_path = os.path.join(config.model_save_dir, "final_model")
        model.save(final_model_path)
        logger.info(f"最终模型已保存到: {final_model_path}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise
    
    return model


def evaluate_model(
    model,
    eval_env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    return_episode_rewards: bool = False
):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        eval_env: 评估环境
        n_eval_episodes: 评估回合数
        deterministic: 是否使用确定性策略
        render: 是否渲染
        return_episode_rewards: 是否返回每个回合的奖励
    
    Returns:
        评估结果
    """
    
    logger.info(f"开始评估模型，评估回合数: {n_eval_episodes}")
    
    try:
        # 评估模型
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            return_episode_rewards=True
        )
        
        # 计算统计信息
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        logger.info(f"评估结果:")
        logger.info(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  平均回合长度: {mean_length:.2f} ± {std_length:.2f}")
        
        results = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "n_eval_episodes": n_eval_episodes
        }
        
        if return_episode_rewards:
            results["episode_rewards"] = episode_rewards
            results["episode_lengths"] = episode_lengths
        
        return results
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        raise


def save_model_and_config(
    model,
    config: SingleAgentTrainingConfig,
    save_dir: str,
    model_name: str = "model"
):
    """保存模型和配置
    
    Args:
        model: 训练好的模型
        config: 训练配置
        save_dir: 保存目录
        model_name: 模型名称
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, f"{model_name}.zip")
    model.save(model_path)
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存配置
    config_path = os.path.join(save_dir, f"{model_name}_config.yaml")
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
    logger.info(f"配置已保存到: {config_path}")


def load_model_and_config(
    model_path: str,
    config_path: Optional[str] = None,
    env = None
):
    """加载模型和配置
    
    Args:
        model_path: 模型路径
        config_path: 配置路径（可选）
        env: 环境（可选）
    
    Returns:
        (model, config) 元组
    """
    
    # 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 根据文件扩展名确定算法类型
    algorithm = None
    for algo_name, algo_class in ALGORITHM_REGISTRY.items():
        try:
            model = algo_class.load(model_path, env=env)
            algorithm = algo_name
            break
        except:
            continue
    
    if model is None:
        raise ValueError(f"无法加载模型: {model_path}")
    
    logger.info(f"成功加载 {algorithm.upper()} 模型: {model_path}")
    
    # 加载配置
    config = None
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 重建配置对象
        config = SingleAgentTrainingConfig(
            algorithm=config_dict.get("algorithm", algorithm),
            environment=config_dict.get("environment", "powerzoo_single"),
            experiment_name=config_dict.get("experiment_name", "loaded_model")
        )
        logger.info(f"成功加载配置: {config_path}")
    
    return model, config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志记录
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {log_level}")
    
    # 配置日志记录器
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"日志记录已设置，级别: {log_level}")


def get_device_info():
    """获取设备信息"""
    import torch
    
    device_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info["current_device"] = torch.cuda.current_device()
        device_info["device_name"] = torch.cuda.get_device_name()
    
    return device_info


def print_training_info(config: SingleAgentTrainingConfig):
    """打印训练信息"""
    
    logger.info("=" * 60)
    logger.info("单智能体强化学习训练信息")
    logger.info("=" * 60)
    logger.info(f"算法: {config.algorithm.upper()}")
    logger.info(f"环境: {config.environment}")
    logger.info(f"实验名称: {config.experiment_name}")
    logger.info(f"总训练步数: {config.total_timesteps}")
    logger.info(f"设备: {config.device}")
    logger.info(f"随机种子: {config.seed}")
    logger.info(f"日志目录: {config.log_dir}")
    logger.info(f"模型保存目录: {config.model_save_dir}")
    
    # 打印设备信息
    device_info = get_device_info()
    logger.info(f"PyTorch版本: {device_info['torch_version']}")
    logger.info(f"CUDA可用: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        logger.info(f"CUDA版本: {device_info['cuda_version']}")
        logger.info(f"GPU设备: {device_info['device_name']}")
    
    # 打印环境配置
    if config.single_agent_env_config:
        env_config = config.single_agent_env_config
        logger.info(f"电路名称: {env_config.circuit_name}")
        logger.info(f"最大步数: {env_config.max_episode_steps}")
        logger.info(f"电压惩罚权重: {env_config.voltage_penalty_weight}")
        logger.info(f"功率损耗权重: {env_config.power_loss_weight}")
    
    logger.info("=" * 60)