# -*- coding: utf-8 -*-
"""
单智能体训练配置管理器

本模块提供了单智能体强化学习训练的完整配置管理功能，包括：
- 算法配置加载和管理
- 环境配置整合
- 训练参数配置
- 配置验证和默认值设置
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# 导入单智能体环境配置
from envs.smartgrid.single_agent.single_agent_config import SingleAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class SingleAgentTrainingConfig:
    """单智能体训练完整配置类"""
    
    # 基础配置
    algorithm: str = "ppo"  # 算法名称
    environment: str = "vvc_single"  # 环境名称
    experiment_name: str = "single_agent_exp"  # 实验名称
    
    # 路径配置
    config_root: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__)))
    log_dir: str = field(default_factory=lambda: "./logs")
    model_save_dir: str = field(default_factory=lambda: "./models")
    
    # 算法配置
    algo_config: Optional[Dict[str, Any]] = None
    
    # 环境配置
    env_config: Optional[Dict[str, Any]] = None
    single_agent_env_config: Optional[SingleAgentConfig] = None
    action_space_type: str = "discrete"  # 动作空间类型: "discrete" 或 "continuous"
    
    # 训练配置
    total_timesteps: int = 100000
    n_eval_episodes: int = 10
    eval_freq: int = 10000
    save_freq: int = 10000
    
    # 设备配置
    device: str = "auto"
    seed: Optional[int] = None
    
    # 日志配置
    verbose: int = 1
    tensorboard_log: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # 加载配置文件
        if self.algo_config is None:
            self.algo_config = self._load_algo_config()
        
        if self.env_config is None:
            self.env_config = self._load_env_config()
        
        if self.single_agent_env_config is None:
            self.single_agent_env_config = self._create_single_agent_env_config()
    
    def _load_algo_config(self) -> Dict[str, Any]:
        """加载算法配置文件"""
        algo_config_path = os.path.join(
            self.config_root, "single_agent_cfgs", f"{self.algorithm}.yaml"
        )
        
        if not os.path.exists(algo_config_path):
            logger.warning(f"算法配置文件不存在: {algo_config_path}，使用默认配置")
            return self._get_default_algo_config()
        
        try:
            with open(algo_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载算法配置: {algo_config_path}")
            return config
        except Exception as e:
            logger.error(f"加载算法配置失败: {e}，使用默认配置")
            return self._get_default_algo_config()
    
    def _load_env_config(self) -> Dict[str, Any]:
        """加载环境配置文件"""
        env_config_path = os.path.join(
            self.config_root, "envs_cfgs", f"{self.environment}.yaml"
        )
        
        if not os.path.exists(env_config_path):
            logger.warning(f"环境配置文件不存在: {env_config_path}，使用默认配置")
            return self._get_default_env_config()
        
        try:
            with open(env_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载环境配置: {env_config_path}")
            return config
        except Exception as e:
            logger.error(f"加载环境配置失败: {e}，使用默认配置")
            return self._get_default_env_config()
    
    def _create_single_agent_env_config(self) -> SingleAgentConfig:
        """创建单智能体环境配置"""
        # 从环境配置中提取参数
        env_name = self.env_config.get("env_name", "13Bus")
        max_steps = self.env_config.get("num_steps", 24)
        seed = self.env_config.get("seed", self.seed or 42)
        
        # 从单智能体特定配置中提取参数
        single_config = self.env_config.get("single_agent", {})
        obs_config = single_config.get("observation", {})
        action_config = single_config.get("action", {})
        reward_config = single_config.get("reward", {})
        
        # 创建SingleAgentConfig实例
        config = SingleAgentConfig(
            circuit_name=env_name,
            max_episode_steps=max_steps,
            seed=seed,
            
            # 观测配置
            normalize_observations=obs_config.get("normalize_obs", True),
            include_bus_voltages=True,
            include_power_flows=True,
            include_device_states=True,
            
            # 动作配置
            enable_capacitors=True,
            enable_regulators=True,
            enable_batteries=True,
            enable_pv_systems=True,
            
            # 奖励配置
            voltage_penalty_weight=reward_config.get("reward_weights", {}).get("voltage", 1.0),
            power_loss_weight=reward_config.get("reward_weights", {}).get("power_loss", 0.5),
            discharge_penalty_weight=0.5,
            
            # 训练配置
            log_level="INFO",
            save_episode_data=False
        )
        
        return config
    
    def _get_default_algo_config(self) -> Dict[str, Any]:
        """获取默认算法配置"""
        if self.algorithm == "ppo":
            return {
                "seed": {"seed_specify": True, "seed": 1},
                "device": {"cuda": True, "cuda_deterministic": True, "torch_threads": 4},
                "train": {
                    "n_rollout_threads": 1,
                    "total_timesteps": 100000,
                    "episode_length": 24,
                    "log_interval": 10,
                    "eval_interval": 1000,
                    "use_linear_lr_decay": True,
                    "save_interval": 10000
                },
                "eval": {
                    "use_eval": True,
                    "n_eval_rollout_threads": 1,
                    "eval_episodes": 5,
                    "deterministic_eval": True
                },
                "ppo": {
                    "learning_rate": 3e-4,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gae_lambda": 0.95,
                    "gamma": 0.99,
                    "clip_range": 0.2,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "n_steps": 2048
                },
                "model": {
                    "hidden_sizes": [64, 64],
                    "activation_func": "tanh",
                    "use_orthogonal_init": True,
                    "gain": 0.01
                }
            }
        elif self.algorithm == "dqn":
            return {
                "seed": {"seed_specify": True, "seed": 1},
                "device": {"cuda": True, "cuda_deterministic": True, "torch_threads": 4},
                "train": {
                    "total_timesteps": 100000,
                    "log_interval": 10,
                    "eval_interval": 1000,
                    "save_interval": 10000
                },
                "dqn": {
                    "learning_rate": 1e-4,
                    "buffer_size": 50000,
                    "learning_starts": 1000,
                    "batch_size": 32,
                    "tau": 1.0,
                    "gamma": 0.99,
                    "train_freq": 4,
                    "gradient_steps": 1,
                    "target_update_interval": 1000,
                    "exploration_fraction": 0.1,
                    "exploration_initial_eps": 1.0,
                    "exploration_final_eps": 0.05
                }
            }
        else:
            # 通用默认配置
            return {
                "seed": {"seed_specify": True, "seed": 1},
                "device": {"cuda": True, "cuda_deterministic": True, "torch_threads": 4},
                "train": {
                    "total_timesteps": 100000,
                    "log_interval": 10,
                    "eval_interval": 1000,
                    "save_interval": 10000
                }
            }
    
    def _get_default_env_config(self) -> Dict[str, Any]:
        """获取默认环境配置"""
        return {
            "env_name": "13Bus",
            "seed": 123456,
            "num_steps": 24,
            "num_workers": 1,
            "use_plot": False,
            "do_testing": False,
            "mode": "single",
            "useS": False,
            "big2small": False,
            "use_render": False,
            "record_node": False,
            "single_agent": {
                "observation": {
                    "include_history": False,
                    "history_length": 1,
                    "normalize_obs": True
                },
                "action": {
                    "action_type": "discrete",
                    "normalize_action": False
                },
                "reward": {
                    "reward_type": "composite",
                    "reward_weights": {
                        "voltage": 1.0,
                        "power_loss": 0.5,
                        "reactive_power": 0.3
                    },
                    "normalize_reward": False
                }
            }
        }
    
    def get_sb3_model_kwargs(self) -> Dict[str, Any]:
        """获取Stable Baselines3模型初始化参数"""
        algo_params = self.algo_config.get(self.algorithm, {})
        model_params = self.algo_config.get("model", {})
        
        kwargs = {
            "verbose": self.verbose,
            "device": self.device,
            "seed": self.seed or self.algo_config.get("seed", {}).get("seed", 1)
        }
        
        # 添加TensorBoard日志
        if self.tensorboard_log:
            kwargs["tensorboard_log"] = os.path.join(self.log_dir, "tensorboard")
        
        # 根据算法类型添加特定参数
        if self.algorithm == "ppo":
            kwargs.update({
                "learning_rate": algo_params.get("learning_rate", 3e-4),
                "n_steps": algo_params.get("n_steps", 2048),
                "batch_size": algo_params.get("batch_size", 64),
                "n_epochs": algo_params.get("n_epochs", 10),
                "gamma": algo_params.get("gamma", 0.99),
                "gae_lambda": algo_params.get("gae_lambda", 0.95),
                "clip_range": algo_params.get("clip_range", 0.2),
                "ent_coef": algo_params.get("ent_coef", 0.0),
                "vf_coef": algo_params.get("vf_coef", 0.5),
                "max_grad_norm": algo_params.get("max_grad_norm", 0.5)
            })
        elif self.algorithm == "dqn":
            kwargs.update({
                "learning_rate": algo_params.get("learning_rate", 1e-4),
                "buffer_size": algo_params.get("buffer_size", 50000),
                "learning_starts": algo_params.get("learning_starts", 1000),
                "batch_size": algo_params.get("batch_size", 32),
                "tau": algo_params.get("tau", 1.0),
                "gamma": algo_params.get("gamma", 0.99),
                "train_freq": algo_params.get("train_freq", 4),
                "gradient_steps": algo_params.get("gradient_steps", 1),
                "target_update_interval": algo_params.get("target_update_interval", 1000),
                "exploration_fraction": algo_params.get("exploration_fraction", 0.1),
                "exploration_initial_eps": algo_params.get("exploration_initial_eps", 1.0),
                "exploration_final_eps": algo_params.get("exploration_final_eps", 0.05)
            })
        elif self.algorithm == "sac":
            kwargs.update({
                "learning_rate": algo_params.get("learning_rate", 3e-4),
                "buffer_size": algo_params.get("buffer_size", 100000),
                "learning_starts": algo_params.get("learning_starts", 1000),
                "batch_size": algo_params.get("batch_size", 256),
                "tau": algo_params.get("tau", 0.005),
                "gamma": algo_params.get("gamma", 0.99),
                "train_freq": algo_params.get("train_freq", 1),
                "gradient_steps": algo_params.get("gradient_steps", 1)
            })
        elif self.algorithm == "a2c":
            kwargs.update({
                "learning_rate": algo_params.get("learning_rate", 7e-4),
                "n_steps": algo_params.get("n_steps", 5),
                "gamma": algo_params.get("gamma", 0.99),
                "gae_lambda": algo_params.get("gae_lambda", 1.0),
                "ent_coef": algo_params.get("ent_coef", 0.0),
                "vf_coef": algo_params.get("vf_coef", 0.5),
                "max_grad_norm": algo_params.get("max_grad_norm", 0.5)
            })
        
        return kwargs
    
    def get_training_kwargs(self) -> Dict[str, Any]:
        """获取训练参数"""
        train_config = self.algo_config.get("train", {})
        
        return {
            "total_timesteps": self.total_timesteps or train_config.get("total_timesteps", 100000),
            "log_interval": train_config.get("log_interval", 10),
            "eval_freq": self.eval_freq or train_config.get("eval_interval", 10000),
            "n_eval_episodes": self.n_eval_episodes or train_config.get("eval_episodes", 10),
            "save_freq": self.save_freq or train_config.get("save_interval", 10000)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "algorithm": self.algorithm,
            "environment": self.environment,
            "experiment_name": self.experiment_name,
            "total_timesteps": self.total_timesteps,
            "device": self.device,
            "seed": self.seed,
            "algo_config": self.algo_config,
            "env_config": self.env_config,
            "single_agent_env_config": self.single_agent_env_config.to_dict() if self.single_agent_env_config else None
        }
    
    @classmethod
    def from_args(cls, args) -> 'SingleAgentTrainingConfig':
        """从命令行参数创建配置"""
        config = cls(
            algorithm=getattr(args, 'algorithm', 'ppo'),
            environment=getattr(args, 'environment', 'vvc_single'),
            experiment_name=getattr(args, 'experiment_name', 'single_agent_exp'),
            total_timesteps=getattr(args, 'total_timesteps', 100000),
            device=getattr(args, 'device', 'auto'),
            seed=getattr(args, 'seed', None)
        )
        return config


# 预定义配置
DEFAULT_PPO_CONFIG = SingleAgentTrainingConfig(
    algorithm="ppo",
    environment="vvc_single",
    experiment_name="ppo_vvc_default",
    total_timesteps=100000
)

DEFAULT_DQN_CONFIG = SingleAgentTrainingConfig(
    algorithm="dqn",
    environment="vvc_single",
    experiment_name="dqn_vvc_default",
    total_timesteps=100000
)

DEFAULT_SAC_CONFIG = SingleAgentTrainingConfig(
    algorithm="sac",
    environment="vvc_single",
    experiment_name="sac_vvc_default",
    total_timesteps=100000
)

DEFAULT_A2C_CONFIG = SingleAgentTrainingConfig(
    algorithm="a2c",
    environment="vvc_single",
    experiment_name="a2c_vvc_default",
    total_timesteps=100000
)

DEFAULT_DDPG_CONFIG = SingleAgentTrainingConfig(
    algorithm="ddpg",
    environment="vvc_single",
    experiment_name="ddpg_vvc_default",
    total_timesteps=100000,
    action_space_type="continuous"
)

DEFAULT_TD3_CONFIG = SingleAgentTrainingConfig(
    algorithm="td3",
    environment="vvc_single",
    experiment_name="td3_vvc_default",
    total_timesteps=100000,
    action_space_type="continuous"
)

DEFAULT_HER_CONFIG = SingleAgentTrainingConfig(
    algorithm="her",
    environment="vvc_single",
    experiment_name="her_vvc_default",
    total_timesteps=100000,
    action_space_type="continuous"
)

# 配置注册表
CONFIG_REGISTRY = {
    "ppo": DEFAULT_PPO_CONFIG,
    "dqn": DEFAULT_DQN_CONFIG,
    "sac": DEFAULT_SAC_CONFIG,
    "a2c": DEFAULT_A2C_CONFIG,
    "ddpg": DEFAULT_DDPG_CONFIG,
    "td3": DEFAULT_TD3_CONFIG,
    "her": DEFAULT_HER_CONFIG
}


def get_config(algorithm: str = "ppo", **kwargs) -> SingleAgentTrainingConfig:
    """获取指定算法的配置
    
    Args:
        algorithm: 算法名称
        **kwargs: 额外的配置参数
    
    Returns:
        SingleAgentTrainingConfig: 配置对象
    """
    if algorithm in CONFIG_REGISTRY:
        config = CONFIG_REGISTRY[algorithm]
        # 更新配置参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    else:
        # 创建新配置
        return SingleAgentTrainingConfig(algorithm=algorithm, **kwargs)