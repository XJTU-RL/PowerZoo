#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用增强TensorBoard回调类
为强化学习训练提供增强的TensorBoard日志记录和模型保存功能
支持多种环境类型，包括PowerZoo、Gym等
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

# 可选的PowerZoo特定日志记录器
try:
    from envs.power_envs.powerzoo_llm.system_logger import SystemLogger
    POWERZOO_SYSTEM_LOGGER_AVAILABLE = True
except ImportError:
    SystemLogger = None
    POWERZOO_SYSTEM_LOGGER_AVAILABLE = False

try:
    from envs.power_envs.powerzoo_llm.powerzoo_llm_logger import PowerZooLLMLogger
    POWERZOO_LLM_LOGGER_AVAILABLE = True
except ImportError:
    PowerZooLLMLogger = None
    POWERZOO_LLM_LOGGER_AVAILABLE = False


class EnhancedTensorBoardCallback(BaseCallback):
    """
    通用增强TensorBoard回调类
    
    功能特性:
    - 支持多种环境类型（PowerZoo、Gym等）
    - 自动检测环境类型并启用相应的日志记录器
    - 提供通用的训练指标记录
    - 智能模型保存和管理
    - 可扩展的日志记录架构
    """
    
    def __init__(self, 
                 log_dir: str,
                 save_freq: int = 1000,
                 name_prefix: str = "enhanced_model",
                 buffer_size: int = 10000,
                 save_interval: int = 100,
                 enable_powerzoo_logging: bool = True,
                 env_type: str = "auto",
                 verbose: int = 1):
        """
        初始化通用增强TensorBoard回调
        
        Args:
            log_dir: 日志保存目录
            save_freq: 模型保存频率
            name_prefix: 模型保存名称前缀
            buffer_size: 缓冲区大小
            save_interval: 保存间隔
            enable_powerzoo_logging: 是否启用PowerZoo特定日志记录
            env_type: 环境类型 ('auto', 'powerzoo', 'gym', 'generic')
            verbose: 详细程度
        """
        super().__init__(verbose)
        
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.buffer_size = buffer_size
        self.save_interval = save_interval
        self.enable_powerzoo_logging = enable_powerzoo_logging
        self.env_type = env_type
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志记录器
        logger_name = f"{self.__class__.__name__}_{id(self)}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        
        # 如果没有处理器，添加控制台处理器
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
        # 初始化计数器
        self.n_calls = 0
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        
        # 环境检测标志
        self.detected_env_type = None
        self.is_powerzoo_env = False
        
        # 初始化PowerZoo系统日志记录器（如果可用且启用）
        self.system_logger = None
        if (self.enable_powerzoo_logging and 
            POWERZOO_SYSTEM_LOGGER_AVAILABLE and 
            self.env_type in ['auto', 'powerzoo']):
            try:
                self.system_logger = SystemLogger(
                    log_dir=os.path.join(self.log_dir, "system_logs"),
                    buffer_size=self.buffer_size,
                    save_interval=self.save_interval
                )
                self.logger.info("PowerZoo系统日志记录器初始化成功")
            except Exception as e:
                self.logger.warning(f"PowerZoo系统日志记录器初始化失败: {e}")
                self.system_logger = None
        
        # 初始化PowerZoo LLM日志记录器（如果可用且启用）
        self.powerzoo_logger = None
        if (self.enable_powerzoo_logging and 
            POWERZOO_LLM_LOGGER_AVAILABLE and 
            self.env_type in ['auto', 'powerzoo']):
            try:
                # 创建模拟的args参数
                mock_args = {
                    'seed': 123456,
                    'cuda': True,
                    'cuda_deterministic': False,
                    'n_training_threads': 1,
                    'n_rollout_threads': 1,
                    'num_mini_batch': 1,
                    'episode_length': 24,
                    'num_env_steps': 10000,
                    'ppo_epoch': 10,
                    'use_value_active_masks': True,
                    'use_eval': True,
                    'eval_interval': 25,
                    'save_interval': 1,
                    'log_interval': 10,
                    'use_wandb': False,
                    'use_tensorboard': True,
                    'model_dir': None
                }
                
                mock_algo_args = {
                    'train': {
                        'n_rollout_threads': 1,
                        'episode_length': 24,
                        'num_env_steps': 10000,
                        'ppo_epoch': 10,
                        'num_mini_batch': 1,
                        'use_value_active_masks': True,
                        'use_eval': True,
                        'eval_interval': 25,
                        'save_interval': 1,
                        'log_interval': 10
                    }
                }
                
                mock_env_args = {
                    'env_name': 'PowerZooLLM',
                    'scenario': 'single_agent',
                    'num_agents': 1
                }
                
                # 创建TensorBoard writer
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
                
                self.powerzoo_logger = PowerZooLLMLogger(
                    args=mock_args,
                    algo_args=mock_algo_args,
                    env_args=mock_env_args,
                    num_agents=1,
                    writter=writer,
                    run_dir=self.log_dir
                )
                self.logger.info("PowerZoo LLM日志记录器初始化成功")
            except Exception as e:
                self.logger.warning(f"PowerZoo LLM日志记录器初始化失败: {e}")
                self.powerzoo_logger = None
    
    def _init_callback(self) -> None:
        """
        初始化回调
        """
        # 创建模型保存目录
        self.save_path = os.path.join(self.log_dir, "models")
        os.makedirs(self.save_path, exist_ok=True)
        
        # 自动检测环境类型
        self._detect_environment_type()
        
        if self.verbose > 0:
            print(f"Enhanced TensorBoard callback initialized")
            print(f"Log directory: {self.log_dir}")
            print(f"Model save path: {self.save_path}")
            print(f"Detected environment type: {self.detected_env_type}")
            print(f"PowerZoo logging enabled: {self.is_powerzoo_env and self.enable_powerzoo_logging}")
    
    def _detect_environment_type(self) -> None:
        """
        自动检测环境类型
        """
        if self.env_type != "auto":
            self.detected_env_type = self.env_type
            self.is_powerzoo_env = (self.env_type == "powerzoo")
            return
        
        # 尝试从训练环境检测
        if hasattr(self, 'training_env') and self.training_env is not None:
            env = self.training_env
            
            # 检查是否为PowerZoo环境
            if hasattr(env, 'envs') and len(env.envs) > 0:
                base_env = env.envs[0]
                # 处理Monitor包装
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                env_class_name = base_env.__class__.__name__
                env_module = base_env.__class__.__module__
                
                if 'powerzoo' in env_module.lower() or 'powerzoo' in env_class_name.lower():
                    self.detected_env_type = "powerzoo"
                    self.is_powerzoo_env = True
                elif 'gym' in env_module.lower():
                    self.detected_env_type = "gym"
                    self.is_powerzoo_env = False
                else:
                    self.detected_env_type = "generic"
                    self.is_powerzoo_env = False
            else:
                self.detected_env_type = "generic"
                self.is_powerzoo_env = False
        else:
            self.detected_env_type = "generic"
            self.is_powerzoo_env = False
        
        if self.verbose > 1:
            self.logger.info(f"Environment type detected: {self.detected_env_type}")
    
    def _on_step(self) -> bool:
        """
        每步调用的回调函数
        """
        self.n_calls += 1
        
        # 记录训练指标到TensorBoard
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # 记录基本训练指标
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                for i, info in enumerate(self.locals['infos']):
                    if info:
                        # 记录环境特定指标
                        self._log_env_metrics(info, self.n_calls)
        
        # 定期保存模型
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        rollout结束时的回调
        """
        # 记录rollout统计信息
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                reward = ep_info['r']
                self.model.logger.record('rollout/ep_rew_mean', reward)
                
                # 检查是否是最佳模型
                if reward > self.best_mean_reward:
                    self.best_mean_reward = reward
                    self._save_best_model()
    
    def _log_env_metrics(self, info: Dict[str, Any], step: int) -> None:
        """
        记录环境特定指标（支持多种环境类型）
        """
        try:
            # 通用指标记录
            self._log_generic_metrics(info, step)
            
            # PowerZoo特定指标记录
            if self.is_powerzoo_env and self.enable_powerzoo_logging:
                self._log_powerzoo_metrics(info, step)
            
            # Gym环境特定指标记录
            elif self.detected_env_type == "gym":
                self._log_gym_metrics(info, step)
        
        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Failed to log env metrics: {e}")
    
    def _log_generic_metrics(self, info: Dict[str, Any], step: int) -> None:
        """
        记录通用环境指标
        """
        # 记录基本奖励信息
        if 'episode' in info:
            episode_info = info['episode']
            if 'r' in episode_info:
                self.model.logger.record('env/episode_reward', episode_info['r'])
            if 'l' in episode_info:
                self.model.logger.record('env/episode_length', episode_info['l'])
        
        # 记录自定义奖励组成
        if 'reward_components' in info:
            components = info['reward_components']
            for key, value in components.items():
                self.model.logger.record(f'env/reward_{key}', value)
        
        # 记录其他通用指标
        for key, value in info.items():
            if isinstance(value, (int, float, np.number)):
                self.model.logger.record(f'env/{key}', value)
    
    def _log_powerzoo_metrics(self, info: Dict[str, Any], step: int) -> None:
        """
        记录PowerZoo特定指标
        """
        # 记录系统状态
        if 'system_state' in info:
            state = info['system_state']
            if 'voltage_violations' in state:
                self.model.logger.record('powerzoo/voltage_violations', state['voltage_violations'])
            if 'power_loss' in state:
                self.model.logger.record('powerzoo/power_loss', state['power_loss'])
            if 'load_served' in state:
                self.model.logger.record('powerzoo/load_served', state['load_served'])
        
        # 使用PowerZoo系统日志记录器记录详细状态
        if self.system_logger and hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):  # 处理Monitor包装
                env = env.env
            
            # 获取最后的动作
            actions = getattr(self.locals, 'actions', np.array([0]))
            reward = info.get('reward', 0.0)
            
            self.system_logger.log_system_state(
                env=env,
                actions=actions,
                reward=reward,
                info=info
            )
    
    def _log_gym_metrics(self, info: Dict[str, Any], step: int) -> None:
        """
        记录Gym环境特定指标
        """
        # Gym环境通常使用标准的episode信息
        # 这里可以添加Gym特定的指标记录逻辑
        pass
    
    def _save_model(self) -> None:
        """
        保存模型
        """
        try:
            model_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{self.n_calls}_steps"
            )
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}: {model_path}")
        
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to save model: {e}")
    
    def _save_best_model(self) -> None:
        """
        保存最佳模型
        """
        try:
            best_model_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_best"
            )
            self.model.save(best_model_path)
            
            if self.verbose > 0:
                print(f"Best model saved with reward {self.best_mean_reward:.2f}: {best_model_path}")
        
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to save best model: {e}")
    
    def _on_training_end(self) -> None:
        """
        训练结束时的回调
        """
        # 保存最终模型
        self._save_model()
        
        # 关闭日志记录器
        if self.system_logger:
            try:
                self.system_logger.close()
                if self.verbose > 1:
                    self.logger.info("PowerZoo系统日志记录器已关闭")
            except Exception as e:
                self.logger.warning(f"Failed to close system logger: {e}")
        
        if self.powerzoo_logger:
            try:
                # PowerZoo logger可能没有close方法，跳过
                if hasattr(self.powerzoo_logger, 'close'):
                    self.powerzoo_logger.close()
                if self.verbose > 1:
                    self.logger.info("PowerZoo LLM日志记录器已关闭")
            except Exception as e:
                self.logger.warning(f"Failed to close PowerZoo logger: {e}")
        
        if self.verbose > 0:
            print("Enhanced TensorBoard callback training ended")
            print(f"Environment type: {self.detected_env_type}")
            print(f"Total steps: {self.n_calls}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print(f"Models saved to: {self.save_path}")


# 使用示例
def create_enhanced_callback(log_dir: str, 
                           env_type: str = "auto",
                           enable_powerzoo_logging: bool = True,
                           **kwargs) -> EnhancedTensorBoardCallback:
    """
    创建增强TensorBoard回调的便捷函数
    
    Args:
        log_dir: 日志目录
        env_type: 环境类型 ('auto', 'powerzoo', 'gym', 'generic')
        enable_powerzoo_logging: 是否启用PowerZoo特定日志记录
        **kwargs: 其他回调参数
    
    Returns:
        配置好的EnhancedTensorBoardCallback实例
    
    Examples:
        # 自动检测环境类型
        callback = create_enhanced_callback("/path/to/logs")
        
        # 明确指定PowerZoo环境
        callback = create_enhanced_callback(
            "/path/to/logs", 
            env_type="powerzoo",
            save_freq=500
        )
        
        # 用于Gym环境，禁用PowerZoo日志记录
        callback = create_enhanced_callback(
            "/path/to/logs", 
            env_type="gym",
            enable_powerzoo_logging=False
        )
    """
    return EnhancedTensorBoardCallback(
        log_dir=log_dir,
        env_type=env_type,
        enable_powerzoo_logging=enable_powerzoo_logging,
        **kwargs
    )