#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerZoo LLM增强TensorBoard回调类
集成系统日志记录器和PowerZoo LLM日志记录器，提供详细的训练可视化
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import torch
from datetime import datetime

# Optional imports for extended functionality
try:
    from envs.power_envs.powerzoo_llm.logging.system_logger import SystemLogger, SystemState
    SYSTEM_LOGGER_AVAILABLE = True
except ImportError:
    SYSTEM_LOGGER_AVAILABLE = False
    SystemLogger = None
    SystemState = None

try:
    from envs.power_envs.powerzoo_llm.logging.powerzoo_llm_logger import PowerZooLLMLogger
    POWERZOO_LOGGER_AVAILABLE = True
except ImportError:
    POWERZOO_LOGGER_AVAILABLE = False
    PowerZooLLMLogger = None


class EnhancedTensorBoardCallback(BaseCallback):
    """
    增强的TensorBoard回调类，集成PowerZoo LLM环境的详细日志记录
    
    Features:
    - 记录训练指标到TensorBoard
    - 集成SystemLogger记录系统状态
    - 集成PowerZooLLMLogger记录环境特定指标
    - 自动保存最佳模型
    - 记录算法特定参数和性能指标
    """
    
    def __init__(
        self,
        log_dir: str,
        log_freq: int = 100,
        save_freq: int = 10000,
        model_save_path: Optional[str] = None,
        verbose: int = 0,
        algorithm_name: str = "unknown",
        enable_system_logging: bool = True,
        enable_powerzoo_logging: bool = True
    ):
        """
        初始化增强TensorBoard回调
        
        Args:
            log_dir: TensorBoard日志目录
            log_freq: 日志记录频率（步数）
            save_freq: 模型保存频率（步数）
            model_save_path: 模型保存路径
            verbose: 详细程度
            algorithm_name: 算法名称
            enable_system_logging: 是否启用系统日志记录
            enable_powerzoo_logging: 是否启用PowerZoo LLM日志记录
        """
        super().__init__(verbose)
        
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.model_save_path = model_save_path or os.path.join(log_dir, "models")
        self.algorithm_name = algorithm_name
        self.enable_system_logging = enable_system_logging
        self.enable_powerzoo_logging = enable_powerzoo_logging
        
        # 创建保存目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # 初始化日志记录器
        self.system_logger = None
        self.powerzoo_logger = None
        self.tensorboard_writer = None
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        
        # 算法特定指标
        self.algorithm_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'learning_rate': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
    def _init_callback(self) -> None:
        """
        初始化回调函数
        """
        # 获取TensorBoard writer
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tensorboard_writer = output_format.writer
                break
        
        if self.tensorboard_writer is None:
            if self.verbose > 0:
                print("Warning: TensorBoard writer not found. Some logging features may not work.")
        
        # 初始化系统日志记录器
        if self.enable_system_logging and SYSTEM_LOGGER_AVAILABLE:
            try:
                system_log_dir = os.path.join(self.log_dir, "system_logs")
                self.system_logger = SystemLogger(
                    log_dir=system_log_dir,
                    max_buffer_size=1000,
                    flush_interval=100
                )
                if self.verbose > 0:
                    print(f"System logger initialized at: {system_log_dir}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to initialize system logger: {e}")
                self.enable_system_logging = False
        else:
            self.enable_system_logging = False
            if self.verbose > 0 and not SYSTEM_LOGGER_AVAILABLE:
                print("System logger module not available")
        
        # 初始化PowerZoo LLM日志记录器
        if self.enable_powerzoo_logging and POWERZOO_LOGGER_AVAILABLE:
            try:
                powerzoo_log_dir = os.path.join(self.log_dir, "powerzoo_logs")
                os.makedirs(powerzoo_log_dir, exist_ok=True)
                self.powerzoo_logger = PowerZooLLMLogger(
                    log_dir=powerzoo_log_dir,
                    experiment_name=f"{self.algorithm_name}_training"
                )
                if self.verbose > 0:
                    print(f"PowerZoo LLM logger initialized at: {powerzoo_log_dir}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to initialize PowerZoo LLM logger: {e}")
                self.enable_powerzoo_logging = False
        else:
            self.enable_powerzoo_logging = False
            if self.verbose > 0 and not POWERZOO_LOGGER_AVAILABLE:
                print("PowerZoo LLM logger module not available")
    
    def _on_step(self) -> bool:
        """
        每步调用的函数
        """
        # 记录基础训练指标
        if self.num_timesteps % self.log_freq == 0:
            self._log_training_metrics()
        
        # 保存模型检查点
        if self.save_freq > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_checkpoint()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        回合结束时调用
        """
        # 记录回合统计
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_info = info['episode']
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
                    self.episode_count += 1
                    
                    # 记录到TensorBoard
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar(
                            'episode/reward', episode_info['r'], self.num_timesteps
                        )
                        self.tensorboard_writer.add_scalar(
                            'episode/length', episode_info['l'], self.num_timesteps
                        )
                    
                    # 记录环境特定信息
                    self._log_environment_info(info)
        
        # 记录算法特定指标
        self._log_algorithm_metrics()
    
    def _log_training_metrics(self) -> None:
        """
        记录训练指标
        """
        if not self.tensorboard_writer:
            return
        
        # 基础统计
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])  # 最近100个回合
            std_reward = np.std(self.episode_rewards[-100:])
            
            self.tensorboard_writer.add_scalar('train/mean_reward', mean_reward, self.num_timesteps)
            self.tensorboard_writer.add_scalar('train/std_reward', std_reward, self.num_timesteps)
            
            # 更新最佳奖励
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_best_model()
        
        if self.episode_lengths:
            mean_length = np.mean(self.episode_lengths[-100:])
            self.tensorboard_writer.add_scalar('train/mean_episode_length', mean_length, self.num_timesteps)
        
        # 记录总回合数
        self.tensorboard_writer.add_scalar('train/episode_count', self.episode_count, self.num_timesteps)
        
        # 记录学习进度
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                lr = self.model.learning_rate(1.0)  # 对于调度器
            else:
                lr = self.model.learning_rate
            self.tensorboard_writer.add_scalar('train/learning_rate', lr, self.num_timesteps)
    
    def _log_algorithm_metrics(self) -> None:
        """
        记录算法特定指标
        """
        if not self.tensorboard_writer:
            return
        
        # 从模型获取训练指标
        if hasattr(self.model, '_last_obs') and hasattr(self.model, 'logger'):
            # 获取最新的日志记录
            logger_dict = self.model.logger.name_to_value
            
            # PPO特定指标
            if self.algorithm_name.lower() == 'ppo':
                metrics_to_log = [
                    'train/policy_gradient_loss',
                    'train/value_loss', 
                    'train/entropy_loss',
                    'train/approx_kl',
                    'train/clip_fraction',
                    'train/explained_variance'
                ]
                
                for metric in metrics_to_log:
                    if metric in logger_dict:
                        clean_name = metric.replace('train/', '')
                        self.tensorboard_writer.add_scalar(
                            f'algorithm/{clean_name}', 
                            logger_dict[metric], 
                            self.num_timesteps
                        )
            
            # SAC特定指标
            elif self.algorithm_name.lower() == 'sac':
                sac_metrics = [
                    'train/actor_loss',
                    'train/critic_loss',
                    'train/ent_coef_loss',
                    'train/entropy'
                ]
                
                for metric in sac_metrics:
                    if metric in logger_dict:
                        clean_name = metric.replace('train/', '')
                        self.tensorboard_writer.add_scalar(
                            f'algorithm/{clean_name}',
                            logger_dict[metric],
                            self.num_timesteps
                        )
    
    def _log_environment_info(self, info: Dict[str, Any]) -> None:
        """
        记录环境特定信息
        """
        if not self.tensorboard_writer:
            return
        
        # 记录PowerZoo LLM环境特定指标
        if 'powerzoo_metrics' in info:
            metrics = info['powerzoo_metrics']
            
            # 功率相关指标
            if 'power_loss' in metrics:
                self.tensorboard_writer.add_scalar(
                    'environment/power_loss', metrics['power_loss'], self.num_timesteps
                )
            
            if 'total_power' in metrics:
                self.tensorboard_writer.add_scalar(
                    'environment/total_power', metrics['total_power'], self.num_timesteps
                )
            
            # 电压相关指标
            if 'voltage_violation' in metrics:
                self.tensorboard_writer.add_scalar(
                    'environment/voltage_violation', metrics['voltage_violation'], self.num_timesteps
                )
            
            # PV利用率
            if 'pv_utilization' in metrics:
                self.tensorboard_writer.add_scalar(
                    'environment/pv_utilization', metrics['pv_utilization'], self.num_timesteps
                )
            
            # 控制成本
            if 'control_cost' in metrics:
                self.tensorboard_writer.add_scalar(
                    'environment/control_cost', metrics['control_cost'], self.num_timesteps
                )
        
        # 记录到PowerZoo LLM日志记录器
        if self.enable_powerzoo_logging and self.powerzoo_logger:
            try:
                # 构造日志数据
                log_data = {
                    'timestep': self.num_timesteps,
                    'episode_count': self.episode_count,
                    **info.get('powerzoo_metrics', {})
                }
                self.powerzoo_logger.log_step(log_data)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to log to PowerZoo logger: {e}")
        
        # 记录到系统日志记录器
        if self.enable_system_logging and self.system_logger and 'system_state' in info:
            try:
                system_state = info['system_state']
                if isinstance(system_state, dict):
                    # 转换为SystemState对象
                    state = SystemState(
                        timestamp=datetime.now(),
                        **system_state
                    )
                    self.system_logger.log_state(state)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to log to system logger: {e}")
    
    def _save_checkpoint(self) -> None:
        """
        保存模型检查点
        """
        try:
            checkpoint_path = os.path.join(
                self.model_save_path, 
                f"checkpoint_{self.num_timesteps}_{self.algorithm_name}"
            )
            self.model.save(checkpoint_path)
            
            if self.verbose > 0:
                print(f"Checkpoint saved at step {self.num_timesteps}: {checkpoint_path}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to save checkpoint: {e}")
    
    def _save_best_model(self) -> None:
        """
        保存最佳模型
        """
        try:
            best_model_path = os.path.join(
                self.model_save_path,
                f"best_model_{self.algorithm_name}"
            )
            self.model.save(best_model_path)
            
            if self.verbose > 0:
                print(f"New best model saved with reward {self.best_mean_reward:.2f}: {best_model_path}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to save best model: {e}")
    
    def _on_training_end(self) -> None:
        """
        训练结束时调用
        """
        # 保存最终模型
        try:
            final_model_path = os.path.join(
                self.model_save_path,
                f"final_model_{self.algorithm_name}"
            )
            self.model.save(final_model_path)
            
            if self.verbose > 0:
                print(f"Final model saved: {final_model_path}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Failed to save final model: {e}")
        
        # 关闭日志记录器
        if self.system_logger:
            try:
                self.system_logger.close()
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to close system logger: {e}")
        
        if self.powerzoo_logger:
            try:
                self.powerzoo_logger.close()
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to close PowerZoo logger: {e}")
        
        # 刷新TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.flush()
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to flush TensorBoard writer: {e}")
        
        if self.verbose > 0:
            print(f"Training completed. Best mean reward: {self.best_mean_reward:.2f}")
            print(f"Total episodes: {self.episode_count}")
            print(f"Total timesteps: {self.num_timesteps}")