# -*- coding: utf-8 -*-
"""
单智能体PowerZoo环境日志记录器

本模块提供了专门针对单智能体训练的日志记录功能，包括:
- 训练过程日志
- 环境状态记录
- 性能指标统计
- 可视化数据导出
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path


class SingleAgentLogger:
    """单智能体训练日志记录器"""
    
    def __init__(self, 
                 log_dir: str = "./logs/single_agent",
                 experiment_name: str = None,
                 log_level: str = "INFO",
                 save_episode_data: bool = True,
                 save_metrics: bool = True):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            log_level: 日志级别
            save_episode_data: 是否保存回合数据
            save_metrics: 是否保存性能指标
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_episode_data = save_episode_data
        self.save_metrics = save_metrics
        
        # 创建日志目录
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self._setup_logger(log_level)
        
        # 初始化数据存储
        self.episode_data = []
        self.step_data = []
        self.metrics_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'voltage_violations': [],
            'power_losses': [],
            'convergence_times': []
        }
        
        # 当前回合信息
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = None
        
        self.logger.info(f"单智能体日志记录器初始化完成，实验名称: {self.experiment_name}")
    
    def _setup_logger(self, log_level: str):
        """设置日志记录器"""
        self.logger = logging.getLogger(f"SingleAgent_{self.experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(
            self.exp_dir / "training.log", 
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_episode_start(self, episode: int, config: Dict[str, Any] = None):
        """记录回合开始"""
        self.current_episode = episode
        self.current_step = 0
        self.episode_start_time = datetime.now()
        
        self.logger.info(f"开始第 {episode} 回合")
        
        if config and episode == 0:
            # 保存配置信息
            config_file = self.exp_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"配置信息已保存到 {config_file}")
    
    def log_step(self, 
                 step: int,
                 action: Union[int, np.ndarray],
                 observation: np.ndarray,
                 reward: float,
                 done: bool,
                 info: Dict[str, Any] = None):
        """记录单步信息"""
        self.current_step = step
        
        step_info = {
            'episode': self.current_episode,
            'step': step,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'reward': float(reward),
            'done': done,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加额外信息
        if info:
            step_info.update(info)
        
        if self.save_episode_data:
            self.step_data.append(step_info)
        
        # 记录关键信息
        if step % 6 == 0 or done:  # 每6步或结束时记录
            self.logger.debug(f"步骤 {step}: 动作={action}, 奖励={reward:.4f}, 完成={done}")
    
    def log_episode_end(self, 
                       total_reward: float,
                       episode_length: int,
                       final_info: Dict[str, Any] = None):
        """记录回合结束"""
        episode_time = (datetime.now() - self.episode_start_time).total_seconds()
        
        episode_info = {
            'episode': self.current_episode,
            'total_reward': float(total_reward),
            'episode_length': episode_length,
            'episode_time': episode_time,
            'avg_reward_per_step': float(total_reward / episode_length) if episode_length > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加最终信息
        if final_info:
            episode_info.update(final_info)
        
        if self.save_episode_data:
            self.episode_data.append(episode_info)
        
        # 更新性能指标
        self.metrics_data['episode_rewards'].append(total_reward)
        self.metrics_data['episode_lengths'].append(episode_length)
        
        if final_info:
            if 'voltage_violations' in final_info:
                self.metrics_data['voltage_violations'].append(final_info['voltage_violations'])
            if 'power_loss' in final_info:
                self.metrics_data['power_losses'].append(final_info['power_loss'])
        
        self.logger.info(
            f"回合 {self.current_episode} 结束: "
            f"总奖励={total_reward:.4f}, "
            f"步数={episode_length}, "
            f"用时={episode_time:.2f}s"
        )
    
    def log_training_metrics(self, metrics: Dict[str, float]):
        """记录训练指标"""
        if self.save_metrics:
            for key, value in metrics.items():
                if key not in self.metrics_data:
                    self.metrics_data[key] = []
                self.metrics_data[key].append(value)
        
        # 记录关键指标
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"训练指标: {metric_str}")
    
    def save_data(self):
        """保存所有数据到文件"""
        if self.save_episode_data and self.episode_data:
            # 保存回合数据
            episode_df = pd.DataFrame(self.episode_data)
            episode_file = self.exp_dir / "episodes.csv"
            episode_df.to_csv(episode_file, index=False, encoding='utf-8')
            self.logger.info(f"回合数据已保存到 {episode_file}")
            
            # 保存步骤数据
            if self.step_data:
                step_df = pd.DataFrame(self.step_data)
                step_file = self.exp_dir / "steps.csv"
                step_df.to_csv(step_file, index=False, encoding='utf-8')
                self.logger.info(f"步骤数据已保存到 {step_file}")
        
        if self.save_metrics and self.metrics_data:
            # 保存性能指标
            metrics_file = self.exp_dir / "metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"性能指标已保存到 {metrics_file}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取训练总结统计"""
        if not self.metrics_data['episode_rewards']:
            return {}
        
        rewards = np.array(self.metrics_data['episode_rewards'])
        lengths = np.array(self.metrics_data['episode_lengths'])
        
        stats = {
            'total_episodes': len(rewards),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'mean_episode_length': float(np.mean(lengths)),
            'total_steps': int(np.sum(lengths))
        }
        
        # 添加最近100回合的统计
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            stats['recent_mean_reward'] = float(np.mean(recent_rewards))
            stats['recent_std_reward'] = float(np.std(recent_rewards))
        
        return stats
    
    def log_summary(self):
        """记录训练总结"""
        stats = self.get_summary_stats()
        if stats:
            self.logger.info("=== 训练总结 ===")
            self.logger.info(f"总回合数: {stats['total_episodes']}")
            self.logger.info(f"平均奖励: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
            self.logger.info(f"最大奖励: {stats['max_reward']:.4f}")
            self.logger.info(f"最小奖励: {stats['min_reward']:.4f}")
            self.logger.info(f"平均回合长度: {stats['mean_episode_length']:.2f}")
            self.logger.info(f"总步数: {stats['total_steps']}")
            
            if 'recent_mean_reward' in stats:
                self.logger.info(f"最近100回合平均奖励: {stats['recent_mean_reward']:.4f}")
    
    def close(self):
        """关闭日志记录器"""
        self.save_data()
        self.log_summary()
        self.logger.info(f"日志记录器关闭，数据已保存到 {self.exp_dir}")
        
        # 关闭日志处理器
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()