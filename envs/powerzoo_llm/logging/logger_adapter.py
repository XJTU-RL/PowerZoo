# -*- coding: utf-8 -*-
"""
PowerZoo LLM环境日志适配器 - 连接SystemLogger和base_logger
整合powerzoo_llm专用的系统日志记录与通用训练日志记录
"""

import os
import time
from typing import Dict, Any, Optional
from threading import Lock

# 使用统一日志系统
from envs.powerzoo_llm.logging.system_logger import get_system_logger, close_system_logger
from envs.powerzoo_llm.logging.base_logger import get_logger
SYSTEM_LOGGER_AVAILABLE = True

logger = get_logger(__name__)

class PowerZooLoggerAdapter:
    """
    PowerZoo LLM环境日志适配器
    
    功能:
    1. 集成SystemLogger用于详细的电力系统参数记录
    2. 与base_logger协作提供统一的日志接口
    3. 提供训练进度的增强记录
    4. 线程安全的日志操作
    """
    
    def __init__(self, 
                 run_dir: str, 
                 enable_system_logging: bool = True,
                 log_interval: int = 10,  # 减少日志间隔
                 buffer_size: int = 1000):
        """
        初始化日志适配器
        
        Args:
            run_dir: 运行目录路径
            enable_system_logging: 是否启用SystemLogger
            log_interval: 系统日志记录间隔(steps)
            buffer_size: 缓冲区大小
        """
        self.run_dir = run_dir
        self.enable_system_logging = enable_system_logging and SYSTEM_LOGGER_AVAILABLE
        self.log_interval = log_interval
        self.buffer_size = buffer_size
        
        # 线程安全锁
        self._lock = Lock()
        
        # 计数器
        self.step_count = 0
        self.episode_count = 0
        self.last_log_time = time.time()
        
        # 性能统计
        self.performance_stats = {
            'log_operations': 0,
            'total_log_time': 0.0,
            'avg_log_time': 0.0
        }
        
        # 初始化SystemLogger（如果可用）
        self.system_logger = None
        if self.enable_system_logging:
            try:
                # 使用统一的system_params目录
                system_log_dir = os.path.join(run_dir, "system_params")
                os.makedirs(system_log_dir, exist_ok=True)
                
                self.system_logger = get_system_logger(
                    log_dir=system_log_dir,
                    buffer_size=buffer_size,
                    save_interval=log_interval,
                    enable_realtime_log=True,
                    compression_level=6
                )
                
                logger.info(f"PowerZooLoggerAdapter initialized with SystemLogger at {system_log_dir}")
                
            except Exception as e:
                logger.error(f"Failed to initialize SystemLogger: {e}")
                self.enable_system_logging = False
                self.system_logger = None
        else:
            logger.info("PowerZooLoggerAdapter initialized without SystemLogger")
    
    def log_step(self, 
                 env, 
                 actions: Any, 
                 reward: float, 
                 info: Dict[str, Any], 
                 computation_time: float = 0.0):
        """
        记录环境步进信息
        
        Args:
            env: 环境实例
            actions: 执行的动作
            reward: 获得的奖励
            info: 额外信息字典
            computation_time: 计算耗时
        """
        start_time = time.time()
        
        with self._lock:
            try:
                self.step_count += 1
                
                # 使用SystemLogger记录详细系统信息（如果启用）
                if self.enable_system_logging and self.system_logger:
                    self.system_logger.log_system_state(
                        env=env,
                        actions=actions,
                        reward=reward,
                        info=info,
                        computation_time=computation_time
                    )
                
                # 更新性能统计
                log_time = time.time() - start_time
                self.performance_stats['log_operations'] += 1
                self.performance_stats['total_log_time'] += log_time
                self.performance_stats['avg_log_time'] = (
                    self.performance_stats['total_log_time'] / 
                    self.performance_stats['log_operations']
                )
                
                # 性能警告
                if log_time > 0.01:  # 超过10ms
                    logger.debug(f"Slow logging operation: {log_time:.4f}s")
                    
            except Exception as e:
                logger.error(f"Failed to log step: {e}")
    
    def log_episode_start(self, episode: int):
        """记录回合开始"""
        with self._lock:
            self.episode_count = episode
            logger.debug(f"Episode {episode} started")
    
    def log_episode_end(self, episode: int, episode_reward: float, episode_length: int):
        """记录回合结束"""
        with self._lock:
            try:
                # 获取系统Logger的回合汇总（如果可用）
                if self.enable_system_logging and self.system_logger:
                    episode_summary = self.system_logger.get_episode_summary(episode)
                    
                    # 记录回合汇总信息
                    if episode_summary:
                        logger.info(
                            f"Episode {episode} summary: "
                            f"Steps: {episode_summary.get('total_steps', episode_length)}, "
                            f"Reward: {episode_reward:.4f}, "
                            f"Avg Reward: {episode_summary.get('avg_reward', 0):.4f}, "
                            f"Voltage Violations: {episode_summary.get('total_voltage_violations', 0)}, "
                            f"Convergence Rate: {episode_summary.get('convergence_rate', 1.0):.3f}"
                        )
                
                logger.debug(f"Episode {episode} ended with reward: {episode_reward:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to log episode end: {e}")
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """获取实时监控指标"""
        with self._lock:
            try:
                base_metrics = {
                    'step_count': self.step_count,
                    'episode_count': self.episode_count,
                    'last_log_time': self.last_log_time,
                    'performance': self.performance_stats.copy()
                }
                
                # 添加SystemLogger指标（如果可用）
                if self.enable_system_logging and self.system_logger:
                    system_metrics = self.system_logger.get_realtime_metrics()
                    base_metrics.update({
                        'system_logger': system_metrics
                    })
                
                return base_metrics
                
            except Exception as e:
                logger.error(f"Failed to get realtime metrics: {e}")
                return {}
    
    def export_system_data(self, output_path: str, format: str = 'csv'):
        """导出系统数据"""
        if self.enable_system_logging and self.system_logger:
            try:
                self.system_logger.export_data(output_path, format)
                logger.info(f"System data exported to {output_path}")
            except Exception as e:
                logger.error(f"Failed to export system data: {e}")
        else:
            logger.warning("SystemLogger not available for data export")
    
    def close(self):
        """关闭日志适配器并清理资源"""
        with self._lock:
            try:
                if self.enable_system_logging and self.system_logger:
                    close_system_logger()
                    logger.info("SystemLogger closed")
                
                # 输出最终统计
                logger.info(
                    f"PowerZooLoggerAdapter closed - "
                    f"Total steps: {self.step_count}, "
                    f"Total episodes: {self.episode_count}, "
                    f"Avg log time: {self.performance_stats['avg_log_time']:.6f}s"
                )
                
            except Exception as e:
                logger.error(f"Error closing PowerZooLoggerAdapter: {e}")


# 全局适配器实例
_global_adapter = None

def get_powerzoo_logger_adapter(**kwargs) -> PowerZooLoggerAdapter:
    """获取全局日志适配器实例"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = PowerZooLoggerAdapter(**kwargs)
    return _global_adapter

def close_powerzoo_logger_adapter():
    """关闭全局日志适配器"""
    global _global_adapter
    if _global_adapter is not None:
        _global_adapter.close()
        _global_adapter = None