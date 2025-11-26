# -*- coding: utf-8 -*-
"""
基础日志器模块 - 提供统一的日志功能
不依赖于utils.py，避免循环依赖
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional

# 自定义日志级别
TRAIN_INFO = 25  # 介于INFO(20)和WARNING(30)之间
REWARD_DEBUG = 15  # 介于DEBUG(10)和INFO(20)之间
ACTION_DEBUG = 12  # 专门用于动作日志

logging.addLevelName(TRAIN_INFO, "TRAIN")
logging.addLevelName(REWARD_DEBUG, "REWARD")
logging.addLevelName(ACTION_DEBUG, "ACTION")


class UnifiedLogger:
    """统一的日志器类"""
    
    _instances = {}  # 存储已创建的日志器实例
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        """
        获取统一的日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 可选的日志文件路径
            level: 日志级别，默认INFO
            
        Returns:
            配置完成的日志记录器
        """
        # 如果已存在该名称的logger，直接返回
        if name in cls._instances:
            return cls._instances[name]
        
        # 创建logger
        logger = logging.getLogger(name)
        
        # 如果logger已经有处理器，先清除
        if logger.handlers:
            logger.handlers.clear()
        
        # 设置日志级别
        logger.setLevel(level)
        logger.propagate = False  # 不传播到父logger
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 创建详细格式化器
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(detailed_formatter)
        
        # 添加控制台处理器
        logger.addHandler(console_handler)
        
        # 如果指定了日志文件，添加文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的信息
            
            # 文件使用更详细的格式
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S.%f'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # 添加自定义日志方法
        def train_info(message, *args, **kwargs):
            if logger.isEnabledFor(TRAIN_INFO):
                logger._log(TRAIN_INFO, message, args, **kwargs)
        
        def reward_debug(message, *args, **kwargs):
            if logger.isEnabledFor(REWARD_DEBUG):
                logger._log(REWARD_DEBUG, message, args, **kwargs)
                
        def action_debug(message, *args, **kwargs):
            if logger.isEnabledFor(ACTION_DEBUG):
                logger._log(ACTION_DEBUG, message, args, **kwargs)
        
        # 绑定自定义方法到logger实例
        logger.train_info = train_info
        logger.reward_debug = reward_debug
        logger.action_debug = action_debug
        
        # 保存实例
        cls._instances[name] = logger
        
        return logger
    
    @classmethod
    def setup_training_logger(cls, name: str, log_dir: str = "logs") -> logging.Logger:
        """
        设置专门用于训练监控的日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            
        Returns:
            配置完成的训练日志记录器
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{name}_{timestamp}.log")
        
        return cls.get_logger(name, log_file, level=logging.DEBUG)
    
    @classmethod
    def close_all_loggers(cls):
        """关闭所有日志器的处理器"""
        for logger in cls._instances.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._instances.clear()


# 便捷函数
def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO):
    """获取统一的日志记录器"""
    return UnifiedLogger.get_logger(name, log_file, level)


def setup_training_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """设置训练日志记录器"""
    return UnifiedLogger.setup_training_logger(name, log_dir)


def log_training_step(logger: logging.Logger, step: int, episode: int, 
                     action: str, reward: float, done: bool, info: dict):
    """
    记录训练步骤的详细信息
    
    Args:
        logger: 日志记录器
        step: 步骤编号
        episode: 回合编号
        action: 动作描述
        reward: 奖励值
        done: 是否结束
        info: 附加信息
    """
    if hasattr(logger, 'train_info'):
        logger.train_info(
            f"Episode {episode:4d} | Step {step:3d} | Action: {action} | "
            f"Reward: {reward:8.4f} | Done: {done} | Info: {info}"
        )
    else:
        logger.info(
            f"Episode {episode:4d} | Step {step:3d} | Action: {action} | "
            f"Reward: {reward:8.4f} | Done: {done} | Info: {info}"
        )


def log_reward_components(logger: logging.Logger, components: dict):
    """
    记录奖励函数各组成部分的详细信息
    
    Args:
        logger: 日志记录器
        components: 奖励组成部分字典
    """
    reward_str = " | ".join([f"{k}: {v:6.3f}" for k, v in components.items()])
    if hasattr(logger, 'reward_debug'):
        logger.reward_debug(f"Reward Components: {reward_str}")
    else:
        logger.debug(f"Reward Components: {reward_str}")


def log_device_actions(logger: logging.Logger, device_type: str, 
                      device_name: str, old_state, new_state, diff: float):
    """
    记录设备动作执行的详细信息
    
    Args:
        logger: 日志记录器
        device_type: 设备类型
        device_name: 设备名称
        old_state: 旧状态
        new_state: 新状态
        diff: 状态差异
    """
    if hasattr(logger, 'action_debug'):
        logger.action_debug(
            f"{device_type} | {device_name} | {old_state} -> {new_state} | Diff: {diff:.3f}"
        )
    else:
        logger.debug(
            f"{device_type} | {device_name} | {old_state} -> {new_state} | Diff: {diff:.3f}"
        )


def log_training_summary(logger: logging.Logger, episode: int, total_reward: float, 
                        episode_length: int, final_info: dict):
    """
    记录训练回合总结信息
    
    Args:
        logger: 日志记录器
        episode: 回合编号
        total_reward: 总奖励
        episode_length: 回合长度
        final_info: 最终信息字典
    """
    avg_reward = total_reward / max(episode_length, 1)
    
    summary_items = [
        f"Episode {episode:4d} 完成",
        f"总奖励: {total_reward:8.4f}",
        f"平均奖励: {avg_reward:6.4f}",
        f"回合长度: {episode_length:3d}"
    ]
    
    # 添加关键性能指标
    if 'power_loss_ratio' in final_info:
        summary_items.append(f"功率损耗: {final_info['power_loss_ratio']:.4f}")
    if 'voltage_violations' in final_info:
        summary_items.append(f"电压违规: {final_info['voltage_violations']}")
    if 'voltage_compliance_rate' in final_info:
        summary_items.append(f"电压合格率: {final_info['voltage_compliance_rate']:.3f}")
    
    if hasattr(logger, 'train_info'):
        logger.train_info(" | ".join(summary_items))
    else:
        logger.info(" | ".join(summary_items))


def create_training_debug_logger(env_name: str = "powerzoo") -> logging.Logger:
    """
    创建专门用于训练调试的日志记录器
    
    Args:
        env_name: 环境名称
        
    Returns:
        调试日志记录器
    """
    debug_logger = get_logger(f"{env_name}_debug", level=logging.DEBUG)
    
    # 添加特殊的调试方法
    def system_state(state_dict: dict):
        """记录系统状态"""
        state_str = " | ".join([f"{k}: {v}" for k, v in state_dict.items()])
        debug_logger.debug(f"系统状态: {state_str}")
    
    def convergence_check(converged: bool, iterations: int = None):
        """记录收敛性检查"""
        status = "收敛" if converged else "未收敛"
        iter_info = f" ({iterations}次迭代)" if iterations else ""
        debug_logger.debug(f"求解状态: {status}{iter_info}")
    
    # 绑定调试方法
    debug_logger.system_state = system_state
    debug_logger.convergence_check = convergence_check
    
    return debug_logger