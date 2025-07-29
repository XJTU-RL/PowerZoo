"""
增强版工具模块，提供完整的日志功能支持训练过程监控
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional


# 自定义日志级别用于训练监控
TRAIN_INFO = 25  # 介于INFO(20)和WARNING(30)之间
REWARD_DEBUG = 15  # 介于DEBUG(10)和INFO(20)之间
ACTION_DEBUG = 12  # 专门用于动作日志

logging.addLevelName(TRAIN_INFO, "TRAIN")
logging.addLevelName(REWARD_DEBUG, "REWARD") 
logging.addLevelName(ACTION_DEBUG, "ACTION")


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO):
    """
    获取增强版日志记录器，支持训练过程详细监控
    
    Args:
        name: 日志记录器名称
        log_file: 可选的日志文件路径
        level: 日志级别，默认INFO
        
    Returns:
        配置完成的日志记录器
    """
    # 创建logger
    logger = logging.getLogger(name)
    
    # 如果logger已经有处理器，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(level)
    
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
    
    return logger


def setup_training_logger(name: str, log_dir: str = "logs") -> logging.Logger:
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
    
    return get_logger(name, log_file, level=logging.DEBUG)


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
    logger.train_info(
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
    logger.reward_debug(f"Reward Components: {reward_str}")


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
    logger.action_debug(
        f"{device_type} | {device_name} | {old_state} -> {new_state} | Diff: {diff:.3f}"
    )