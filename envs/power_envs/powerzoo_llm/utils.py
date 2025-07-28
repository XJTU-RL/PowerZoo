"""
简化版工具模块，提供日志功能
"""
import logging
import sys


def get_logger(name):
    """获取日志记录器"""
    # 创建logger
    logger = logging.getLogger(name)
    
    # 如果logger已经有处理器，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(console_handler)
    
    return logger