"""
PowerZoo - 电力系统多智能体强化学习环境

该模块提供了基于OpenDSS的电力系统仿真环境，支持多智能体强化学习训练。
"""

# 核心环境类
from .env import Env, ActionSpace
from .powerzoo_env import PowerZooEnv, OptimizedPowerZooEnv

# 电路和负载管理
from .circuit_system import Circuits
from .loadprofile import LoadProfile

# 环境注册和工具
from .env_register import make_base_env, remove_parallel_dss
from .env_wrapper import PowerZooEnvWrapper

# 配置类
from .powerzoo_config import (
    OpenDSSScenario,
    OpenDSSConstraints,
    OpenDSSMetrics,
    PowerZooEnvConfig,
    OpenDSSExpertRules,
    OpenDSSStateAnalyzer,
    PowerZooActionSelector,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "Xiaodong Zheng"
__email__ = "zxd_xjtu@stu.xjtu.edu.cn"

# 导出的公共API
__all__ = [
    # 核心环境
    "Env",
    "PowerZooEnv", 
    "OptimizedPowerZooEnv",
    
    # 辅助类
    "ActionSpace",
    "Circuits",
    "LoadProfile",
    
    # 工具函数
    "make_base_env",
    "remove_parallel_dss",
    "PowerZooEnvWrapper",
    
    # 配置类
    "OpenDSSScenario",
    "OpenDSSConstraints", 
    "OpenDSSMetrics",
    "PowerZooEnvConfig",
    "PowerZooEnvConfig",
    "OpenDSSExpertRules",
    "OpenDSSStateAnalyzer",
    "PowerZooActionSelector",
    "PowerZooActionSelector",
]

# 环境配置常量
SUPPORTED_SYSTEMS = ["13Bus", "34Bus", "123Bus", "8500Node"]

# 快速创建环境的便捷函数
def create_env(env_name="13Bus", **kwargs):
    """
    快速创建PowerZoo环境
    
    Args:
        env_name: 环境名称，支持 13Bus, 34Bus, 123Bus, 8500Node
        **kwargs: 其他环境参数
        
    Returns:
        Env: 环境实例
    """
    from .env_wrapper import PowerZooEnvWrapper
    from argparse import Namespace
    
    config = Namespace(
        env_name=env_name,
        **kwargs
    )
    
    return PowerZooEnvWrapper.create_base_env(config, env_name=env_name)


def create_marl_env(env_name="13Bus", num_env=1, use_s=False, seed=0, **kwargs):
    """
    快速创建多智能体强化学习环境
    
    Args:
        env_name: 环境名称
        num_env: 环境数量（用于并行训练）
        use_s: 是否使用敏感性矩阵
        seed: 随机种子
        **kwargs: 其他参数
        
    Returns:
        PowerZooEnv: 多智能体环境实例
    """
    from argparse import Namespace
    
    # 创建基础环境
    base_env = make_base_env(env_name)
    
    # 创建配置
    config = Namespace(
        env_name=env_name,
        num_env=num_env,
        useS=use_s,
        seed=seed,
        **kwargs
    )
    
    return PowerZooEnv(env=base_env, config=config)


# 环境信息查询
def get_env_info(env_name):
    """
    获取环境信息
    
    Args:
        env_name: 环境名称
        
    Returns:
        dict: 环境配置信息
    """
    from .env_register import _ENV_INFO
    
    if env_name not in _ENV_INFO:
        raise ValueError(f"不支持的环境: {env_name}. 支持的环境: {list(ENV_LIST.keys())}")
    
    return _ENV_INFO[env_name]


# 日志配置
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())