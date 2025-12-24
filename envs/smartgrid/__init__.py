"""
PowerZoo - 电力系统多智能体强化学习环境

该模块提供了基于OpenDSS的电力系统仿真环境，支持多智能体强化学习训练。
"""

# 核心环境类（从本地core_env导入，不再依赖legacy powerzoo）
from .base_env.core_env import Env, ActionSpace
from .base_env.powerzoo_env import PowerZooEnv, OptimizedPowerZooEnv

# 电路和负载管理
from .circuit_system import Circuits
from .data_process.loadprofile import LoadProfile

# 环境注册和工具
from .base_env.env_register import make_base_env, remove_parallel_dss


# 配置类
from .base_env.powerzoo_config import (
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

    
    # 配置类
    "OpenDSSScenario",
    "OpenDSSConstraints",
    "OpenDSSMetrics",
    "PowerZooEnvConfig",
    "OpenDSSExpertRules",
    "OpenDSSStateAnalyzer",
    "PowerZooActionSelector",
]

# 环境配置常量
SUPPORTED_SYSTEMS = ["13Bus", "34Bus", "123Bus", "8500Node"]


# 环境信息查询
def get_env_info(env_name):
    """
    获取环境信息
    
    Args:
        env_name: 环境名称
        
    Returns:
        dict: 环境配置信息
    """
    from .base_env.env_register import _ENV_INFO
    
    if env_name not in _ENV_INFO:
        raise ValueError(f"不支持的环境: {env_name}. 支持的环境: {list(_ENV_INFO.keys())}")
    
    return _ENV_INFO[env_name]


# 日志配置
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())