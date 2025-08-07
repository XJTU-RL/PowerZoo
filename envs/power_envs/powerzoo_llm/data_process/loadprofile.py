# -*- coding: utf-8 -*-
"""
LoadProfile模块 - 向后兼容导入
将精简重构的模块导出为统一接口
"""

# 导入核心类和常量
from envs.power_envs.powerzoo_llm.data_process.loadprofile_core import LoadProfile
from envs.power_envs.powerzoo_llm.data_process.loadprofile_dss_parser import Constants, DSSFileParser
from envs.power_envs.powerzoo_llm.data_process.loadprofile_episode import EpisodeGenerator
from envs.power_envs.powerzoo_llm.data_process.loadprofile_config import ConfigGenerator

# 保持向后兼容
__all__ = [
    'LoadProfile',
    'Constants', 
    'DSSFileParser',
    'EpisodeGenerator',
    'ConfigGenerator'
]

# 版本信息
__version__ = "2.0.0"
__description__ = "精简重构版LoadProfile - 更好的多进程支持和模块化设计"