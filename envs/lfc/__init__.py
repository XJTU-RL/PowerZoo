# -*- coding: utf-8 -*-
"""LFC（负荷频率控制）任务家族：OpenDSS 配电网 + Simulink 9 机输电网联合仿真。

来源：OHASAC 论文实现（papers/ohasac/code，P0 修复版），按 PowerZoo
环境契约适配（见 lfc_env.py 的忠实性说明）。
依赖：OpenDSS（pip 包 dss）；MATLAB/Simulink（R2024a~R2025b，Simscape
Electrical 含 Specialized Power Systems）——MATLAB 缺席时本包可正常
import（配置与数据校验可用），环境构造会给出明确报错。
"""


def __getattr__(name):
    if name == "LFCEnv":
        from envs.lfc.lfc_env import LFCEnv

        return LFCEnv
    if name == "LFCConfig":
        from envs.lfc.lfc_config import LFCConfig

        return LFCConfig
    if name == "LFCLogger":
        from envs.lfc.lfc_logger import LFCLogger

        return LFCLogger
    raise AttributeError(name)
