# -*- coding: utf-8 -*-
"""
@File      : __init__.py
@Time      : 2025-04-08 17:47
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该文件的主要作用是创建一个算法注册表，用于管理和存储多种算法类。
- 关键组件 `ALGO_REGISTRY`：是一个字典，作为算法注册表。
- 职责：将算法的名称字符串（如 "happo"）映射到对应的算法类（如 HAPPO）。
- 依赖模块：从 `algorithms.actors` 导入了多个算法类，包括 HAPPO、HATRPO 等。
通过此注册表，可以方便地根据算法名称获取对应的算法类，便于后续的算法调用和使用。
"""
"""Algorithm registry."""
from algorithms.actors.happo import HAPPO
from algorithms.actors.dan_happo import DAN_HAPPO
from algorithms.actors.hatrpo import HATRPO
from algorithms.actors.haa2c import HAA2C
from algorithms.actors.haddpg import HADDPG
from algorithms.actors.hatd3 import HATD3
from algorithms.actors.hasac import HASAC
from algorithms.actors.had3qn import HAD3QN
from algorithms.actors.maddpg import MADDPG
from algorithms.actors.matd3 import MATD3
from algorithms.actors.mappo import MAPPO
from algorithms.actors.shom import SHOM
from algorithms.actors.sn_mappo import SN_MAPPO

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "hasac": HASAC,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "matd3": MATD3,
    "mappo": MAPPO,
    "shom": SHOM,
    "sn_mappo": SN_MAPPO,
}
