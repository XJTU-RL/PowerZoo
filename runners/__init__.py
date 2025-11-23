# -*- coding: utf-8 -*-
"""
@File      : __init__.py
@Time      : 2025-04-08 17:41
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此文件的主要作用是创建一个运行器注册表。
关键组件为 `RUNNER_REGISTRY` 字典，其职责是将不同的算法名称映射到对应的运行器类。
依赖的关键模块有 `OnPolicyHARunner`、`OnPolicyMARunner`、`OffPolicyHARunner`、
`OffPolicyMARunner` 和 `QMIXRunner`，分别来自 `runners` 包下的对应模块。
通过该注册表，可根据算法名称方便地获取对应的运行器类。
"""
"""Runner registry."""
from runners.on_policy_ha_runner import OnPolicyHARunner
from runners.on_policy_ma_runner import OnPolicyMARunner
from runners.off_policy_ha_runner import OffPolicyHARunner
from runners.off_policy_ma_runner import OffPolicyMARunner
from runners.Qmix_runner import QMIXRunner
from runners.two_ts_runner import TwoTSRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "qmix": QMIXRunner,
    "shom": OnPolicyHARunner,
    "sn_mappo": OnPolicyHARunner,  # Stackelberg-Nash MAPPO uses heterogeneous runner
    "dan_happo": OnPolicyHARunner,  # DAN-HAPPO uses heterogeneous runner
    "2ts_vvc": TwoTSRunner,
}
