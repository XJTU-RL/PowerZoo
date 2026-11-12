"""Algorithm registry.

单一事实源在 ``algorithms/actors/__init__.py``（运行时 runner 消费该表）；
本表是其投影 + 无独立执行器的注册项（qmix/dan_happo/2ts_vvc 由各自
runner 自建 actor，仅在此登记供顶层枚举）。新增算法只需改 actors 表，
此处自动同步，避免两份手工字典漂移（曾出现 qmix 等三键只在一侧的存量债务）。
"""
from algorithms.actors import ALGO_REGISTRY as _ACTOR_REGISTRY
from algorithms.actors.m_Qmix import M_QMix as QMIX
from algorithms.actors.dan_happo import DAN_HAPPO
from algorithms.twots_vvc import TwoTSVVC

ALGO_REGISTRY = {
    **_ACTOR_REGISTRY,
    # 以下算法无独立执行器类（runner 自建或研究入口专用），仅登记枚举
    "qmix": QMIX,
    "dan_happo": DAN_HAPPO,
    "2ts_vvc": TwoTSVVC,
}
