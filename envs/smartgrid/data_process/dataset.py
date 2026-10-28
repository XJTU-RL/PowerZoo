"""历史导入路径映射到公共场景模块，不另行实现加载或生成。"""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("envs.powergrid.scenario")
