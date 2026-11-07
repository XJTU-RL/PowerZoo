"""公共 API 按需导入，配置预览不加载训练或 OpenDSS 运行时。"""

from importlib import import_module

_EXPORTS = {
    "VVCEnv": (".vvc_env", "VVCEnv"),
    "VVCLogger": (".vvc_logger", "VVCLogger"),
    "Env": (".runtime_env", "Env"),
    "DSSSession": ("envs.powergrid.session", "DSSSession"),
    "make_base_env": (".vvc.env_register", "make_base_env"),
    "remove_parallel_dss": (".vvc.env_register", "remove_parallel_dss"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


__version__ = "1.0.0"

__author__ = "Xiaodong Zheng"

__email__ = "zxd_xjtu@stu.xjtu.edu.cn"
