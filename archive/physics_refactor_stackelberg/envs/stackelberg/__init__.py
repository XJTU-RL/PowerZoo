"""公共 API 按需导入，配置预览不加载训练或 OpenDSS 运行时。"""

from importlib import import_module

_EXPORTS = {
    "StackelbergBaseEnv": (
        "envs.stackelberg.stackelberg_game.stackelberg_base_env",
        "StackelbergBaseEnv",
    ),
    "AsyncMultiAgentWrapper": (
        "envs.stackelberg.stackelberg_game.async_wrapper",
        "AsyncMultiAgentWrapper",
    ),
    "StackelbergMonitor": (
        "envs.stackelberg.stackelberg_game.stackelberg_monitor",
        "StackelbergMonitor",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
