"""公共 API 按需导入，配置预览不加载训练或 OpenDSS 运行时。"""

from importlib import import_module

_EXPORTS = {
    "Env": ("envs.smartgrid.base_env.env", "Env"),
    "ActionSpace": ("envs.smartgrid.base_env.env", "ActionSpace"),
    "VVCEnv": ("envs.smartgrid.base_env.vvc_env", "VVCEnv"),
    "OptimizedVVCEnv": ("envs.smartgrid.base_env.vvc_env", "OptimizedVVCEnv"),
    "make_base_env": ("envs.smartgrid.base_env.env_register", "make_base_env"),
    "make_env": ("envs.smartgrid.base_env.env_register", "make_env"),
    "remove_parallel_dss": (
        "envs.smartgrid.base_env.env_register",
        "remove_parallel_dss",
    ),
    "SmartGridConfig": ("envs.smartgrid.base_env.env_config", "SmartGridConfig"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
