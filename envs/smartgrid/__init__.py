"""公共 API 按需导入，配置预览不加载训练或 OpenDSS 运行时。"""

from importlib import import_module

_EXPORTS = {
    "Env": (".base_env.runtime_env", "Env"),
    "VVCEnv": (".base_env.vvc_env", "VVCEnv"),
    "OptimizedVVCEnv": (".base_env.vvc_env", "OptimizedVVCEnv"),
    "DSSSession": ("envs.powergrid.session", "DSSSession"),
    "SmartGridConfig": (".base_env.env_config", "SmartGridConfig"),
    "ScenarioDataset": (".data_process.dataset", "ScenarioDataset"),
    "make_base_env": (".base_env.env_register", "make_base_env"),
    "remove_parallel_dss": (".base_env.env_register", "remove_parallel_dss"),
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

SUPPORTED_SYSTEMS = ["13Bus", "34Bus", "123Bus", "8500Node"]


def get_env_info(env_name):
    """旧查询接口复用正式解析器，不读取另一份默认表。"""
    from utils.config_pipeline import ConfigLoader

    system = {"34Bus_pv": "34Bus_PV"}.get(env_name, env_name)
    values = {"env": {"pv_plan": "none"}} if system != "34Bus_PV" else None
    return (
        ConfigLoader()
        .load("happo", "smartgrid", system_ref=system, values=values)
        .env_args
    )
