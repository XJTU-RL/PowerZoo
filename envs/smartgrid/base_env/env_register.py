"""SmartGrid 工厂；模型只读，worker 文件由独立 context 生命周期管理。"""

from pathlib import Path
from envs.smartgrid.base_env.env import Env
from envs.smartgrid.base_env.env_config import SmartGridConfig


def get_data_root():
    return Path(__file__).resolve().parents[3]


def get_node_systems_path():
    return get_data_root() / "node_systems"


def make_env(env_name_or_config, dss_act=False, worker_idx=None, config_dict=None):
    if isinstance(env_name_or_config, SmartGridConfig):
        config = env_name_or_config
    else:
        from utils.config_pipeline import ConfigLoader
        from utils.config_validation import build_dataclass

        system = {"34Bus_pv": "34Bus_PV"}.get(env_name_or_config, env_name_or_config)
        values = (config_dict or {}).get("env_args", config_dict or {})
        values = dict(values)
        values["dss_act"] = dss_act
        if system != "34Bus_PV":
            values.setdefault("pv_plan", "none")
        loaded = ConfigLoader().load(
            "happo", "smartgrid", system_ref=system, values={"env": values}
        )
        config = build_dataclass(SmartGridConfig, loaded.env_args)
    if worker_idx is not None:
        config = config.with_worker_idx(worker_idx)
    return Env(str(get_data_root()), config)


make_base_env = make_env


def remove_parallel_dss(env_name, num_workers):
    """兼容旧调用；新内核没有写入模型目录的 worker 文件，使用 env.close()。"""
    return None
