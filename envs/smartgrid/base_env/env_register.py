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
        # pv_plans/ 目录的 overlay 只属于 34Bus_PV；外部系统（如 SMART-DS）
        # 自带 PV 时必须显式 pv_plan=none，防止家族预设把 34Bus 的方案
        # 静默叠加到别的模型上。
        if config.system_name != "34Bus_PV" and config.pv_plan not in (None, "none"):
            raise ValueError(
                f"系统 {config.system_name} 不是 34Bus_PV，pv_plan="
                f"{config.pv_plan!r} 无效；外部模型自带 PV 时请显式 pv_plan=none"
            )
    else:
        from utils.config_pipeline import ConfigLoader
        from utils.config_validation import build_dataclass

        system = {"34Bus_pv": "34Bus_PV"}.get(env_name_or_config, env_name_or_config)
        values = (config_dict or {}).get("env_args", config_dict or {})
        values = dict(values)
        if dss_act:
            raise ValueError("SmartGrid只支持策略控制，不支持dss_act")
        if system != "34Bus_PV":
            values.setdefault("pv_plan", "none")
        loaded = ConfigLoader().load(
            "happo", "smartgrid", system_ref=system, values={"env": values}
        )
        config = build_dataclass(SmartGridConfig, loaded.env_args)
    return Env(str(get_data_root()), config)


make_base_env = make_env


def remove_parallel_dss(env_name, num_workers):
    """兼容旧调用；新内核没有写入模型目录的 worker 文件，使用 env.close()。"""
    return None
