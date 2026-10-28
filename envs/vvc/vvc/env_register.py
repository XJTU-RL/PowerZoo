"""VVC工厂仅选择正式配置与任务，不生成worker DSS文件。"""

from dataclasses import asdict
from pathlib import Path
from envs.vvc.vvc.vvc_config import VVCConfig

ROOT = Path(__file__).resolve().parents[3]


def _config(value):
    if isinstance(value, VVCConfig):
        return value
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass, expand_legacy_variant

    variant = dict(expand_legacy_variant(value) or {})
    system = variant.pop("system_ref", value)
    loaded = ConfigLoader().load(
        "happo", "vvc", system_ref=system, values={"env": variant}
    )
    return build_dataclass(VVCConfig, loaded.env_args)


def make_base_env(config_or_name, dss_act=False, worker_idx=None):
    from envs.vvc.runtime_env import Env

    if dss_act:
        raise ValueError("VVC训练由策略控制；原生自动控制仅作独立基线")
    return Env(str(ROOT), _config(config_or_name))


def get_info_and_folder(env_name):
    return asdict(_config(env_name)), str(ROOT / "node_systems")


def remove_parallel_dss(env_name, num_workers):
    """旧调用兼容；新context不向源模型写worker文件，由env.close释放。"""
    return None
