"""UI、预览和训练共用 ConfigLoader；此模块只负责保存快照。"""

import uuid
from pathlib import Path
import yaml
from utils.config_pipeline import ConfigLoader
from utils.config_validation import deep_merge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "training_ui" / ".cache"


def load_base_config(algo, env):
    config = ConfigLoader(PROJECT_ROOT).load(algo, env)
    return config.algo_args, config.env_args


def resolve_config(algo, env, exp_name, ui_overrides=None):
    return ConfigLoader(PROJECT_ROOT).load(
        algo, env, exp_name=exp_name, values=ui_overrides
    )


def build_config(algo, env, exp_name, ui_overrides):
    config = resolve_config(algo, env, exp_name, ui_overrides)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"task_{uuid.uuid4().hex[:12]}.yaml"
    config.save(path)
    return str(path), config.to_dict()


def config_to_yaml_str(config):
    return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
