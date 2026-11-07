"""兼容导入位置；唯一实现见 config_pipeline，无单例配置缓存。"""

from utils.config_pipeline import ConfigLoader, UnifiedConfig
from utils.config_validation import ConfigError, expand_legacy_variant


def get_loader():
    return ConfigLoader()


def load_config(algo, env):
    config = get_loader().load(algo, env)
    return config.algo_args, config.env_args


def load_system_config(system_name):
    return get_loader().load_system(system_name)


def load_training_config(training_ref):
    return get_loader().load_training(training_ref)
