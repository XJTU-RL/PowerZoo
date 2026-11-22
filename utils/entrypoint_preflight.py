"""公共训练入口复用能力目录，不在配置预览阶段加载运行时。"""

from utils.capabilities import device_layout, discover_capabilities, preflight
from utils.config_validation import ConfigError, environment_family


def uses_marl_preflight(config):
    """仅对已声明的 VVC MARL 训练路径启用，保留其它环境与渲染路径。"""
    if (
        environment_family(config.main_args["env"]) != "vvc"
        or config.algo_args["render"]["use_render"]
    ):
        return False
    algorithms = discover_capabilities()["algorithms"]
    profile = algorithms.get(config.main_args["algo"])
    return profile is not None and profile["entrypoint"] == "marl"


def require_preflight(config, *, backend=None, wrapper=None, checkpoint=None):
    """静态检查或用训练器当前的物理设备清单复核，不构造额外 worker。"""
    layout = device_layout(wrapper) if wrapper is not None else None
    report = preflight(config, backend=backend, layout=layout, checkpoint=checkpoint)
    if not report.compatible:
        raise ConfigError("；".join(report.issues))
    return report
