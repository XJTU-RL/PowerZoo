"""电力环境注册表：配置类型、MARL 包装器和进程策略集中声明，按需导入。"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class EnvironmentSpec:
    config_module: str
    config_class: str
    wrapper_module: str
    wrapper_class: str
    isolate_process: bool = True

    def create(self, config, rank):
        wrapper = getattr(import_module(self.wrapper_module), self.wrapper_class)
        return wrapper(config, rank=rank)


ENVIRONMENTS = {
    "vvc": EnvironmentSpec(
        "envs.vvc.vvc.vvc_config", "VVCConfig", "envs.vvc.vvc_env", "VVCEnv"
    ),
    "smartgrid": EnvironmentSpec(
        "envs.smartgrid.base_env.env_config",
        "SmartGridConfig",
        "envs.smartgrid.base_env.marl_adapter",
        "VVCEnv",
    ),
    "dsr": EnvironmentSpec(
        "envs.dsr.core.config", "DSRConfig", "envs.dsr.dsr_env", "DSREnv"
    ),
    "stackelberg": EnvironmentSpec(
        "envs.stackelberg.stackelberg_config",
        "StackelbergConfig",
        "envs.stackelberg.stackelberg_vvc_env",
        "StackelbergVVCEnv",
    ),
    "district_dispatch": EnvironmentSpec(
        "envs.district_dispatch.core.config",
        "DistrictDispatchConfig",
        "envs.district_dispatch.district_dispatch_env",
        "DistrictDispatchEnv",
    ),
}

# 保留配置模块的历史导出；它是注册表投影，不是第二份独立清单。
ENV_CONFIG_CLASSES = {
    name: (spec.config_module, spec.config_class) for name, spec in ENVIRONMENTS.items()
}


def environment_spec(name):
    from utils.config_validation import environment_family

    return ENVIRONMENTS[environment_family(name)]
