"""运行时注册表延迟加载；读取环境配置不初始化仿真库。"""


def __getattr__(name):
    if name not in ("LOGGER_REGISTRY", "ENV_REGISTRY", "ENV_FACTORY", "FLAGS"):
        raise AttributeError(name)
    from absl import flags
    from envs.vvc.vvc_logger import VVCLogger
    from envs.smartgrid.logging.smartgrid_logger import SmartGridLogger
    from envs.dsr.dsr_logger import DSRLogger
    from envs.stackelberg.stackelberg_logger import StackelbergLogger
    from envs.stackelberg.stackelberg_vvc_env import (
        StackelbergVVCEnv,
        make_stackelberg_env,
    )
    from envs.district_dispatch.district_dispatch_logger import DistrictDispatchLogger

    FLAGS = flags.FLAGS
    FLAGS(["train_sc.py"])

    # Logger registry - maps environment names to logger classes
    LOGGER_REGISTRY = {
        "vvc": VVCLogger,
        "powerzoo": VVCLogger,  # Backward compatibility alias
        "smartgrid": SmartGridLogger,
        "dsr": DSRLogger,
        "stackelberg": StackelbergLogger,
        "stackelberg_13bus": StackelbergLogger,
        "stackelberg_34bus": StackelbergLogger,
        "stackelberg_123bus": StackelbergLogger,
        # Backward compatibility - these were incorrectly mapped to Env/factory in main
        "stackelberg_vvc": StackelbergLogger,
        "district_dispatch": DistrictDispatchLogger,
    }

    # Environment registry - maps environment names to environment classes (Stackelberg only)
    ENV_REGISTRY = {
        "stackelberg": StackelbergVVCEnv,
        "stackelberg_13bus": StackelbergVVCEnv,
        "stackelberg_34bus": StackelbergVVCEnv,
        "stackelberg_123bus": StackelbergVVCEnv,
        "stackelberg_vvc": StackelbergVVCEnv,
    }

    # Environment factory functions (Stackelberg only)
    ENV_FACTORY = {
        "stackelberg": make_stackelberg_env,
        "stackelberg_vvc": make_stackelberg_env,
    }
    globals().update(
        LOGGER_REGISTRY=LOGGER_REGISTRY,
        ENV_REGISTRY=ENV_REGISTRY,
        ENV_FACTORY=ENV_FACTORY,
        FLAGS=FLAGS,
    )
    return globals()[name]
