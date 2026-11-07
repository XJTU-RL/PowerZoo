"""VVC公共MARL入口，物理动作和数组适配不重复实现。"""

from envs.powergrid.adapters import DeviceMARLAdapter
from envs.vvc.vvc.env_register import make_base_env


class VVCEnv(DeviceMARLAdapter):
    def __init__(self, config_or_args, rank=None):
        from envs.vvc.vvc.vvc_config import VVCConfig

        config = (
            config_or_args
            if isinstance(config_or_args, VVCConfig)
            else VVCConfig.from_env_args(config_or_args)
        )
        super().__init__(make_base_env(config), rank=rank)
