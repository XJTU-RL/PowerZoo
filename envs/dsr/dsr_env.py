"""同步恢复控制组适配器：共享全局观测、团队收益及实际可行动作掩码。"""

from envs.powergrid.adapters import DeviceMARLAdapter
from envs.dsr.core.config import DSRConfig
from envs.dsr.core.dsr_core import DSRCoreEnv


class DSREnv(DeviceMARLAdapter):
    def __init__(self, config_or_args, rank=None):
        cfg = (
            config_or_args
            if isinstance(config_or_args, DSRConfig)
            else DSRConfig.from_env_args(config_or_args)
        )
        super().__init__(DSRCoreEnv(cfg), rank=rank)
        self.core_env = self.env

    def get_avail_actions(self):
        return self.env.available_actions()
