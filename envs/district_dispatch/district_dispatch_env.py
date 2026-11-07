"""区域actor局部观测、全局critic状态、共享团队收益的薄适配器。"""

from envs.powergrid.adapters import DeviceMARLAdapter
from envs.district_dispatch.core.config import DistrictDispatchConfig
from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore


class DistrictDispatchEnv(DeviceMARLAdapter):
    def __init__(self, config_or_args, rank=None):
        config = self._parse_config(config_or_args)
        super().__init__(DistrictDispatchCore(config), rank=rank)
        self.core = self.env
        self.observation_space = [self.env.local_observation_space] * self.n_agents
        self.share_observation_space = [self.env.observation_space] * self.n_agents

    @staticmethod
    def _parse_config(values):
        return (
            values
            if isinstance(values, DistrictDispatchConfig)
            else DistrictDispatchConfig.from_env_args(values)
        )

    def _observations(self, obs):
        return self.env.get_local_obs()
