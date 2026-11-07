"""区域回放复用正式任务、局部actor观测及原生快照。"""

from envs.powergrid.episode_runner import DeviceEpisodeRunner
from envs.render_common.engine.base_episode_runner import EpisodeData
from envs.district_dispatch.core.config import DistrictDispatchConfig
from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore


class EpisodeRunner(DeviceEpisodeRunner):
    def _make_core(self):
        cfg = (
            self.config
            if isinstance(self.config, DistrictDispatchConfig)
            else DistrictDispatchConfig.from_env_args(self.config)
        )
        return DistrictDispatchCore(cfg, self.project_root)

    def _actor_observations(self, env, obs):
        return env.get_local_obs()
