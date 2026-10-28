"""恢复回放使用真实任务与动作掩码。"""

from envs.powergrid.episode_runner import DeviceEpisodeRunner
from envs.dsr.core.config import DSRConfig
from envs.dsr.core.dsr_core import DSRCoreEnv


class DSREpisodeRunner(DeviceEpisodeRunner):
    def _make_core(self):
        cfg = (
            self.config
            if isinstance(self.config, DSRConfig)
            else DSRConfig.from_env_args(self.config)
        )
        return DSRCoreEnv(cfg, self.project_root)
