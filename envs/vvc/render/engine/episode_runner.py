"""VVC回放使用与训练相同的任务、真实设备动作空间及公共物理快照。"""

from envs.powergrid.episode_runner import DeviceEpisodeRunner


class VVCEpisodeRunner(DeviceEpisodeRunner):
    def _make_core(self):
        from envs.vvc.vvc.env_register import make_base_env
        from envs.vvc.vvc.vvc_config import VVCConfig

        cfg = (
            self.config
            if isinstance(self.config, VVCConfig)
            else VVCConfig.from_env_args(self.config)
        )
        return make_base_env(cfg)
