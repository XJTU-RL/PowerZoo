"""SmartGrid回放仅补充CMDP快照，物理和动作适配复用公共运行器。"""

from envs.powergrid.episode_runner import DeviceEpisodeRunner
from envs.smartgrid.render.data.snapshot_assembler import assemble_snapshot


class SmartGridEpisodeRunner(DeviceEpisodeRunner):
    def _make_core(self):
        from envs.smartgrid.base_env.runtime_env import Env
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        cfg = (
            self.config
            if isinstance(self.config, SmartGridConfig)
            else SmartGridConfig.from_dict(self.config)
        )
        return Env(self.project_root, cfg)

    def _snapshot(self, env, step, actions, rewards, infos):
        return assemble_snapshot(env, step, actions, rewards, infos)
