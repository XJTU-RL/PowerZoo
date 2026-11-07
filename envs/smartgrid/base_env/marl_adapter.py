"""SmartGrid的公共MARL入口，数组适配复用公共层。"""

from envs.powergrid.adapters import DeviceMARLAdapter


class VVCEnv(DeviceMARLAdapter):
    def __init__(self, env, config_or_args=None, rank=None):
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        if isinstance(env, SmartGridConfig):
            from envs.smartgrid.base_env.env_register import make_base_env

            env = make_base_env(env)
        super().__init__(env, rank=rank)


OptimizedVVCEnv = VVCEnv
