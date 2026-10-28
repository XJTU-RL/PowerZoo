"""经典VVC的Gymnasium集中控制基线，共享同一任务实例和物理路径。"""

from envs.powergrid.adapters import CentralizedAdapter
from envs.vvc.vvc.env_register import make_base_env


class SingleAgentVVCEnv(CentralizedAdapter):
    def __init__(self, config, action_space_type="continuous"):
        super().__init__(make_base_env(config), action_space_type)
