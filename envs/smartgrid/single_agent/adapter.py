"""SmartGrid集中控制入口，共用物理内核和设备动作适配器。"""

from pathlib import Path
from envs.powergrid.adapters import CentralizedAdapter
from envs.smartgrid.base_env.env_config import SmartGridConfig
from envs.smartgrid.base_env.runtime_env import Env


class SingleAgentVVCEnv(CentralizedAdapter):
    def __init__(self, config, action_space_type="continuous", folder_path=None):
        if not isinstance(config, SmartGridConfig):
            raise TypeError("请使用公共ConfigLoader/SmartGridConfig")
        super().__init__(
            Env(str(folder_path or Path(__file__).resolve().parents[3]), config),
            action_space_type,
        )
