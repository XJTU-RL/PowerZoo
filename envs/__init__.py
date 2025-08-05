from absl import flags
# Other environments
from envs.other_envs.smac.smac_logger import SMACLogger
from envs.other_envs.smacv2.smacv2_logger import SMACv2Logger
from envs.other_envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from envs.other_envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from envs.other_envs.gym.gym_logger import GYMLogger
from envs.other_envs.football.football_logger import FootballLogger
from envs.other_envs.dexhands.dexhands_logger import DexHandsLogger
from envs.other_envs.lag.lag_logger import LAGLogger
# Power system environments
from envs.power_envs.powerzoo.powerzoo_logger import PowerZooLogger
from envs.power_envs.powerzoo_llm.powerzoo_llm_logger import PowerZooLLMLogger
from envs.power_envs.dsr.dsr_logger import DSRLogger
from envs.power_envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.power_envs.stackelberg.stackelberg_powerzoo_env import StackelbergPowerZooEnv, make_stackelberg_env

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "lag": LAGLogger,
    "powerzoo": PowerZooLogger,
    "powerzoo_llm": PowerZooLLMLogger,
    "dsr": DSRLogger,
    "stackelberg": StackelbergBaseEnv,
    "stackelberg_powerzoo": StackelbergPowerZooEnv,
    "stackelberg_powerzoo_make_env": make_stackelberg_env,
}
