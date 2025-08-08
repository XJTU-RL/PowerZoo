from absl import flags
from envs.powerzoo.powerzoo_logger import PowerZooLogger
from envs.powerzoo_llm.logging.powerzoo_llm_logger import PowerZooLLMLogger
from envs.dsr.dsr_logger import DSRLogger
from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.stackelberg.stackelberg_powerzoo_env import StackelbergPowerZooEnv, make_stackelberg_env

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "powerzoo": PowerZooLogger,
    "powerzoo_llm": PowerZooLLMLogger,
    "dsr": DSRLogger,
    "stackelberg": StackelbergBaseEnv,
    "stackelberg_powerzoo": StackelbergPowerZooEnv,
    "stackelberg_powerzoo_make_env": make_stackelberg_env,
}
