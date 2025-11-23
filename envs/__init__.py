from absl import flags
from envs.powerzoo.powerzoo_logger import PowerZooLogger
from envs.powerzoo_llm.logging.powerzoo_llm_logger import PowerZooLLMLogger
from envs.dsr.dsr_logger import DSRLogger
from envs.stackelberg.stackelberg_logger import StackelbergLogger
from envs.stackelberg.stackelberg_powerzoo_env import StackelbergPowerZooEnv, make_stackelberg_env

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

# Logger registry - maps environment names to logger classes
LOGGER_REGISTRY = {
	"powerzoo": PowerZooLogger,
	"powerzoo_llm": PowerZooLLMLogger,
	"dsr": DSRLogger,
	"stackelberg": StackelbergLogger,
	"stackelberg_13bus": StackelbergLogger,
	"stackelberg_34bus": StackelbergLogger,
	"stackelberg_123bus": StackelbergLogger,
	# Backward compatibility - these were incorrectly mapped to Env/factory in main
	"stackelberg_powerzoo": StackelbergLogger,
}

# Environment registry - maps environment names to environment classes (Stackelberg only)
ENV_REGISTRY = {
	"stackelberg": StackelbergPowerZooEnv,
	"stackelberg_13bus": StackelbergPowerZooEnv,
	"stackelberg_34bus": StackelbergPowerZooEnv,
	"stackelberg_123bus": StackelbergPowerZooEnv,
	"stackelberg_powerzoo": StackelbergPowerZooEnv,
}

# Environment factory functions (Stackelberg only)
ENV_FACTORY = {
	"stackelberg": make_stackelberg_env,
	"stackelberg_powerzoo": make_stackelberg_env,
}
