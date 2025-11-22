"""
PowerZoo Environments Module
============================

This module provides centralized registries for environment loggers and environment factories.

Registries:
-----------
- LOGGER_REGISTRY: Maps environment names to their logger classes
- ENV_REGISTRY: Maps environment names to their environment classes or factory functions

Usage:
------
    from envs import LOGGER_REGISTRY, ENV_REGISTRY

    logger = LOGGER_REGISTRY["powerzoo"](...)
    env = ENV_REGISTRY["stackelberg"](args)
"""

from absl import flags

# Logger imports
from envs.powerzoo.powerzoo_logger import PowerZooLogger
from envs.powerzoo_llm.logging.powerzoo_llm_logger import PowerZooLLMLogger
from envs.dsr.dsr_logger import DSRLogger

# Environment imports
from envs.stackelberg.stackelberg_game.stackelberg_base_env import StackelbergBaseEnv
from envs.stackelberg.stackelberg_powerzoo_env import StackelbergPowerZooEnv, make_stackelberg_env

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])


# Logger registry - maps environment names to logger classes
LOGGER_REGISTRY = {
	"powerzoo": PowerZooLogger,
	"powerzoo_llm": PowerZooLLMLogger,
	"dsr": DSRLogger,
}


# Environment registry - maps environment names to environment classes or factory functions
ENV_REGISTRY = {
	"stackelberg": StackelbergBaseEnv,
	"stackelberg_powerzoo": StackelbergPowerZooEnv,
	"stackelberg_powerzoo_make_env": make_stackelberg_env,
}
