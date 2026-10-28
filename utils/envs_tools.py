"""Tools."""

import os
import random
import numpy as np
import torch
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_power_env(env_name, env_args, rank, seed):
    """只在此将最终字典转为消费方配置；构造前设置实际 worker seed。"""
    from utils.config_validation import build_dataclass, environment_class

    # 旧 core 在 __init__ 中可能生成随机数据，不能等 wrapper 构造完再播种。
    set_seed({"seed_specify": True, "seed": seed})
    values = dict(env_args, seed=seed)
    cls = environment_class(env_name)
    if "worker_idx" in cls.__dataclass_fields__:
        values["worker_idx"] = rank
    config = build_dataclass(cls, values)
    if env_name in ("vvc", "powerzoo"):
        from envs.vvc.vvc_env import VVCEnv

        env = VVCEnv(config, rank)
    elif env_name == "smartgrid":
        from envs.smartgrid.base_env.env_register import make_base_env
        from envs.smartgrid.base_env.vvc_env import VVCEnv

        env = VVCEnv(make_base_env(config, worker_idx=rank), config, rank)
    elif env_name.startswith("dsr"):
        from envs.dsr.dsr_env import DSREnv

        env = DSREnv(config, rank)
    elif env_name.startswith("stackelberg"):
        from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv

        env = StackelbergVVCEnv(config, rank)
    elif env_name.startswith("district_dispatch"):
        from dataclasses import asdict
        from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv

        env = DistrictDispatchEnv(asdict(config), rank)
    else:
        raise ValueError(f"未知电力环境 {env_name}")
    env.seed(seed)
    return env


def _vector_env(env_name, seed, n_threads, env_args, evaluation=False):
    if n_threads < 1:
        raise ValueError("并行环境数必须 >= 1")
    if env_name == "smartgrid":
        import copy
        env_args = copy.deepcopy(env_args)
        if evaluation:
            env_args["data"]["split"] = "validation"

    def get_env_fn(rank):
        def init_env():
            actual_seed = (
                seed * 50000 + rank * 10000 if evaluation else seed + rank * 1000
            ) % 2**32
            if env_name == "lag":
                from envs.other_envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
                env.seed(actual_seed)
                return env
            return make_power_env(env_name, env_args, rank, actual_seed)

        return init_env

    # OpenDSS 进程使用 spawn；Stackelberg / District 保留既有顺序执行限制。
    wrapper = (
        ShareDummyVecEnv
        if n_threads == 1 or env_name.startswith(("stackelberg", "district_dispatch"))
        else ShareSubprocVecEnv
    )
    return wrapper([get_env_fn(rank) for rank in range(n_threads)])


def make_train_env(env_name, seed, n_threads, env_args):
    if env_name == "dexhands":
        from envs.other_envs.dexhands.dexhands_env import DexHandsEnv

        return DexHandsEnv({"n_threads": n_threads, **env_args})
    return _vector_env(env_name, seed, n_threads, env_args)


def make_eval_env(env_name, seed, n_threads, env_args):
    return _vector_env(env_name, seed, n_threads, env_args, evaluation=True)


def make_render_env(env_name, seed, env_args):
    env = make_power_env(env_name, env_args, 4, (seed * 60000) % 2**32)
    return env, False, False, False, 1


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env in ("vvc", "powerzoo", "PowerZoo"):  # powerzoo/PowerZoo are backward compat
        return envs.n_agents
    elif env == "smartgrid":
        return envs.n_agents
    elif env == "dsr":
        return envs.n_agents
    elif env.startswith("stackelberg"):
        return envs.n_agents
    elif env.startswith("district_dispatch"):
        return envs.n_agents


# def get_agents_orders(env, env_args, envs):
#     """Get the update_orders of agents in the environment."""
#     if env == "powerzoo":
#         return envs.update_orders
def get_ordered_agents_pairs(env, env_args, envs):
    """Get the update_orders of agents in the environment."""
    if env in ("vvc", "powerzoo", "smartgrid", "dsr"):  # powerzoo is backward compat
        if env_args.get("useS", False):
            return envs.ordered_agents_pairs
        else:
            return None
    else:
        return None


def get_agents_bus(env, env_args, envs):
    """Get the agents_bus mapping in the environment."""
    if env in ("vvc", "powerzoo", "smartgrid", "dsr"):  # powerzoo is backward compat
        if env_args.get("useS", False):
            return envs.agents_bus
        else:
            return None
    else:
        return None
