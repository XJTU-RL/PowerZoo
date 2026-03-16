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


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name == "dexhands":
        from envs.other_envs.dexhands.dexhands_env import DexHandsEnv

        return DexHandsEnv({"n_threads": n_threads, **env_args})

    def get_env_fn(rank):
        def init_env():
            if env_name in ("vvc", "powerzoo"):  # powerzoo is backward compat alias
                from envs.vvc.vvc_env import VVCEnv

                env = VVCEnv(env_args,rank) 
                
            elif env_name == "smartgrid":
                from envs.smartgrid.base_env.env_config import SmartGridConfig
                from envs.smartgrid.base_env.vvc_env import VVCEnv
                from envs.smartgrid.base_env.env_register import make_base_env

                config = SmartGridConfig.from_env_args(env_args)
                base_env = make_base_env(config, worker_idx=rank)
                env = VVCEnv(base_env, config, rank)

            elif env_name == "dsr":
                from envs.dsr.core.config import DSRConfig
                from envs.dsr.dsr_env import DSREnv

                config = DSRConfig.from_env_args(env_args)
                env = DSREnv(config, rank)

            elif env_name.startswith("stackelberg"):
                from envs.stackelberg.stackelberg_config import StackelbergConfig
                from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv

                config = StackelbergConfig.from_env_args({**env_args, 'env_name': env_name})
                env = StackelbergVVCEnv(config, rank)

            elif env_name.startswith("district_dispatch"):
                from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv

                env = DistrictDispatchEnv({**env_args, 'worker_idx': rank}, rank)

            elif env_name == "lag":
                from envs.other_envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)

            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env
    print("train env的数量是=",n_threads)
    # Stackelberg/DistrictDispatch环境的OpenDSS引擎不支持fork-based多进程，
    # 强制使用DummyVecEnv避免SubprocVecEnv的EOFError
    if n_threads == 1 or env_name.startswith("stackelberg") or env_name.startswith("district_dispatch"):
        if n_threads > 1 and (env_name.startswith("stackelberg") or env_name.startswith("district_dispatch")):
            print(f"注意: {env_name}环境不支持SubprocVecEnv，降级为DummyVecEnv ({n_threads}个顺序环境)")
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])#get_env_fn(i)返回值是单个的环境


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name in ("vvc", "powerzoo"):  # powerzoo is backward compat alias
                from envs.vvc.vvc_env import VVCEnv
                env = VVCEnv(env_args,rank)
                
            elif env_name == "smartgrid":
                from envs.smartgrid.base_env.env_config import SmartGridConfig
                from envs.smartgrid.base_env.vvc_env import VVCEnv
                from envs.smartgrid.base_env.env_register import make_base_env

                config = SmartGridConfig.from_env_args(env_args)
                base_env = make_base_env(config, worker_idx=rank)
                env = VVCEnv(base_env, config, rank)

            elif env_name == "dsr":
                from envs.dsr.core.config import DSRConfig
                from envs.dsr.dsr_env import DSREnv

                config = DSRConfig.from_env_args(env_args)
                env = DSREnv(config, rank)

            elif env_name.startswith("stackelberg"):
                from envs.stackelberg.stackelberg_config import StackelbergConfig
                from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv

                config = StackelbergConfig.from_env_args({**env_args, 'env_name': env_name})
                env = StackelbergVVCEnv(config, rank)

            elif env_name.startswith("district_dispatch"):
                from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv

                env = DistrictDispatchEnv({**env_args, 'worker_idx': rank}, rank)

            elif env_name == "lag":
                from envs.other_envs.lag.lag_env import LAGEnv

                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env
    print("eval env 的数量是 =",n_threads)
    if n_threads == 1 or env_name.startswith("stackelberg") or env_name.startswith("district_dispatch"):
        if n_threads > 1 and (env_name.startswith("stackelberg") or env_name.startswith("district_dispatch")):
            print(f"注意: {env_name}环境不支持SubprocVecEnv，降级为DummyVecEnv ({n_threads}个顺序环境)")
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    
    if env_name in ("vvc", "powerzoo"):  # 没有环境渲染,这里仅做参数匹配
        from envs.vvc.vvc_env import VVCEnv

        env = VVCEnv(env_args,rank=4)
        manual_render = False  
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "smartgrid": #smartgrid环境渲染支持
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        from envs.smartgrid.base_env.vvc_env import VVCEnv
        from envs.smartgrid.base_env.env_register import make_base_env

        config = SmartGridConfig.from_env_args(env_args)
        base_env = make_base_env(config, worker_idx=4)
        env = VVCEnv(base_env, config, rank=4)
        manual_render = False  
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)
        
    elif env_name == "dsr":
        from envs.dsr.core.config import DSRConfig
        from envs.dsr.dsr_env import DSREnv

        config = DSRConfig.from_env_args(env_args)
        env = DSREnv(config, rank=4)
        manual_render = False
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)

    elif env_name.startswith("stackelberg"):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv

        config = StackelbergConfig.from_env_args({**env_args, 'env_name': env_name})
        env = StackelbergVVCEnv(config, rank=4)
        manual_render = False
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)

    elif env_name.startswith("district_dispatch"):
        from envs.district_dispatch.district_dispatch_env import DistrictDispatchEnv

        env = DistrictDispatchEnv({**env_args, 'worker_idx': 4}, rank=4)
        manual_render = False
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)

    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


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