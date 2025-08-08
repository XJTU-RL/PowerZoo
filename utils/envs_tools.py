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
            if env_name == "powerzoo":
                from envs.powerzoo.powerzoo_env import PowerZooEnv   
                
                env = PowerZooEnv(env_args,rank) 
                
            elif env_name == "powerzoo_llm":
                # Use PowerZooEnv for powerzoo_llm environment
                from envs.powerzoo_llm.base_env.powerzoo_env import PowerZooEnv
                from envs.powerzoo_llm.base_env.env_register import make_base_env
                
                # 简化的配置传递 - 只传递 env_args
                config_dict = {'env_args': env_args}
                
                base_env = make_base_env(
                    env_args.get('env_name', '34Bus_pv'),
                    env_args.get('dss_act', False), 
                    worker_idx=rank,
                    config_dict=config_dict
                )
                env = PowerZooEnv(base_env, env_args, rank)
                
            elif env_name == "dsr":
                from envs.dsr.dsr_env import DSREnv
                
                env = DSREnv(env_args, rank)
                
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
    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])#get_env_fn(i)返回值是单个的环境


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "powerzoo":
                from envs.powerzoo.powerzoo_env import PowerZooEnv
                env = PowerZooEnv(env_args,rank)
                
            elif env_name == "powerzoo_llm":
                from envs.powerzoo_llm.base_env.powerzoo_env import PowerZooEnv
                from envs.powerzoo_llm.base_env.env_register import make_base_env
                
                # 简化的配置传递 - 只传递 env_args
                config_dict = {'env_args': env_args}
                
                base_env = make_base_env(
                    env_args.get('env_name', '34Bus_pv'),
                    env_args.get('dss_act', False), 
                    worker_idx=rank,
                    config_dict=config_dict
                )
                env = PowerZooEnv(base_env, env_args, rank)
                
            elif env_name == "dsr":
                from envs.dsr.dsr_env import DSREnv
                env = DSREnv(env_args, rank)
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
    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    
    if env_name == "powerzoo": #没有环境渲染,这里仅做参数匹配
        from envs.powerzoo.powerzoo_env import PowerZooEnv

        env = PowerZooEnv(env_args,rank=4)
        manual_render = False  
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "powerzoo_llm": #powerzoo_llm环境渲染支持
        from envs.powerzoo_llm.base_env.powerzoo_env import PowerZooEnv
        from envs.powerzoo_llm.base_env.env_register import make_base_env
        
        # 简化的配置传递
        config_dict = {'env_args': env_args}
        base_env = make_base_env(
            env_args.get('env_name', '34Bus_pv'),
            env_args.get('dss_act', False),
            worker_idx=4,
            config_dict=config_dict
        )
        env = PowerZooEnv(base_env, env_args, rank=4)
        manual_render = False  
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)
        
    elif env_name == "dsr":
        from envs.dsr.dsr_env import DSREnv

        env = DSREnv(env_args, rank=4)
        manual_render = False  
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)
        
    elif env_name == "dexhands":
        from envs.other_envs.dexhands.dexhands_env import DexHandsEnv

        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False  # dexhands renders automatically
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env_num = 64
    elif env_name == "lag":
        from envs.other_envs.lag.lag_env import LAGEnv

        env = LAGEnv(env_args)
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
    if env == "powerzoo":
        return envs.n_agents
    elif env == "PowerZoo":
        return envs.n_agents
    elif env == "powerzoo_llm":
        return envs.n_agents
    elif env == "dsr":
        return envs.n_agents

# def get_agents_orders(env, env_args, envs):
#     """Get the update_orders of agents in the environment."""
#     if env == "powerzoo":
#         return envs.update_orders
def get_ordered_agents_pairs(env, env_args, envs):
    """Get the update_orders of agents in the environment."""
    if env in ["powerzoo", "powerzoo_llm"]:
       if env_args.get("useS", False):
           return envs.ordered_agents_pairs
       else:
           return None

def get_agents_bus(env, env_args, envs):
    """Get the update_orders of agents in the environment."""
    if env in ["powerzoo", "powerzoo_llm"]:
       if env_args.get("useS", False):
           return envs.agents_bus
       else:
           return None
    else:
        return None