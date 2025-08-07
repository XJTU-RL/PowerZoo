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
            if env_name == "smac":
                from envs.other_envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from envs.other_envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from envs.other_envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                    MujocoMulti,
                )

                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from envs.other_envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                assert env_args["scenario"] in [
                    "simple_v2",
                    "simple_spread_v2",
                    "simple_reference_v2",
                    "simple_speaker_listener_v3",
                ], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from envs.other_envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from envs.other_envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "powerzoo":
                from envs.power_envs.powerzoo.powerzoo_env import PowerZooEnv   
                
                env = PowerZooEnv(env_args,rank) 
                
            elif env_name == "powerzoo_llm":
                # Use PowerZooEnv for powerzoo_llm environment
                from envs.power_envs.powerzoo_llm.base_env.powerzoo_env import PowerZooEnv
                from envs.power_envs.powerzoo_llm.base_env.env_register import make_base_env
                
                # Create config dict for environment initialization
                config_dict = None
                # Check if we have a dss_file in env_args (from YAML config)
                if 'dss_file' in env_args:
                    # Build complete config dict structure from YAML config
                    config_dict = {
                        'dss_file': env_args.get('dss_file'),  # Pass dss_file at top level
                        'environment_specific': {
                            'system_name': env_args.get('system_name', '34Bus_PV'),
                            'dss_file': env_args.get('dss_file'),
                            'devices': {
                                'regulators': {'action_num': env_args.get('reg_act_num', 33)},
                                'batteries': {'action_num': env_args.get('bat_act_num', 33)},
                                'pv_systems': {
                                    'control_enabled': env_args.get('pv_control', True),
                                    'action_space': 'continuous' if env_args.get('pv_act_num', float('inf')) == float('inf') else 'discrete',
                                    'action_num': env_args.get('pv_act_num', 21) if env_args.get('pv_act_num', float('inf')) != float('inf') else None
                                }
                            },
                            'reward_weights': env_args.get('reward_weights', {})
                        },
                        'env_args': env_args,
                        'train': {'episode_length': env_args.get('episode_length', env_args.get('num_steps', 360))}
                    }
                elif 'env_specific_config' in env_args:
                    # Old path for compatibility
                    config_dict = {
                        'environment_specific': env_args['env_specific_config'],
                        'env_args': env_args,
                        'train': {'episode_length': env_args.get('num_steps', 360)}
                    }
                
                base_env = make_base_env(
                    env_args.get('env_name', '34Bus_pv'),  # Use env_name from config if available
                    env_args.get('dss_act', False), 
                    worker_idx=rank,
                    config_dict=config_dict
                )
                # Create PowerZooEnv wrapper with base environment and config
                env = PowerZooEnv(base_env, env_args, rank)
                
            elif env_name == "dsr":
                from envs.power_envs.dsr.dsr_env import DSREnv
                
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
            if env_name == "smac":
                from envs.other_envs.smac.StarCraft2_Env import StarCraft2Env

                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from envs.other_envs.smacv2.smacv2_env import SMACv2Env

                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from envs.other_envs.mamujoco.multiagent_mujoco.mujoco_multi import (
                    MujocoMulti,
                )
                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from envs.other_envs.pettingzoo_mpe.pettingzoo_mpe_env import (
                    PettingZooMPEEnv,
                )

                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from envs.other_envs.gym.gym_env import GYMEnv

                env = GYMEnv(env_args)
            elif env_name == "football":
                from envs.other_envs.football.football_env import FootballEnv

                env = FootballEnv(env_args)
            elif env_name == "powerzoo":
                from envs.power_envs.powerzoo.powerzoo_env import PowerZooEnv
                env = PowerZooEnv(env_args,rank)
                
            elif env_name == "powerzoo_llm":
                from envs.power_envs.powerzoo_llm.base_env.powerzoo_env import PowerZooEnv
                from envs.power_envs.powerzoo_llm.base_env.env_register import make_base_env
                
                # Create config dict for environment initialization
                config_dict = None
                # Check if we have a dss_file in env_args (from YAML config)
                if 'dss_file' in env_args:
                    # Build complete config dict structure from YAML config
                    config_dict = {
                        'dss_file': env_args.get('dss_file'),  # Pass dss_file at top level
                        'environment_specific': {
                            'system_name': env_args.get('system_name', '34Bus_PV'),
                            'dss_file': env_args.get('dss_file'),
                            'devices': {
                                'regulators': {'action_num': env_args.get('reg_act_num', 33)},
                                'batteries': {'action_num': env_args.get('bat_act_num', 33)},
                                'pv_systems': {
                                    'control_enabled': env_args.get('pv_control', True),
                                    'action_space': 'continuous' if env_args.get('pv_act_num', float('inf')) == float('inf') else 'discrete',
                                    'action_num': env_args.get('pv_act_num', 21) if env_args.get('pv_act_num', float('inf')) != float('inf') else None
                                }
                            },
                            'reward_weights': env_args.get('reward_weights', {})
                        },
                        'env_args': env_args,
                        'train': {'episode_length': env_args.get('episode_length', env_args.get('num_steps', 360))}
                    }
                elif 'env_specific_config' in env_args:
                    # Old path for compatibility
                    config_dict = {
                        'environment_specific': env_args['env_specific_config'],
                        'env_args': env_args,
                        'train': {'episode_length': env_args.get('num_steps', 360)}
                    }
                
                base_env = make_base_env(
                    env_args.get('env_name', '34Bus_pv'),  # Use env_name from config if available
                    env_args.get('dss_act', False), 
                    worker_idx=rank,
                    config_dict=config_dict
                )
                env = PowerZooEnv(base_env, env_args, rank)
                
            elif env_name == "dsr":
                from envs.power_envs.dsr.dsr_env import DSREnv
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
    if env_name == "smac":
        from envs.other_envs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(args=env_args)
        manual_render = (
            False  # smac does not support manually calling the render() function
        )
        # instead, it use save_replay()
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "smacv2":
        from envs.other_envs.smacv2.smacv2_env import SMACv2Env

        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "mamujoco":
        from envs.other_envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        env = MujocoMulti(env_args=env_args)
        env.seed(seed * 60000)
    elif env_name == "pettingzoo_mpe":
        from envs.other_envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
        env.seed(seed * 60000)
    elif env_name == "gym":
        from envs.other_envs.gym.gym_env import GYMEnv

        env = GYMEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "football":
        from envs.other_envs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
        manual_render = False  # football renders automatically
        env.seed(seed * 60000)
    elif env_name == "powerzoo": #没有环境渲染,这里仅做参数匹配
        from envs.power_envs.powerzoo.powerzoo_env import PowerZooEnv

        env = PowerZooEnv(env_args,rank=4)
        manual_render = False  
        manual_expand_dims = (
            False  # dexhands uses parallel envs, thus dimension is already expanded
        )
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "powerzoo_llm": #powerzoo_llm环境渲染支持
        from envs.power_envs.powerzoo_llm.base_env.powerzoo_env import PowerZooEnv
        from envs.power_envs.powerzoo_llm.base_env.env_register import make_base_env
        
        base_env = make_base_env(env_args['env_name'], env_args.get('dss_act', False), worker_idx=4)
        env = PowerZooEnv(base_env, env_args, rank=4)
        manual_render = False  
        manual_expand_dims = False
        manual_delay = False
        env.seed(seed * 60000)
        
    elif env_name == "dsr":
        from envs.power_envs.dsr.dsr_env import DSREnv

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
    if env == "smac":
        from envs.other_envs.smac.smac_maps import get_map_params

        return get_map_params(env_args["map_name"])["n_agents"]
    elif env == "smacv2":
        return envs.n_agents
    elif env == "mamujoco":
        return envs.n_agents
    elif env == "pettingzoo_mpe":
        return envs.n_agents
    elif env == "gym":
        return envs.n_agents
    elif env == "football":
        return envs.n_agents
    elif env == "dexhands":
        return envs.n_agents
    elif env == "lag":
        return envs.n_agents
    elif env == "powerzoo":
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