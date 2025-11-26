"""Tools for loading and updating configs."""
import json
import os
import time

import yaml


def get_defaults_yaml_args(algo, env):
    """加载用户指定的算法和环境的配置文件.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
        # 提取环境参数（不包含environment_specific等嵌套配置）
        env_args = {}
        for key, value in env_config.items():
            if key not in ['environment_specific', 'power_system']:
                env_args[key] = value
        
        # 将environment_specific配置单独传递
        if 'environment_specific' in env_config:
            env_args['env_specific_config'] = env_config['environment_specific']
            
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    """使用未解析的命令行参数更新加载的配置。
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    if env == "powerzoo": 
        task = env_args["env_name"]
    elif env == "smartgrid":
        task = env_args["env_name"]
    else:
        task = "unknown"
    return task


def init_dir(env, env_args, algo, exp_name, seed, logger_path):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # 使用绝对路径
    if not os.path.isabs(logger_path):
        logger_path = os.path.abspath(logger_path)
    results_path = os.path.join(
        logger_path,
        env,
        task,
        algo,
        exp_name,
        "-".join(["seed-{:0>5}".format(seed), hms_time]),
    )
    os.makedirs(results_path, exist_ok=True)
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(log_path)
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    # 返回绝对路径
    return os.path.abspath(results_path), os.path.abspath(log_path), os.path.abspath(models_path), writer


def is_json_serializable(value):
    """Check if value is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
    # 不再写入测试代码到progress.txt - 留给base_logger正确处理
        
def save_render(args,run_dir):
    renderdata=convert_json(args)
    output = json.dumps(renderdata, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "renderdata.json"), "w", encoding="utf-8") as out:
        out.write(output)