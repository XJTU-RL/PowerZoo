"""Tools for loading and updating configs."""
import json
import os
import time

import yaml


def get_defaults_yaml_args(algo, env):
    """加载用户指定的算法和环境的配置文件.

    自动检测配置格式:
    - 新格式 (v2): 包含 system_ref 字段，使用 ConfigLoader 解析引用
    - 旧格式 (v1): 传统扁平配置，直接解析

    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    # 检测是否使用新格式 (system_ref)
    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_config_peek = yaml.load(f, Loader=yaml.FullLoader) or {}

    if 'system_ref' in env_config_peek:
        # v2 格式: 使用 ConfigLoader 加载并解析引用
        from utils.unified_config_loader import ConfigLoader
        loader = ConfigLoader(base_path)
        config = loader.load(algo, env)
        return config.algo_args, config.env_args

    # v1 格式: 旧格式直接解析
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)

    env_args = {}
    for key, value in env_config_peek.items():
        if key == 'environment_specific':
            env_args['env_specific_config'] = value
        elif key == 'power_system':
            env_args['power_system'] = value
        else:
            env_args[key] = value

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
    """获取任务名称用于结果目录"""
    # 优先使用 system_name，其次使用 env_name
    if 'system_name' in env_args and env_args['system_name']:
        return env_args['system_name']
    if 'env_name' in env_args:
        return env_args['env_name']
    return env


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