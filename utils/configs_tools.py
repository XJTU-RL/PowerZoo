"""Tools for loading and updating configs."""

import json
import os
import time

import yaml


def get_defaults_yaml_args(algo, env):
    """所有环境使用同一解析器；返回值保持 runner 接口兼容。"""
    from utils.unified_config_loader import load_config

    return load_config(algo, env)


def update_args(unparsed_dict, *args):
    """旧接口只允许唯一的叶子名；拒绝拼写错误和歧义。新入口使用 --set。"""
    from utils.config_pipeline import apply_override, leaves
    from utils.config_validation import ConfigError

    for name, value in unparsed_dict.items():
        matches = [
            (item, path)
            for item in args
            for path, _ in leaves(item)
            if path == name or path.rsplit(".", 1)[-1] == name
        ]
        if len(matches) != 1:
            raise ConfigError(
                f"参数 {name!r} 匹配 {len(matches)} 个位置，请使用完整路径 --set"
            )
        apply_override(matches[0][0], matches[0][1], value)


def get_task_name(env, env_args):
    """获取任务名称用于结果目录"""
    # 优先使用 system_name，其次使用 env_name
    if "system_name" in env_args and env_args["system_name"]:
        return os.path.basename(os.path.normpath(env_args["system_name"]))
    if "env_name" in env_args:
        return env_args["env_name"]
    return env


def init_dir(env, env_args, algo, exp_name, seed, logger_path):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    from datetime import datetime

    hms_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
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
    from utils.tensorboard_writer import ScalarWriter

    writer = ScalarWriter(log_path)
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    # 返回绝对路径
    return (
        os.path.abspath(results_path),
        os.path.abspath(log_path),
        os.path.abspath(models_path),
        writer,
    )


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
    from utils.config_pipeline import UnifiedConfig

    main_args = {key: value for key, value in args.items() if not key.startswith("_")}
    trace = args.get("_config_trace", {})
    actual_env_args = dict(env_args, seed=algo_args["seed"]["seed"])
    snapshot = UnifiedConfig(
        algo_args=algo_args,
        env_args=actual_env_args,
        main_args=main_args,
        history=trace.get("parameters", {}),
        files=trace.get("files", {}),
    )
    snapshot.history.setdefault("seed.seed", []).append(
        {"source": "runtime effective seed", "value": algo_args["seed"]["seed"]}
    )
    snapshot.history.setdefault("env.seed", []).append(
        {"source": "runtime effective base seed", "value": actual_env_args["seed"]}
    )
    import copy

    snapshot.algo_args = copy.deepcopy(algo_args)
    snapshot.algo_args["seed"]["seed_specify"] = True
    snapshot.history.setdefault("seed.seed_specify", []).append(
        {"source": "freeze runtime seed for replay", "value": True}
    )
    snapshot.save(os.path.join(run_dir, "config.resolved.yaml"))
    config = {
        "main_args": main_args,
        "algo_args": snapshot.algo_args,
        "env_args": actual_env_args,
    }
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
    # 不再写入测试代码到progress.txt - 留给base_logger正确处理


def save_render(args, run_dir):
    renderdata = convert_json(args)
    output = json.dumps(renderdata, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "renderdata.json"), "w", encoding="utf-8") as out:
        out.write(output)
