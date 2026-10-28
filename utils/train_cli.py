"""公共训练入口；配置检查、预览不导入 runner，不创建仿真环境。"""

import argparse
import yaml

from utils.config_pipeline import ConfigLoader, leaves, parse_assignments
from utils.config_validation import ConfigError


def main(argv=None, *, default_env=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo")
    parser.add_argument("--env", default=default_env)
    parser.add_argument("--exp-name", "--exp_name", dest="exp_name")
    parser.add_argument("--config", "--load-config", "--load_config", dest="experiment")
    parser.add_argument("--algo-config", "--algo_config", dest="algo_config")
    parser.add_argument("--env-config", "--env_config", dest="env_config")
    parser.add_argument("--system", dest="system_ref")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--explain", metavar="PATH")
    parser.add_argument("--save-config", metavar="PATH")
    args, legacy = parser.parse_known_args(argv)
    try:
        options = {
            key: getattr(args, key)
            for key in (
                "algo",
                "env",
                "exp_name",
                "experiment",
                "algo_config",
                "env_config",
                "system_ref",
            )
        }
        overrides = parse_assignments(args.set)
        loader = ConfigLoader()
        if legacy:
            if len(legacy) % 2:
                raise ConfigError(
                    "旧参数必须成对提供 --name value；建议使用 --set path=value"
                )
            # 不先校验默认组合，允许 CLI 同时修正采样长度和 RNN 分块等约束。
            preliminary = loader.load(**options, overrides=overrides, validate=False)
            paths = dict(leaves(preliminary.to_dict()))
            for flag, raw in zip(legacy[::2], legacy[1::2]):
                if not flag.startswith("--"):
                    raise ConfigError(f"无效参数 {flag!r}")
                key = flag[2:]
                matches = [
                    path
                    for path in paths
                    if path == key or path.rsplit(".", 1)[-1] == key
                ]
                if len(matches) != 1:
                    raise ConfigError(
                        f"{flag} 匹配 {matches}；使用 --set 完整路径=value"
                    )
                path = matches[0]
                if path in overrides:
                    raise ConfigError(f"重复 CLI 参数 {path}")
                overrides.update(parse_assignments([f"{path}={raw}"]))
        config = loader.load(**options, overrides=overrides)
        if args.save_config:
            config.save(args.save_config)
        if args.print_config:
            print(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False))
        if args.explain:
            print(
                yaml.safe_dump(
                    config.explain(args.explain), allow_unicode=True, sort_keys=False
                )
            )
        if args.validate_config:
            print("配置校验通过")
        if args.print_config or args.validate_config or args.explain:
            return config
    except (ConfigError, FileNotFoundError) as exc:
        parser.error(str(exc))
    from runners import RUNNER_REGISTRY

    if config.main_args["env"] == "smartgrid":
        from envs.smartgrid.data_process.dataset import prepare_dataset
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        from utils.config_validation import build_dataclass

        cfg = build_dataclass(SmartGridConfig, config.env_args)
        dataset = prepare_dataset(cfg.data, cfg.max_episode_steps)
        # 父进程先生成/验证；worker 读取同一数据集，快照记录实际 manifest 路径。
        config.env_args["data"]["path"] = str(dataset.path)
        config.env_args["data"]["manifest_sha256"] = dataset.sha256
        config.files[str(dataset.path)] = dataset.sha256
        for key in ("path", "manifest_sha256"):
            config.history.setdefault(f"env.data.{key}", []).append(
                {
                    "source": "runtime dataset preparation",
                    "value": config.env_args["data"][key],
                }
            )

    runner = RUNNER_REGISTRY[config.main_args["algo"]](
        config.runner_args(), config.algo_args, config.env_args
    )
    try:
        runner.run()
    finally:
        runner.close()
    return config
