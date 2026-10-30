"""正式训练启动前的共享准备；不导入 Torch、DSS 或训练器。"""

import secrets

def prepare_run(config):
    """确定一次实际种子和数据集，并把派生值记入来源表。

    env_args 是唯一可序列化配置。消费方 dataclass 在 worker 构造一次，
    此处只读取已校验的数据配置，不重复构造完整环境配置。
    """
    from utils.config_validation import environment_family

    family = environment_family(config.main_args["env"])
    seed = config.algo_args["seed"]
    if not seed["seed_specify"]:
        seed["seed"] = secrets.randbelow(2**32)
        seed["seed_specify"] = True
        config.env_args["seed"] = seed["seed"]
        for path, value in (
            ("seed.seed", seed["seed"]),
            ("env.seed", seed["seed"]),
            ("seed.seed_specify", True),
        ):
            config.history.setdefault(path, []).append(
                {"source": "runtime seed selection", "value": value}
            )
    if family in ("smartgrid", "vvc", "district_dispatch", "dsr", "stackelberg"):
        data = config.env_args["data"]
        if family == "dsr":
            from envs.dsr.core.config import FaultDataConfig
            from envs.dsr.core.scenario import RestorationDataset

            dataset = RestorationDataset(FaultDataConfig(**data))
        else:
            from envs.powergrid.scenario import DataConfig, prepare_dataset

            dataset = prepare_dataset(
                DataConfig(**data), config.env_args["max_episode_steps"]
            )
        data.update(path=str(dataset.path), manifest_sha256=dataset.sha256)
        config.files[str(dataset.path)] = dataset.sha256
        for key in ("path", "manifest_sha256"):
            path = f"env.data.{key}"
            entry = {"source": "runtime dataset preparation", "value": data[key]}
            if not config.history.get(path) or config.history[path][-1] != entry:
                config.history.setdefault(path, []).append(entry)
    return config
