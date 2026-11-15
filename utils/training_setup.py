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
        smartds_spec = config.env_args.get("smartds_series")
        if smartds_spec:
            # SMART-DS 真实逐设备时序：worker 实际消费的是真实曲线而非合成
            # 场景清单——冻结真实数据内容摘要与外部清单指纹，不把未使用的
            # 合成清单登记为数据来源（复审 4d8af3b5）。
            from envs.powergrid.smartds_timeseries import SmartDSEpisodeDataset
            from utils.external_cases import locate_entry, manifest_path

            folder, master = locate_entry(
                smartds_spec["case"], smartds_spec["entry"]
            )
            layout = folder.parents[4]
            rel_master = str(folder.relative_to(layout)) + "/" + master
            dataset = SmartDSEpisodeDataset(
                layout,
                rel_master,
                days={
                    "train": smartds_spec["train_days"],
                    "validation": smartds_spec.get("validation_days", []),
                },
                split=smartds_spec.get("split", "train"),
            )
            from hashlib import sha256 as _sha256
            from pathlib import Path as _Path

            external_manifest = _Path(manifest_path(smartds_spec["case"]))
            manifest_sha256 = _sha256(external_manifest.read_bytes()).hexdigest()
            frozen = {
                "case": smartds_spec["case"],
                "entry": smartds_spec["entry"],
                "master": rel_master,
                "days": dataset.day_indices,
                "split": dataset.split,
                # 曲线/脚本内容指纹：worker 恢复时按各自 split 校验
                # （control_env 构造期）。摘要组成含 split——train 与
                # validation 是不同数据切片，须分别冻结。
                "dataset_sha256": dataset.sha256,
                # 外部清单指纹：逐文件哈希与派生补丁版本的载体。
                "manifest_sha256": manifest_sha256,
            }
            if dataset.day_indices["validation"]:
                validation_dataset = SmartDSEpisodeDataset(
                    layout,
                    rel_master,
                    days={
                        "train": smartds_spec["train_days"],
                        "validation": smartds_spec.get("validation_days", []),
                    },
                    split="validation",
                )
                frozen["dataset_sha256_validation"] = validation_dataset.sha256
            config.env_args["smartds_series"] = {**smartds_spec, **frozen}
            config.files[str(external_manifest)] = manifest_sha256
            for key in (
                "dataset_sha256",
                "dataset_sha256_validation",
                "manifest_sha256",
            ):
                if frozen.get(key) is None:
                    continue
                path = f"env.smartds_series.{key}"
                entry = {"source": "runtime smartds dataset preparation", "value": frozen[key]}
                if not config.history.get(path) or config.history[path][-1] != entry:
                    config.history.setdefault(path, []).append(entry)
        elif family == "dsr":
            from envs.dsr.core.config import FaultDataConfig
            from envs.dsr.core.scenario import RestorationDataset

            dataset = RestorationDataset(FaultDataConfig(**data))
            _freeze_synthetic_dataset(config, data, dataset)
        else:
            from envs.powergrid.scenario import DataConfig, prepare_dataset

            dataset = prepare_dataset(
                DataConfig(**data), config.env_args["max_episode_steps"]
            )
            _freeze_synthetic_dataset(config, data, dataset)
    return config


def _freeze_synthetic_dataset(config, data, dataset):
    """合成数据路径的既有冻结：清单路径与摘要进来源表（行为不变）。"""
    data.update(path=str(dataset.path), manifest_sha256=dataset.sha256)
    config.files[str(dataset.path)] = dataset.sha256
    for key in ("path", "manifest_sha256"):
        path = f"env.data.{key}"
        entry = {"source": "runtime dataset preparation", "value": data[key]}
        if not config.history.get(path) or config.history[path][-1] != entry:
            config.history.setdefault(path, []).append(entry)
