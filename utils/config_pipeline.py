"""唯一配置解析过程：文件和 CLI -> 完整值、校验、来源；不创建环境或训练器。"""

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from pathlib import Path

import yaml

from utils.config_validation import (
    ConfigError,
    REWARD_ALIASES,
    build_dataclass,
    deep_merge,
    environment_class,
    environment_family,
    normalize_env,
    typed_value,
)

ALGORITHM_SECTIONS = (
    "seed",
    "device",
    "train",
    "eval",
    "render",
    "model",
    "algo",
    "logger",
)


class UniqueKeyLoader(yaml.SafeLoader):
    """禁止 YAML 重复键，避免后一个值悄悄覆盖前一个值。"""


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ConfigError(f"{key_node.start_mark}: 重复配置键 {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def read_yaml(path):
    path = Path(path)
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: 配置顶层必须是映射")
    return value


def leaves(value, prefix=""):
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict) and item:
            yield from leaves(item, path)
        else:
            yield path, copy.deepcopy(item)


def apply_override(document, path, value):
    """CLI 只能修改已存在的确切路径，不会猜测多个同名字段。"""
    keys = path.split(".")
    target = document
    for key in keys[:-1]:
        if (
            isinstance(target, dict)
            and key not in target
            and key.isdigit()
            and int(key) in target
        ):
            key = int(key)
        if not isinstance(target, dict) or key not in target:
            raise ConfigError(f"未知参数 {path!r}：路径 {key!r} 不存在")
        target = target[key]
    key = keys[-1]
    if (
        isinstance(target, dict)
        and key not in target
        and key.isdigit()
        and int(key) in target
    ):
        key = int(key)
    if not isinstance(target, dict) or key not in target:
        raise ConfigError(
            f"未知参数 {path!r}，请先在该环境配置类或算法默认 YAML 中声明"
        )
    target[key] = copy.deepcopy(value)


def parse_assignments(assignments):
    parsed = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ConfigError(f"--set 需要 path=value，实际为 {assignment!r}")
        path, raw = assignment.split("=", 1)
        if not path or path in parsed:
            raise ConfigError(f"重复或空的 CLI 参数路径 {path!r}")
        try:
            parsed[path] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: 无效 YAML 值 {raw!r}") from exc
    return parsed


def validate_algorithm(values, defaults, prefix=""):
    """算法 YAML 本身即字段表；新增参数不需要再维护第二份 schema。"""
    result = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in defaults:
            raise ConfigError(f"未知算法参数 {path!r}；请在算法默认 YAML 中声明")
        expected = defaults[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}: 必须是映射")
            result[key] = (
                validate_algorithm(value, expected, path)
                if expected
                else copy.deepcopy(value)
            )
        elif expected is None:
            result[key] = copy.deepcopy(value)
        else:
            # PyYAML 将 1e-4 读为字符串；算法数值仍按数值校验。
            numeric = isinstance(expected, str) and re.fullmatch(
                r"[-+]?\d+(?:\.\d*)?[eE][-+]?\d+", expected
            )
            result[key] = typed_value(value, float if numeric else type(expected), path)
    return result


@dataclass
class UnifiedConfig:
    algo_args: dict = field(default_factory=dict)
    env_args: dict = field(default_factory=dict)
    system_config: dict = field(default_factory=dict)
    training_config: dict | None = None
    main_args: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "run": copy.deepcopy(self.main_args),
            **copy.deepcopy(self.algo_args),
            "env": copy.deepcopy(self.env_args),
        }

    def explain(self, path):
        target = self.to_dict()
        for key in path.split("."):
            if isinstance(target, dict) and key not in target and key.isdigit():
                key = int(key)
            if not isinstance(target, dict) or key not in target:
                raise ConfigError(f"未知参数路径 {path!r}")
            target = target[key]
        children = {
            name: entries
            for name, entries in self.history.items()
            if name.startswith(path + ".")
        }
        if children:
            return {
                "path": path,
                "value": target,
                "parameters": copy.deepcopy(children),
            }
        return {
            "path": path,
            "value": target,
            "history": copy.deepcopy(self.history[path]),
        }

    def provenance(self):
        return {"files": self.files, "parameters": self.history}

    def runner_args(self):
        return {**self.main_args, "_config_trace": self.provenance()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"config_kind": "resolved", **self.to_dict()}
        path.write_text(
            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        path.with_name(path.stem + ".sources.yaml").write_text(
            yaml.safe_dump(self.provenance(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


class ConfigLoader:
    """每次 load 重新读取文件；没有单例值缓存，也没有递归继承或隐式插值。"""

    def __init__(self, base_path=None):
        self.base_path = Path(base_path or Path(__file__).resolve().parents[1])

    def _path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.base_path / path

    def _read(self, path, files):
        path = self._path(path).resolve()
        data = read_yaml(path)
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return data, str(path)

    def load_system(self, name):
        return read_yaml(self.base_path / "configs/systems" / f"{name}.yaml")

    def load_training(self, name):
        return read_yaml(self.base_path / "configs/training" / f"{name}.yaml")

    def clear_cache(self):
        """保留旧 API；加载器已不缓存配置内容。"""

    def _system_values(self, system, family):

        def check(values, allowed, path):
            if not isinstance(values, dict):
                raise ConfigError(f"{path}必须是映射")
            unknown = set(values) - set(allowed)
            if unknown:
                raise ConfigError(f"{path}: 未知或无消费者的系统参数 {sorted(unknown)}")

        check(
            system,
            {"system", "episode", "devices", "constraints", "default_rewards"},
            "system document",
        )
        check(
            system.get("system", {}),
            {"name", "dss_file", "dss_folder", "solve_algorithm"},
            "system",
        )
        check(system.get("episode", {}), {"max_steps"}, "episode")
        check(
            system.get("constraints", {}), {"voltage_min", "voltage_max"}, "constraints"
        )
        weights = environment_class("smartgrid")().reward_weights
        check(system.get("default_rewards", {}), asdict(weights), "default_rewards")
        devices = system.get("devices", {})
        check(devices, {"regulators", "batteries", "pv_systems"}, "devices")
        for name, values in devices.items():
            check(
                values,
                (
                    {"action_num", "control_enabled"}
                    if name == "pv_systems"
                    else {"action_num"}
                ),
                f"devices.{name}",
            )
        info = system.get("system", {})
        values = {key: info[key] for key in ("dss_file", "dss_folder") if key in info}
        if "name" in info:
            values["system_name"] = info["name"]
        if "solve_algorithm" in info:
            values["solve_algorithm"] = info["solve_algorithm"]
        if "max_steps" in system.get("episode", {}):
            values["max_episode_steps"] = system["episode"]["max_steps"]
        if family in ("vvc", "smartgrid"):
            values["environment_specific"] = {
                "devices": system.get("devices", {}),
                "reward_weights": system.get("default_rewards", {}),
                "constraints": system.get("constraints", {}),
            }
        elif family in ("dsr", "district_dispatch"):
            values.update(system.get("constraints", {}))
        return normalize_env(values, family)

    @staticmethod
    def _document(data):
        """仅入口兼容历史完整配置；内部只有 run / 算法各节 / env 一种形状。"""
        if "main_args" in data or "algo_args" in data:
            run = data.get("main_args", {}) or {
                "algo": data.get("algo_name"),
                "env": data.get("env_name"),
                "exp_name": data.get("exp_name"),
            }
            return {
                "run": {
                    key: value
                    for key, value in run.items()
                    if key in ("algo", "env", "exp_name") and value is not None
                },
                **data.get("algo_args", {}),
                "env": data.get("env_args", {}),
            }
        if "algo_name" in data:
            run = {
                key: data[old]
                for key, old in (
                    ("algo", "algo_name"),
                    ("env", "env_name"),
                    ("exp_name", "exp_name"),
                )
                if old in data
            }
            return {
                "run": run,
                **{
                    key: value
                    for key, value in data.items()
                    if key not in ("algo_name", "env_name", "exp_name", "env_args")
                },
                "env": data.get("env_args", {}),
            }
        if "env_args" in data:
            return {
                **{key: value for key, value in data.items() if key != "env_args"},
                "env": data["env_args"],
            }
        return copy.deepcopy(data)

    def load(
        self,
        algo=None,
        env=None,
        *,
        algo_config=None,
        env_config=None,
        experiment=None,
        overrides=None,
        system_ref=None,
        exp_name=None,
        validate=True,
        values=None,
    ):
        files, history = {}, {}
        experiment_data, experiment_source = {}, None
        if experiment:
            raw, experiment_source = self._read(experiment, files)
            experiment_data = self._document(raw)
        for data in (experiment_data, values or {}):
            if not isinstance(data, dict) or not isinstance(data.get("run", {}), dict):
                raise ConfigError("实验和 run 必须是映射")
        selected = deep_merge(
            experiment_data.get("run", {}), (values or {}).get("run", {})
        )
        algo = algo or selected.get("algo", "happo")
        env = env or selected.get("env", "vvc")
        env = "vvc" if env == "powerzoo" else env
        profile_name = env
        family = environment_family(env)
        env = family
        cls = environment_class(env)
        defaults, algo_source = self._read(f"configs/algos_cfgs/{algo}.yaml", files)
        if not all(isinstance(defaults.get(key), dict) for key in ALGORITHM_SECTIONS):
            raise ConfigError(
                f"{algo_source}: 此研究算法尚未适配公共 runner 的八节配置契约，不能通过统一入口启动"
            )
        replay = experiment_data.pop("config_kind", None) == "resolved"
        result = UnifiedConfig(
            main_args={
                "algo": algo,
                "env": env,
                "exp_name": exp_name or selected.get("exp_name", "test"),
            },
            files=files,
            history=history,
        )
        document = {
            "run": result.main_args,
            **copy.deepcopy(defaults),
            "env": asdict(cls()),
        }

        def merge(values, source):
            nonlocal document
            document = deep_merge(document, values)
            for path, value in leaves(values):
                history.setdefault(path, []).append({"source": source, "value": value})

        merge({"run": result.main_args}, "entry point")
        if not replay:
            merge(defaults, algo_source)
            merge({"env": asdict(cls())}, f"{cls.__module__}.{cls.__name__} defaults")
        if algo_config and replay:
            raise ConfigError("重放最终配置时不能重新选择算法配置；请用 --set 显式修改")
        if algo_config:
            custom, source = self._read(algo_config, files)
            merge(custom.get("algo_args", custom), source)

        profiles = []
        if not replay:
            base, source = self._read(f"configs/envs_cfgs/{profile_name}.yaml", files)
            profiles.append((base, source))
        if env_config:
            if replay:
                raise ConfigError(
                    "重放最终配置时不能重新选择环境配置；请用 --set 显式修改"
                )
            profile, source = self._read(env_config, files)
            profiles.append((self._document(profile), source))
        if experiment_data:
            profiles.append((experiment_data, experiment_source))
        if values:
            profiles.append((values, "Python / UI overrides"))

        # 先确定最终引用，再展开一次，避免系统切换后残留上一系统的派生字段。
        final_system, final_training = system_ref, None
        if not replay:
            for profile, _ in profiles:
                if "env" in profile and not isinstance(profile["env"], dict):
                    raise ConfigError("env 必须是映射")
                final_system = profile.get(
                    "system_ref", profile.get("env", {}).get("system_ref", final_system)
                )
                final_training = profile.get("training_ref", final_training)
            if system_ref:
                final_system = system_ref
            if final_system:
                result.system_config, source = self._read(
                    f"configs/systems/{final_system}.yaml", files
                )
                merge(
                    {"env": self._system_values(result.system_config, family)}, source
                )
            if final_training:
                result.training_config, source = self._read(
                    f"configs/training/{final_training}.yaml", files
                )
                if not set(result.training_config) <= set(defaults):
                    raise ConfigError(
                        f"{source}: 旧课程策略没有训练消费者；training_ref 只接受算法参数节"
                    )
                training_source = source
        for index, (profile, source) in enumerate(profiles):
            profile = copy.deepcopy(profile)
            for key in ("system_ref", "training_ref"):
                profile.pop(key, None)
            run = profile.get("run", {})
            if set(run) - {"algo", "env", "exp_name"}:
                raise ConfigError(f"{source}: run 只允许 algo、env、exp_name")
            if (
                "env" not in profile
                and "run" not in profile
                and not set(profile) <= set(defaults)
            ):
                # 旧环境文件的平铺键在边界转换，正式文件使用 env: 节。
                merge({"env": normalize_env(profile, family)}, source)
            else:
                values = profile.pop("env", {})
                values.pop("system_ref", None)
                merge({**profile, "env": normalize_env(values, family)}, source)
            if not replay and index == 0 and result.training_config is not None:
                # 环境文件包含推荐训练设置；显式选择的共享训练预设覆盖它。
                merge(result.training_config, training_source)

        # main_args 只描述选择，不能被环境里的 env_name 或算法里的名字悄悄改写。
        merge({"run": result.main_args}, "final entry point selection")
        for path, value in (overrides or {}).items():
            if path.startswith("run."):
                raise ConfigError("run 中的选择请用 --algo、--env、--exp-name 指定")
            apply_override(document, path, value)
            history.setdefault(path, []).append(
                {"source": "CLI --set", "value": copy.deepcopy(value)}
            )
        unknown = set(document) - {"run", "env", *defaults}
        if unknown:
            raise ConfigError(f"未知配置节 {sorted(unknown)}")
        result.algo_args = validate_algorithm(
            {
                key: value
                for key, value in document.items()
                if key not in ("run", "env")
            },
            defaults,
        )
        seed = result.algo_args.get("seed", {}).get("seed")
        if "env.seed" in (overrides or {}):
            raise ConfigError(
                "请使用 --set seed.seed 设置全局种子；env.seed 由它和 worker 编号派生"
            )
        explicit_seeds = [
            entry
            for entry in history.get("env.seed", [])
            if not entry["source"].endswith(" defaults")
        ]
        if explicit_seeds and explicit_seeds[-1]["value"] != seed:
            raise ConfigError("env.seed 与 seed.seed 冲突；请只设置 seed.seed")
        if seed is not None:
            document["env"]["seed"] = seed
            history.setdefault("env.seed", []).append(
                {
                    "source": "derived from seed.seed; workers add their rank offset",
                    "value": seed,
                }
            )
        config = build_dataclass(cls, document["env"])
        result.env_args = asdict(config)
        if validate:
            self._validate_run(result)
        # 数值转换后的值也进入来源表；预览、快照与消费方看到相同类型。
        for path, value in leaves(result.to_dict()):
            if path not in history:
                history[path] = [{"source": "dataclass default", "value": value}]
            elif history[path][-1]["value"] != value or type(
                history[path][-1]["value"]
            ) is not type(value):
                history[path].append(
                    {"source": "validated numeric conversion", "value": value}
                )
        return result

    @staticmethod
    def _validate_run(config):
        env = config.env_args
        if env.get("max_episode_steps", 1) < 1:
            raise ConfigError("env.max_episode_steps 必须为正整数")
        low, high = env.get("voltage_min", env.get("v_min")), env.get(
            "voltage_max", env.get("v_max")
        )
        if low is not None and not 0 < low < high < 2:
            raise ConfigError("环境电压约束必须满足 0 < min < max < 2")
        algo = config.algo_args
        seed = algo["seed"]["seed"]
        if not 0 <= seed < 2**32:
            raise ConfigError("seed.seed 必须在 0 与 2**32 - 1 之间")
        for path, value in leaves(config.to_dict()):
            if (
                isinstance(value, float)
                and not math.isfinite(value)
                and not path.endswith(("act_num", "action_num"))
            ):
                raise ConfigError(f"{path}: 必须是有限数值")
        train, model = algo.get("train", {}), algo.get("model", {})
        if model.get("hidden_sizes") is not None and (
            not model["hidden_sizes"]
            or any(type(size) is not int or size < 1 for size in model["hidden_sizes"])
        ):
            raise ConfigError("model.hidden_sizes 必须是非空的正整数列表")
        for section, key in (
            ("device", "torch_threads"),
            ("train", "log_interval"),
            ("train", "eval_interval"),
            ("train", "save_interval"),
            ("eval", "n_eval_rollout_threads"),
            ("eval", "eval_episodes"),
        ):
            if (
                key in algo[section]
                and algo[section][key] is not None
                and algo[section][key] < 1
            ):
                raise ConfigError(f"{section}.{key} 必须为正整数")
        for key in ("episode_length", "n_rollout_threads", "num_env_steps"):
            if key in train and train[key] <= 0:
                raise ConfigError(f"train.{key} 必须为正整数")
        steps = train.get("episode_length", 1) * train.get("n_rollout_threads", 1)
        if (
            not algo.get("render", {}).get("use_render", False)
            and train.get("num_env_steps", steps) < steps
        ):
            raise ConfigError(
                "train.num_env_steps 小于一次采样所需步数，训练将执行零次更新"
            )
        if model.get("use_recurrent_policy"):
            chunk = model.get("data_chunk_length", 1)
            if chunk < 1 or train.get("episode_length", 1) % chunk:
                raise ConfigError(
                    "train.episode_length 必须是 model.data_chunk_length 的整数倍"
                )
            steps //= chunk
        for key in ("actor_num_mini_batch", "critic_num_mini_batch"):
            batches = algo.get("algo", {}).get(key, 1)
            if not 1 <= batches <= steps:
                raise ConfigError(f"algo.{key} 必须在 1 与可用样本块数 {steps} 之间")
        for name in ("reg_act_num", "bat_act_num", "pv_act_num"):
            value = env.get(name, 33)
            if value != math.inf and (value < 2 or int(value) != value):
                raise ConfigError(f"env.{name} 必须为 >= 2 的整数或 .inf")
        family = environment_family(config.main_args["env"])
        sequential = config.main_args["algo"] == "sequential_ppo"
        if (family == "stackelberg") != sequential:
            raise ConfigError(
                "Stackelberg顺序非合作任务只使用sequential_ppo；同步协作任务不能使用该训练器"
            )
        if sequential:
            from envs.stackelberg.training import validate_settings

            validate_settings(config)
        if env.get("dss_act") and not algo["render"]["use_render"]:
            raise ConfigError(
                "env.dss_act=true 会启用 DSS 自动控制，使 RL 动作无效；训练时请关闭"
            )
