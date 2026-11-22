"""集中控制 checkpoint 与实际消费的配置、场景和接口绑定。"""

import json
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from math import inf
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from gymnasium import spaces

from utils.config_validation import environment_family


def _binding_value(value):
    if isinstance(value, dict):
        return {key: _binding_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_binding_value(item) for item in value]
    if isinstance(value, float) and value == inf:
        # 配置用 .inf 表示原生连续动作。其余数值仍按严格 JSON 校验。
        return {"positive_infinity": True}
    return value


def fingerprint(value):
    return sha256(
        json.dumps(_binding_value(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _space(space):
    if isinstance(space, spaces.Box):
        return {
            "kind": "Box",
            "shape": list(space.shape),
            "dtype": space.dtype.str,
            "bounds": fingerprint(
                [space.low.astype(str).tolist(), space.high.astype(str).tolist()]
            ),
        }
    if isinstance(space, spaces.MultiDiscrete):
        return {
            "kind": "MultiDiscrete",
            "choices": space.nvec.tolist(),
            "start": space.start.tolist(),
        }
    if isinstance(space, spaces.Discrete):
        return {"kind": "Discrete", "choices": int(space.n), "start": int(space.start)}
    raise ValueError(f"集中控制不支持的空间：{space}")


def action_semantics(env):
    if isinstance(env.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
        return "discrete"
    agents = getattr(getattr(env, "core", None), "agent_spaces", ())
    if isinstance(env.action_space, spaces.Box):
        if any(not isinstance(space, spaces.Box) for space in agents):
            return "quantized_box"
        return "continuous_box"
    raise ValueError(f"集中控制不支持的动作空间：{env.action_space}")


def physics_manifest(env):
    return getattr(env, "core", env).get_physics_manifest()


def _scenario(env, environment):
    if hasattr(env, "scenario_sha256"):
        return {"split": env.split, "sha256": env.scenario_sha256}
    dataset = env.core.dataset
    data = environment.get("smartds_series") or environment["data"]
    return {"split": data["split"], "sha256": dataset.sha256}


def make_binding(backend, cfg, env, physics):
    environment = asdict(getattr(env, "core", env).config)
    scenario = _scenario(env, environment)
    resolved = cfg.to_dict()
    compatible = deepcopy(resolved)
    if cfg.main_args["env"] == "cbess":
        compatible["env"].pop("split", None)
        environment.pop("split", None)
    eval_config_sha256 = fingerprint(compatible)
    # 继续跑同一训练时可延长目标回合数或调整保存频率。
    for key in ("total_episodes", "checkpoint_every"):
        compatible["train"].pop(key, None)
    # seed 影响采样和优化器，按 resolved config 校验。评估允许 reset(seed=...)。
    for key in ("seed", "worker_idx"):
        environment.pop(key, None)
    return {
        "version": 1,
        "backend": backend,
        "family": cfg.main_args["env"],
        "observation": _space(env.observation_space),
        "action": {"semantics": action_semantics(env), **_space(env.action_space)},
        "physics_sha256": fingerprint(physics),
        "scenario": scenario,
        "environment_sha256": fingerprint(environment),
        "config_sha256": fingerprint(resolved),
        "eval_config_sha256": eval_config_sha256,
        "resume_config_sha256": fingerprint(compatible),
    }


def checkpoint_path(output, backend):
    output = Path(output)
    if backend == "ldcppo":
        return output if output.suffix == ".pt" else output / "checkpoint.pt"
    return output if output.suffix else output.with_suffix(".zip")


# 标记随模型原子写入；外部伴随文件丢失或残留时，不能把新策略当老 PPO。
_SB3_BINDING_MEMBER = "powerzoo/binding_version"


def save_bound_sb3(path, model):
    with path.open("wb") as stream:
        model.save(stream)
    with ZipFile(path, "a") as archive:
        archive.writestr(_SB3_BINDING_MEMBER, "1")


def requires_binding(checkpoint):
    try:
        with ZipFile(checkpoint) as archive:
            return _SB3_BINDING_MEMBER in archive.namelist()
    except BadZipFile as exc:
        raise ValueError("无效的 SB3 checkpoint") from exc


def binding_path(checkpoint):
    return checkpoint.with_name(checkpoint.name + ".binding.json")


def atomic_write(path, write):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_binding(checkpoint, binding):
    payload = {**binding, "checkpoint_sha256": _file_sha256(checkpoint)}
    atomic_write(
        binding_path(checkpoint),
        lambda path: path.write_text(json.dumps(payload, indent=2), encoding="utf-8"),
    )


def validate_binding(checkpoint, current, *, resume=False, eval_scenario_sha256=None):
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint 不存在：{checkpoint}")
    sidecar = binding_path(checkpoint)
    if not sidecar.is_file():
        raise ValueError(f"checkpoint 缺少绑定清单：{sidecar}")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if saved.get("checkpoint_sha256") != _file_sha256(checkpoint):
        raise ValueError("checkpoint 文件摘要与绑定清单不一致")
    validate_metadata(
        saved, current, resume=resume, eval_scenario_sha256=eval_scenario_sha256
    )


def validate_metadata(saved, current, *, resume=False, eval_scenario_sha256=None):
    if not isinstance(saved, dict):
        raise TypeError("checkpoint 缺少绑定元数据")
    source, target = saved.get("scenario"), current["scenario"]
    if environment_family(current["family"]) in ("vvc", "dsr") and source != target:
        raise ValueError(
            "VVC/DSR binding_v1 仅支持同场景重载；指定验证摘要不能替换训练场景"
        )
    mode_specific = "eval_config_sha256" if resume else "resume_config_sha256"
    for key, value in current.items():
        if key in ("config_sha256", "scenario", mode_specific):
            continue
        if saved.get(key) != value:
            raise ValueError(f"checkpoint 的 {key} 与当前运行不一致")
    if saved.get("family") == "cbess" and "validation" == target["split"]:
        if resume or not source or source.get("split") != "train":
            raise ValueError("CBESS validation 仅用于评估 train checkpoint")
        if target["sha256"] != eval_scenario_sha256:
            raise ValueError("CBESS validation 需要匹配的 --eval-scenario-sha256")
    elif source != target or (
        eval_scenario_sha256 and target["sha256"] != eval_scenario_sha256
    ):
        raise ValueError("checkpoint 的场景划分或摘要与当前运行不一致")
