"""从训练快照及真实空间恢复策略，拒绝猜测观测、动作或网络尺寸。"""

from pathlib import Path


def _validate_replay_algo(cfg, family, expected_family, allow_mappo, read_snapshot):
    if type(allow_mappo) is not bool:
        raise ValueError("allow_mappo must be a boolean")
    if allow_mappo and (expected_family != "vvc" or family != "vvc"):
        raise ValueError("MAPPO benchmark opt-in is only available for explicit VVC")
    algo = cfg.main_args["algo"]
    if algo == "happo":
        return
    if algo != "mappo" or not allow_mappo:
        raise ValueError("默认回放仅支持 HAPPO；MAPPO 需要显式 VVC benchmark opt-in")
    actual = cfg.algo_args
    frozen = read_snapshot()
    if not isinstance(frozen, dict) or frozen.get("config_kind") != "resolved":
        raise ValueError("MAPPO requires a resolved checkpoint configuration")
    saved_algo = frozen.get("algo")
    if (
        not isinstance(saved_algo, dict)
        or saved_algo.get("share_param") is not False
        or actual["algo"].get("share_param") is not False
    ):
        raise ValueError("MAPPO VVC replay requires explicit share_param: false")
    saved_model = frozen.get("model")
    resolved_model = actual["model"]
    if (
        not isinstance(saved_model, dict)
        or saved_model.keys() != resolved_model.keys()
        or any(
            type(value) is not type(saved_model[key]) or value != saved_model[key]
            for key, value in resolved_model.items()
        )
        or any(
            type(saved_model.get(key)) is not bool
            for key in ("use_recurrent_policy", "use_naive_recurrent_policy")
        )
        or type(saved_algo.get("use_policy_active_masks")) is not bool
        or saved_algo["use_policy_active_masks"]
        is not actual["algo"].get("use_policy_active_masks")
    ):
        raise ValueError("MAPPO policy model flags must be explicit in the checkpoint")


def load_checkpoint(
    checkpoint_dir, device="cpu", expected_family=None, *, allow_mappo=False
):
    from envs.render_common.engine.inference_engine import InferenceEngine
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import (
        build_dataclass,
        environment_class,
        environment_family,
    )
    from utils.environment_registry import environment_spec
    from utils.physics_manifest import validate_checkpoint
    import yaml

    directory = Path(checkpoint_dir).resolve()
    snapshot = next(
        (
            p / "config.resolved.yaml"
            for p in (directory, directory.parent, directory.parent.parent)
            if (p / "config.resolved.yaml").is_file()
        ),
        None,
    )
    if snapshot is None:
        raise ValueError("模型缺少config.resolved.yaml；不能猜测策略尺寸或环境配置")
    cfg = ConfigLoader().load(experiment=snapshot)
    family = environment_family(cfg.main_args["env"])
    if family not in ("vvc", "smartgrid", "district_dispatch", "dsr") or (
        expected_family and family != expected_family
    ):
        raise ValueError("检查点任务与当前回放入口不一致")
    _validate_replay_algo(
        cfg,
        family,
        expected_family,
        allow_mappo,
        lambda: yaml.safe_load(snapshot.read_text(encoding="utf-8")),
    )
    typed = build_dataclass(environment_class(family), cfg.env_args)
    wrapper = environment_spec(family).create(typed, rank=0)
    env = wrapper.env
    try:
        manifest = env.get_physics_manifest()
        validate_checkpoint(directory, manifest)
        spaces = env.agent_spaces
        sizes = [1 if hasattr(s, "n") else s.shape[0] for s in spaces]
        engine = InferenceEngine(
            str(directory),
            len(spaces),
            [space.shape[0] for space in wrapper.observation_space],
            sizes,
            hidden_size=cfg.algo_args["model"]["hidden_sizes"][-1],
            device=device,
        )
        engine.load_actors(
            {**cfg.algo_args["model"], **cfg.algo_args["algo"]}, action_spaces=spaces
        )
        engine.physics_manifest = manifest
        engine.config = env.config
        return engine
    finally:
        env.close()
