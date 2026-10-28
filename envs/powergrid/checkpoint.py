"""从训练快照及真实空间恢复策略，拒绝猜测观测、动作或网络尺寸。"""

from pathlib import Path

from envs.render_common.engine.inference_engine import InferenceEngine
from utils.config_pipeline import ConfigLoader
from utils.config_validation import build_dataclass
from utils.physics_manifest import validate_checkpoint


def load_checkpoint(checkpoint_dir, device="cpu", expected_family=None):
    from utils.config_validation import environment_class
    from utils.environment_registry import environment_spec

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
    family = cfg.main_args["env"]
    if family not in ("vvc", "smartgrid") or (
        expected_family and family != expected_family
    ):
        raise ValueError("检查点任务与当前回放入口不一致")
    if cfg.main_args["algo"] != "happo":
        raise ValueError("当前回放只验收了HAPPO策略结构")
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
            [env.observation_space.shape[0]] * len(spaces),
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
