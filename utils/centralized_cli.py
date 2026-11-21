"""集中控制/单智能体训练入口：可插拔后端（sb3_ppo | ldcppo | …）。

后端注册表见 utils/single_agent/backends.py；集中控制算法本体在
algorithms/centralized/（CENTRALIZED_ALGOS）。本入口统一承担：
配置加载（ConfigLoader）、启动准备（prepare_run）、环境构造
（make_power_env；cbess 家族直接用注册包装器，其余家族过
CentralizedAdapter）与伴随文件（.environment.yaml；物理清单仅当
环境暴露 .core 时写出）。算法语义全部在后端内。
"""

from pathlib import Path
import argparse


def build_env(family: str, cfg):
    """按家族构造 gymnasium 风格单智能体环境。"""
    from utils.envs_tools import make_power_env
    from envs.powergrid.adapters import CentralizedAdapter

    wrapper = make_power_env(family, cfg.env_args, 0, cfg.env_args["seed"])
    if family == "cbess":
        return wrapper  # CBESSTaskEnv：gymnasium 接口 + info["costs"]
    action_mode = "discrete" if family == "dsr" else "continuous"
    return CentralizedAdapter(wrapper.env, action_space_type=action_mode)


def main(
    default_experiment="configs/experiments/vvc_13bus_smoke.yaml",
    expected_family=None,
):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=default_experiment)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--output", default=None,
                   help="模型/日志输出前缀（默认 results/centralized/<backend>）")
    p.add_argument("--backend", default="sb3_ppo",
                   help="训练后端（默认 sb3_ppo，原基线行为；可选 ldcppo 等）")
    p.add_argument("--resume", action="store_true",
                   help="断点续训（支持 episode 驱动后端，如 ldcppo）")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps 必须为正")

    from utils.single_agent import BACKENDS
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import environment_family
    from utils.training_setup import prepare_run

    if args.backend not in BACKENDS:
        p.error(f"未知后端 {args.backend}；可用：{sorted(BACKENDS)}")
    backend = BACKENDS[args.backend]

    cfg = ConfigLoader().load(experiment=args.config)
    family = environment_family(cfg.main_args["env"])
    if family not in ("vvc", "smartgrid", "district_dispatch", "dsr", "cbess") or (
        expected_family and family != expected_family
    ):
        p.error("当前集中控制入口支持 VVC/SmartGrid/District/DSR/CBESS 任务")
    prepare_run(cfg)
    env = build_env(family, cfg)
    try:
        output = Path(
            args.output or f"results/centralized/{args.backend}"
        ).resolve()
        backend(cfg, env, output, args.steps, resume=args.resume)
        cfg.save(output.with_suffix(".environment.yaml"))
        core = getattr(env, "core", None) or getattr(
            getattr(env, "env", None), "core", None
        )
        if core is not None and hasattr(core, "get_physics_manifest"):
            import json

            output.with_suffix(".physics.json").write_text(
                json.dumps(
                    core.get_physics_manifest(),
                    ensure_ascii=False, indent=2, allow_nan=False,
                ),
                encoding="utf-8",
            )
    finally:
        env.close()


def load_model(model_path, env):
    """先验证训练时的设备、观测及任务绑定，再恢复集中策略。"""
    from stable_baselines3 import PPO
    from utils.physics_manifest import validate_sidecar

    validate_sidecar(model_path, env.core.get_physics_manifest())
    return PPO.load(model_path, env=env, device="cpu")
