"""集中控制训练与评估，支持 SB3 PPO/SAC/TD3 和 LDCPPO。"""

import argparse
import json
from pathlib import Path

from envs.powergrid.adapters import CentralizedAdapter


class SingleAgentAdapter(CentralizedAdapter):
    """保留设备动作映射，向 Gymnasium 转发核心的回合边界。"""

    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        truncated = bool(info.get("truncated", info.get("TimeLimit.truncated", False)))
        terminated = bool(info.get("terminated", done and not truncated))
        return obs, reward, terminated, truncated, info


def build_env(family: str, cfg):
    """按家族构造 Gymnasium 单智能体环境。"""
    from utils.envs_tools import make_power_env

    wrapper = make_power_env(family, cfg.env_args, 0, cfg.env_args["seed"])
    if family == "cbess":
        return wrapper
    action_mode = "discrete" if family == "dsr" else "continuous"
    return SingleAgentAdapter(wrapper.env, action_space_type=action_mode)


def evaluate(model, env, episodes, seed):
    """按环境边界重置并逐回合记录约束成本和回报。"""
    from utils.single_agent.costs import extract_costs

    results = []
    for index in range(episodes):
        obs, reset_info = env.reset(seed=seed if index == 0 else None)
        reward_sum, steps, costs = 0.0, 0, {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            steps += 1
            if hasattr(env, "scenario_sha256"):
                for key, value in extract_costs(info).items():
                    costs[key] = costs.get(key, 0.0) + value
            if terminated or truncated:
                break
        results.append(
            {
                **{
                    key: reset_info[key]
                    for key in ("day_key", "scenario_id")
                    if key in reset_info
                },
                "steps": steps,
                "reward": reward_sum,
                "costs": costs,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
    return results


def main(
    default_experiment="configs/experiments/vvc_13bus_smoke.yaml",
    expected_family=None,
):
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import environment_family
    from utils.single_agent import BACKENDS
    from utils.single_agent.backends import validate_backend
    from utils.single_agent.binding import (
        checkpoint_path,
        make_binding,
        physics_manifest,
    )
    from utils.training_setup import prepare_run

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default=default_experiment, help="实验或已解析的 .environment.yaml"
    )
    p.add_argument(
        "--steps", type=int, default=240, help="训练总步数（LDCPPO 含已完成步数）"
    )
    p.add_argument("--output", default=None, help="模型前缀（LDCPPO 为运行目录）")
    p.add_argument("--backend", default="sb3_ppo", choices=sorted(BACKENDS))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="从 LDCPPO 断点继续")
    mode.add_argument(
        "--evaluate", action="store_true", help="评估 --output 指定的 checkpoint"
    )
    p.add_argument("--split", choices=("train", "validation"), help="CBESS 剖面划分")
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--eval-scenario-sha256", help="CBESS validation 使用的剖面摘要")
    args = p.parse_args()
    if args.steps < 1 or args.eval_episodes < 1:
        p.error("--steps 和 --eval-episodes 必须为正")
    if args.resume and args.backend != "ldcppo":
        p.error(f"{args.backend} 不支持 --resume")
    if args.evaluate and args.output is None:
        p.error("--evaluate 需要显式指定 --output")
    if args.eval_scenario_sha256 and not args.evaluate:
        p.error("--eval-scenario-sha256 仅用于评估")

    loader = ConfigLoader()
    cfg = loader.load(experiment=args.config)
    family = environment_family(cfg.main_args["env"])
    if family not in ("vvc", "smartgrid", "district_dispatch", "dsr", "cbess") or (
        expected_family and family != expected_family
    ):
        p.error("当前集中控制入口支持 VVC/SmartGrid/District/DSR/CBESS 任务")
    try:
        validate_backend(args.backend, family)
    except ValueError as exc:
        p.error(str(exc))
    if args.split is not None:
        if family != "cbess":
            p.error("--split 仅用于 CBESS")
        cfg = loader.load(experiment=args.config, overrides={"env.split": args.split})
    data = cfg.env_args.get("smartds_series") or cfg.env_args.get("data", {})
    split = cfg.env_args["split"] if family == "cbess" else data.get("split", "train")
    if split not in ("train", "validation") or (not args.evaluate and split != "train"):
        p.error("训练/续训只使用 train，评估可使用 train 或 validation")
    if family == "cbess" and split == "validation" and not args.eval_scenario_sha256:
        p.error("CBESS validation 需要 --eval-scenario-sha256")
    if (args.resume or args.evaluate) and not cfg.algo_args["seed"]["seed_specify"]:
        p.error("继续训练或评估请使用含实际种子的已解析配置")
    output = Path(args.output or f"results/centralized/{args.backend}").resolve()
    checkpoint = checkpoint_path(output, args.backend)
    if (args.resume or args.evaluate) and not checkpoint.is_file():
        p.error(f"checkpoint 不存在：{checkpoint}")
    prepare_run(cfg)
    env = build_env(family, cfg)
    try:
        validate_backend(args.backend, family, env)
        physics = physics_manifest(env)
        binding = make_binding(args.backend, cfg, env, physics)
        if args.evaluate:
            policy = load_model(
                output,
                env,
                backend=args.backend,
                cfg=cfg,
                eval_scenario_sha256=args.eval_scenario_sha256,
            )
            report = {
                "backend": args.backend,
                "scenario": binding["scenario"],
                "environment_sha256": binding["environment_sha256"],
                "config_sha256": binding["config_sha256"],
                "episodes": evaluate(
                    policy, env, args.eval_episodes, cfg.env_args["seed"]
                ),
            }
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return
        # 首次保存使运行中断后也能从 checkpoint 对应的已解析配置恢复。
        config_path = output.with_suffix(".environment.yaml")
        if not args.resume and not checkpoint.exists():
            cfg.save(config_path)
        BACKENDS[args.backend](
            cfg, env, output, args.steps, resume=args.resume, binding=binding
        )
        cfg.save(config_path)
        output.with_suffix(".physics.json").write_text(
            json.dumps(
                {**physics, "single_agent_binding_version": binding["version"]},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    finally:
        env.close()


class _LDCPPOPolicy:
    def __init__(self, agent, space):
        self.agent = agent
        self.space = space

    def predict(self, obs, deterministic=True):
        import numpy as np

        action, _ = self.agent.act(obs, deterministic=deterministic)
        return np.clip(action, self.space.low, self.space.high), None


def load_model(
    model_path, env, *, backend="sb3_ppo", cfg=None, eval_scenario_sha256=None
):
    """校验绑定后加载集中控制策略，兼容已有 PPO 物理伴随文件。"""
    from utils.single_agent.backends import (
        load_ldcppo_state,
        make_ldcppo,
        sb3_class,
        validate_backend,
    )
    from utils.single_agent.binding import (
        binding_path,
        checkpoint_path,
        make_binding,
        physics_manifest,
        validate_binding,
        requires_binding,
    )

    checkpoint = checkpoint_path(model_path, backend)
    if backend == "sb3_ppo" and not binding_path(checkpoint).is_file():
        from utils.physics_manifest import validate_sidecar

        sidecar = Path(model_path).with_suffix(".physics.json")
        if (
            sidecar.is_file()
            and "single_agent_binding_version"
            in json.loads(sidecar.read_text(encoding="utf-8"))
        ) or (checkpoint.is_file() and requires_binding(checkpoint)):
            raise ValueError("集中模型缺少 checkpoint 绑定清单")
        if eval_scenario_sha256 is not None:
            raise ValueError("旧 PPO 物理伴随文件不能验证指定的场景摘要")
        validate_sidecar(model_path, physics_manifest(env))
        return sb3_class(backend).load(checkpoint, env=env, device="cpu")
    if cfg is None:
        from utils.config_pipeline import ConfigLoader

        snapshot = Path(model_path).with_suffix(".environment.yaml")
        cfg = ConfigLoader().load(experiment=snapshot)
    validate_backend(backend, cfg.main_args["env"], env)
    binding = make_binding(backend, cfg, env, physics_manifest(env))
    if backend == "ldcppo":
        state = load_ldcppo_state(
            checkpoint, binding, eval_scenario_sha256=eval_scenario_sha256
        )
        agent = make_ldcppo(cfg, env)
        agent.load_state_dict(state)
        return _LDCPPOPolicy(agent, env.action_space)
    validate_binding(checkpoint, binding, eval_scenario_sha256=eval_scenario_sha256)
    return sb3_class(backend).load(checkpoint, env=env, device="cpu")
