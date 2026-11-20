"""已验收同步运行任务的SB3 PPO基线，共用正式配置和物理内核。"""

from pathlib import Path
import argparse
import json
from hashlib import sha256
from time import monotonic


def main(
    default_experiment="configs/experiments/vvc_13bus_smoke.yaml", expected_family=None
):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=default_experiment)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--output", default="results/centralized/ppo")
    p.add_argument("--action-space", choices=("continuous", "discrete"), default=None,
                   help="显式选择动作编码；默认保留各任务原行为")
    p.add_argument("--torch-threads", type=int, default=None)
    args = p.parse_args()
    if args.torch_threads is not None and args.torch_threads < 1:
        p.error("--torch-threads 必须为正数")
    if args.steps < 1:
        p.error("--steps 必须为正数")
    from stable_baselines3 import PPO
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import environment_family
    from utils.training_setup import prepare_run
    from utils.envs_tools import make_power_env
    from envs.powergrid.adapters import CentralizedAdapter

    cfg = ConfigLoader().load(experiment=args.config)
    family = environment_family(cfg.main_args["env"])
    if family not in ("vvc", "smartgrid", "district_dispatch", "dsr") or (
        expected_family and family != expected_family
    ):
        p.error("当前集中控制基线支持VVC/SmartGrid/District/DSR同步任务")
    prepare_run(cfg)
    action_mode = args.action_space or ("discrete" if family == "dsr" else "continuous")
    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)
    if family == "vvc":
        from envs.vvc.single_agent import SingleAgentVVCEnv
        from envs.vvc.vvc.vvc_config import VVCConfig
        env = SingleAgentVVCEnv(VVCConfig.from_env_args(cfg.env_args), action_mode)
    else:
        wrapper = make_power_env(family, cfg.env_args, 0, cfg.env_args["seed"])
        env = CentralizedAdapter(wrapper.env, action_space_type=action_mode)
    try:
        n_steps = max(2, int(cfg.algo_args["train"]["episode_length"]))
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=n_steps,
            n_epochs=1,
            learning_rate=cfg.algo_args["model"]["lr"],
            gamma=cfg.algo_args["algo"]["gamma"],
            gae_lambda=cfg.algo_args["algo"]["gae_lambda"],
            policy_kwargs={"net_arch": cfg.algo_args["model"]["hidden_sizes"]},
            device="cpu",
            seed=cfg.algo_args["seed"]["seed"],
            verbose=1,
        )
        import torch
        initial_parameters = {k:v.detach().clone() for k,v in model.policy.state_dict().items()}
        optimizer_steps = []
        hook = model.policy.optimizer.register_step_post_hook(
            lambda *unused: optimizer_steps.append(1)
        )
        started = monotonic()
        try:
            model.learn(total_timesteps=args.steps)
        finally:
            hook.remove()
        training_seconds = monotonic() - started
        parameters_changed = any(not torch.equal(v, initial_parameters[k])
                                 for k,v in model.policy.state_dict().items())
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        cfg.save(output.with_suffix(".environment.yaml"))

        output.with_suffix(".physics.json").write_text(
            json.dumps(
                env.core.get_physics_manifest(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        output.with_suffix(".sb3.json").write_text(
            json.dumps(
                {
                    "algorithm": "SB3 PPO",
                    "steps": model.num_timesteps,
                    "requested_steps": args.steps,
                    "training_seconds": training_seconds,
                    "optimizer_updates": model._n_updates,
                    "optimizer_steps": len(optimizer_steps),
                    "policy_parameters_changed": parameters_changed,
                    "training_seed": cfg.algo_args["seed"]["seed"],
                    "action_space_type": action_mode,
                    "action_nvec": env.action_space.nvec.tolist() if action_mode == "discrete" else None,
                    "files_sha256": {output.with_suffix(suffix).name:
                                     sha256(output.with_suffix(suffix).read_bytes()).hexdigest()
                                     for suffix in (".zip", ".environment.yaml", ".physics.json")},
                    "n_steps": n_steps,
                    "batch_size": n_steps,
                    "n_epochs": 1,
                    "learning_rate": model.learning_rate,
                    "gamma": model.gamma,
                    "gae_lambda": model.gae_lambda,
                    "device": "cpu",
                    "action_adapter": (
                        "MultiDiscrete device choices"
                        if action_mode == "discrete"
                        else "Box with discrete-device quantization"
                    ),
                },
                indent=2,
            )
        )
        restored = load_model(
            output, env,
            require_vvc_binding=(family == "vvc" and args.action_space == "discrete"),
        )
        if not all(torch.equal(v, restored.policy.state_dict()[k])
                   for k,v in model.policy.state_dict().items()):
            raise RuntimeError("PPO save/reload parameters differ")
        metadata_path = output.with_suffix(".sb3.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["reload_parameters_equal"] = True
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    finally:
        env.close()


def load_model(model_path, env, *, require_vvc_binding=False):
    """默认保持原物理绑定恢复；显式 VVC 评测另要求完整 SB3 工件绑定。"""
    from utils.physics_manifest import validate_sidecar

    if type(require_vvc_binding) is not bool:
        raise ValueError("require_vvc_binding must be a boolean")
    model_path = Path(model_path)
    manifest = env.core.get_physics_manifest()
    if require_vvc_binding:
        if manifest.get("task") != "vvc" or env.mode != "discrete":
            raise ValueError("strict VVC binding requires a discrete VVC environment")
        try:
            metadata = json.loads(model_path.with_suffix(".sb3.json").read_text())
        except OSError as exc:
            raise ValueError("VVC 模型缺少 .sb3.json 工件绑定") from exc
        if not isinstance(metadata, dict) or (
            metadata.get("algorithm") != "SB3 PPO"
            or metadata.get("action_space_type") != "discrete"
            or metadata.get("action_nvec") != env.action_space.nvec.tolist()
        ):
            raise ValueError("VVC 模型缺少匹配的显式 MultiDiscrete 动作绑定")
        expected_files = {
            model_path.with_suffix(suffix).name
            for suffix in (".zip", ".environment.yaml", ".physics.json")
        }
        hashes = metadata.get("files_sha256")
        if not isinstance(hashes, dict) or set(hashes) != expected_files:
            raise ValueError("VVC files_sha256 必须完整绑定归档、resolved 配置和物理 sidecar")
        for name, digest in hashes.items():
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("VVC 工件 SHA256 格式无效")
            try:
                actual = sha256((model_path.parent / name).read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError("VVC 绑定工件缺失或不可读") from exc
            if actual != digest:
                raise ValueError("VVC 模型或配置/物理 sidecar SHA256 不一致")
        import yaml
        resolved = yaml.safe_load(model_path.with_suffix(".environment.yaml").read_text())
        if not isinstance(resolved, dict) or resolved.get("config_kind") != "resolved":
            raise ValueError("VVC 绑定配置必须是训练保存的 resolved 快照")
    validate_sidecar(model_path, manifest)
    # Deserialize only after all selected binding checks pass.
    from stable_baselines3 import PPO
    return PPO.load(model_path, env=env, device="cpu")
