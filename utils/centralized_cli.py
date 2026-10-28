"""已验收同步运行任务的SB3 PPO基线，共用正式配置和物理内核。"""

from pathlib import Path
import argparse


def main(
    default_experiment="configs/experiments/vvc_13bus_smoke.yaml", expected_family=None
):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=default_experiment)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--output", default="results/centralized/ppo")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps 必须为正数")
    from stable_baselines3 import PPO
    from utils.config_pipeline import ConfigLoader
    from utils.training_setup import prepare_run
    from utils.envs_tools import make_power_env
    from envs.powergrid.adapters import CentralizedAdapter

    cfg = ConfigLoader().load(experiment=args.config)
    family = cfg.main_args["env"]
    if family not in ("vvc", "smartgrid") or (
        expected_family and family != expected_family
    ):
        p.error("当前集中控制基线仅支持已校准的VVC/SmartGrid任务")
    prepare_run(cfg)
    wrapper = make_power_env(family, cfg.env_args, 0, cfg.env_args["seed"])
    env = CentralizedAdapter(wrapper.env)
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
        model.learn(total_timesteps=args.steps)
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        cfg.save(output.with_suffix(".environment.yaml"))
        import json

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
                    "n_steps": n_steps,
                    "batch_size": n_steps,
                    "n_epochs": 1,
                    "learning_rate": model.learning_rate,
                    "gamma": model.gamma,
                    "gae_lambda": model.gae_lambda,
                    "device": "cpu",
                    "action_adapter": "Box with discrete-device quantization",
                },
                indent=2,
            )
        )
    finally:
        env.close()


def load_model(model_path, env):
    """先验证训练时的设备、观测及任务绑定，再恢复集中策略。"""
    from stable_baselines3 import PPO
    from utils.physics_manifest import validate_sidecar

    validate_sidecar(model_path, env.core.get_physics_manifest())
    return PPO.load(model_path, env=env, device="cpu")
