"""三馈线VVC物理、双worker、训练和检查点实际回放验收。"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_logged(command, path):
    with path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=240,
        )
    content = path.read_text(encoding="utf-8")
    assert (
        "Traceback (most recent call last)" not in content
        and "| ERROR |" not in content
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    from scripts.check_smartgrid import check_native, check_spawn

    output = ROOT / "results/validation/vvc"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for system in ("13Bus", "34Bus", "123Bus"):
        profile = f"configs/experiments/vvc_{system.lower()}_smoke.yaml"
        entry = {
            "native": check_native(profile, steps=24),
            "spawn": check_spawn(profile),
        }
        if args.train:
            import numpy as np
            import torch

            before = set((output / "runs").rglob("*.pt"))
            devices = (
                ["cpu", "cuda"] if args.require_cuda and system == "13Bus" else ["cpu"]
            )
            for device in devices:
                run_logged(
                    [
                        "scripts/train.py",
                        "--config",
                        profile,
                        "--exp-name",
                        f"check_{system}_{device}",
                        "--set",
                        f"device.cuda={str(device=='cuda').lower()}",
                        "--set",
                        f"logger.log_dir={(output/'runs').as_posix()}",
                        "--set",
                        "eval.use_eval=true",
                        "--set",
                        "eval.n_eval_rollout_threads=1",
                        "--set",
                        "eval.eval_episodes=6",
                        "--set",
                        "train.eval_interval=5",
                    ],
                    output / f"{system}_{device}.log",
                )
                entry["happo_" + device] = {
                    "steps": 240,
                    "updates": 10,
                    "validation_episodes": 12,
                }
            created = set((output / "runs").rglob("*.pt")) - before
            assert created
            for path in created:
                state = torch.load(path, map_location="cpu", weights_only=True)

                def finite(value):
                    if isinstance(value, torch.Tensor):
                        assert torch.isfinite(value).all(), path
                    elif isinstance(value, dict):
                        for item in value.values():
                            finite(item)

                finite(state)
            from envs.powergrid.checkpoint import load_checkpoint
            from envs.vvc.render.engine.episode_runner import VVCEpisodeRunner

            checkpoint = max(
                (p.parent for p in created if p.name == "actor_agent0.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            engine = load_checkpoint(checkpoint, expected_family="vvc")
            runner = VVCEpisodeRunner(engine.config, engine, project_root=str(ROOT))
            try:
                episode = runner.run_episode(seed=5, max_steps=24)
                assert episode.episode_length == 24
                assert np.isclose(
                    episode.total_reward,
                    sum(s["info"]["reward_total"] for s in episode.snapshots[1:]),
                    rtol=1e-6,
                )
                entry["happo_replay"] = {
                    "actors": engine.n_agents,
                    "steps": 24,
                    "finite_model_files": len(created),
                }
            finally:
                runner.close()
            model = output / f"{system}_ppo"
            run_logged(
                [
                    "scripts/train_centralized.py",
                    "--config",
                    profile,
                    "--steps",
                    "96",
                    "--output",
                    str(model),
                ],
                output / f"{system}_sb3.log",
            )
            from utils.config_pipeline import ConfigLoader
            from utils.envs_tools import make_power_env
            from envs.powergrid.adapters import CentralizedAdapter
            from utils.centralized_cli import load_model

            cfg = ConfigLoader().load(experiment=model.with_suffix(".environment.yaml"))
            wrapper = make_power_env("vvc", cfg.env_args, 0, cfg.env_args["seed"])
            central = CentralizedAdapter(wrapper.env)
            try:
                policy = load_model(model, central)
                obs, _ = central.reset(seed=7)
                for _ in range(24):
                    action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = central.step(action)
                    assert np.isfinite(reward) and info["dss_converged"]
                assert done and not truncated
                entry["sb3_ppo"] = {"steps": 96, "updates": 4, "replay_steps": 24}
            finally:
                central.close()
        report[system] = entry
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("VVC_CASE_OK", system, flush=True)
    print("VVC_RUNTIME_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
