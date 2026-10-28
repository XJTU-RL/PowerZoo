"""SmartGrid 原生、spawn、MARL/SB3 短训练验收；失败返回非零退出码。"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def check_native():
    import numpy as np
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env

    cfg = ConfigLoader().load(
        "happo", "smartgrid", values={"env": {"data": {"episodes_per_split": 6}}}
    )
    env = make_power_env("smartgrid", cfg.env_args, 0, 0)
    core, records = env.env, []
    try:
        a = [1] * core.cap_num
        a += [
            round(
                (r.initial - r.mintap) / (r.maxtap - r.mintap) * (core.reg_act_num - 1)
            )
            for r in core.circuit.regulators.values()
        ]
        a += [core.bat_act_num // 2] * core.bat_num
        a += [v for _ in core.pv_names for v in (1, 0)]
        for i in range(6):
            core.reset(load_profile_idx=i)
            rewards = []
            for _ in range(core.horizon):
                _, reward, done, info = core.step(a)
                assert info["dss_converged"] and np.isfinite(reward)
                rewards.append(reward)
            assert done
            records.append(
                dict(
                    scenario=info["scenario"],
                    steps=core.horizon,
                    mean_reward=float(np.mean(rewards)),
                    pv_kw=info["pv_power_kw"],
                    voltage_min_pu=info["voltage_min_pu"],
                    voltage_max_pu=info["voltage_max_pu"],
                )
            )
        return {
            "counts": {
                "capacitors": core.cap_num,
                "regulators": core.reg_num,
                "storage": core.bat_num,
                "pv": core.pv_num,
            },
            "dataset": str(core.dataset.path),
            "dataset_sha256": core.dataset.sha256,
            "scenarios": records,
        }
    finally:
        env.close()


def check_spawn():
    import numpy as np
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_train_env

    cfg = ConfigLoader().load(
        "happo", "smartgrid", values={"env": {"max_episode_steps": 24}}
    )
    env = make_train_env("smartgrid", 0, 2, cfg.env_args)
    try:
        obs, _, _ = env.reset()
        for _ in range(26):  # 覆盖自动 reset
            actions = np.zeros((2, len(env.action_space), 2), dtype=np.float32)
            for i, space in enumerate(env.action_space):
                if hasattr(space, "n"):
                    actions[:, i, 0] = space.n // 2
            obs, _, rewards, _, infos, _ = env.step(actions)
            assert np.isfinite(np.asarray(obs)).all() and np.isfinite(rewards).all()
            assert all(record[0]["dss_converged"] for record in infos)
        return {
            "workers": 2,
            "steps": 26,
            "observation_shape": list(np.asarray(obs).shape),
        }
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    report = {"native": check_native(), "spawn": check_spawn()}
    output = ROOT / "results/validation/smartgrid"
    output.mkdir(parents=True, exist_ok=True)
    if a.train:
        previous_models = set((output / "runs").rglob("*.pt"))
        for device in ["cpu", "cuda"] if a.require_cuda else ["cpu"]:
            evaluation = (
                [
                    "--set",
                    "eval.use_eval=true",
                    "--set",
                    "eval.n_eval_rollout_threads=1",
                    "--set",
                    "eval.eval_episodes=6",
                    "--set",
                    "train.eval_interval=5",
                ]
                if device == "cpu"
                else []
            )
            with (output / f"happo_{device}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "examples/multi_agent/scripts/train.py",
                        "--config",
                        "configs/experiments/smartgrid_synthetic_smoke.yaml",
                        "--exp-name",
                        f"check_{device}",
                        "--set",
                        f"device.cuda={'true' if device == 'cuda' else 'false'}",
                        "--set",
                        "logger.log_dir=results/validation/smartgrid/runs",
                        *evaluation,
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            report[f"happo_{device}"] = {"steps": 240, "updates": 10}
            if device == "cpu":
                report[f"happo_{device}"]["validation_episodes"] = 12
            assert "| ERROR |" not in (output / f"happo_{device}.log").read_text(
                encoding="utf-8"
            ), "训练日志报告了运行错误"
        with (output / "sb3.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "examples/single_agent/scripts/train_smartgrid.py",
                    "--steps",
                    "96",
                    "--output",
                    str(output / "sb3_ppo"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        report["sb3_ppo"] = {"steps": 96, "updates": 4}
        import torch

        checked = 0
        for path in set((output / "runs").rglob("*.pt")) - previous_models:
            state = torch.load(path, map_location="cpu", weights_only=True)

            def visit(obj):
                if isinstance(obj, torch.Tensor):
                    assert torch.isfinite(obj).all(), f"非有限模型参数：{path}"
                elif isinstance(obj, dict):
                    for value in obj.values():
                        visit(value)

            visit(state)
            checked += 1
        assert checked > 0, "训练没有保存模型"
        report["finite_model_files"] = checked
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("SMARTGRID_RUNTIME_OK " + json.dumps(report))


if __name__ == "__main__":
    main()
