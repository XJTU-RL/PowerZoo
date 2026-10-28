"""SmartGrid 原生、spawn、MARL/SB3 短训练验收；失败返回非零退出码。"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "configs/experiments/smartgrid_synthetic_smoke.yaml"


def check_native(experiment=DEFAULT_EXPERIMENT):
    import numpy as np
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env

    cfg = ConfigLoader().load(
        experiment=experiment,
        values={"env": {"max_episode_steps": 360, "data": {"episodes_per_split": 6}}},
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
        a += [0] * len(core.svg_names)
        for controller in ("fixed", "native_cap_reg"):
            for i in range(6):
                core.reset(load_profile_idx=i)
                samples = []
                for _ in range(core.horizon):
                    if controller == "fixed":
                        _, reward, done, info = core.step(a)
                        assert np.isfinite(reward)
                    else:
                        # 比较原生RegControl/CapControl；Storage空闲、PV最大可用有功、SVG零Q。
                        # 只采集物理指标，避免把未计算的任务奖励填为零。
                        c = core.circuit
                        c.begin_interval(float(core.profile[core.t, 0]))
                        core._apply_data(core.t)
                        c.apply({"Storage." + n: {"kw": 0} for n in c.batteries})
                        c.available_pv_power()
                        c.solve(native_controls=True)
                        c.finish_interval()
                        core.t += 1
                        core._observe()
                        info = core._info()
                        done = core.t == core.horizon
                    assert info["dss_converged"]
                    assert abs(info["active_balance_error_kw"]) < 0.05, {
                        "controller": controller,
                        "scenario": i,
                        "step": core.t,
                        "error_kw": info["active_balance_error_kw"],
                        "grid_kw": info["total_power_kw"],
                        "load_kw": info["total_load_kw"],
                        "loss_kw": info["power_loss_kw"],
                        "pv_kw": info["pv_power_kw"],
                    }
                    samples.append(info)
                assert done
                dt = core.config.data.step_seconds
                records.append(
                    dict(
                        controller=controller,
                        scenario=info["scenario"],
                        steps=core.horizon,
                        loss_kwh=sum(s["power_loss_kw"] for s in samples) * dt / 3600,
                        grid_import_kwh=sum(s["total_power_kw"] for s in samples)
                        * dt
                        / 3600,
                        pv_kwh=sum(s["pv_power_kw"] for s in samples) * dt / 3600,
                        phase_violation_minutes=sum(
                            s["voltage_violation_count"] for s in samples
                        )
                        * dt
                        / 60,
                        voltage_min_pu=min(s["voltage_min_pu"] for s in samples),
                        voltage_max_pu=max(s["voltage_max_pu"] for s in samples),
                        max_branch_loading_pu=max(
                            s["max_branch_loading_pu"] for s in samples
                        ),
                        max_balance_error_kw=max(
                            abs(s["active_balance_error_kw"]) for s in samples
                        ),
                        final_soc={n: b.soc for n, b in core.circuit.batteries.items()},
                    )
                )
        return {
            "counts": {
                "capacitors": core.cap_num,
                "regulators": core.reg_num,
                "storage": core.bat_num,
                "pv": core.pv_num,
                "svg": len(core.svg_names),
            },
            "dataset": str(core.dataset.path),
            "dataset_sha256": core.dataset.sha256,
            "scenarios": records,
        }
    finally:
        env.close()


def check_spawn(experiment=DEFAULT_EXPERIMENT):
    import numpy as np
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_train_env

    cfg = ConfigLoader().load(
        experiment=experiment, values={"env": {"max_episode_steps": 24}}
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
    p.add_argument("--config", default=DEFAULT_EXPERIMENT)
    p.add_argument("--output", default="results/validation/smartgrid")
    a = p.parse_args()
    report = {"native": check_native(a.config), "spawn": check_spawn(a.config)}
    output = (ROOT / a.output).resolve()
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
                        "scripts/train.py",
                        "--config",
                        a.config,
                        "--exp-name",
                        f"check_{device}",
                        "--set",
                        f"device.cuda={'true' if device == 'cuda' else 'false'}",
                        "--set",
                        f"logger.log_dir={(output / 'runs').as_posix()}",
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
                    "--config",
                    a.config,
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
        from envs.smartgrid.render.engine.checkpoint import load_checkpoint
        from envs.smartgrid.render.engine.episode_runner import SmartGridEpisodeRunner
        import numpy as np

        checkpoint = max(
            (
                p.parent
                for p in set((output / "runs").rglob("actor_agent0.pt"))
                - previous_models
            ),
            key=lambda p: p.stat().st_mtime,
        )
        engine = load_checkpoint(checkpoint)
        runner = SmartGridEpisodeRunner(engine.config, engine, project_root=str(ROOT))
        try:
            episode = runner.run_episode(seed=5, max_steps=24)
            assert episode.episode_length == 24 and len(episode.snapshots) == 25
            assert np.isclose(
                episode.total_reward,
                sum(s["info"]["reward_total"] for s in episode.snapshots[1:]),
                rtol=1e-6,
            )
            report["checkpoint_replay"] = {
                "steps": 24,
                "actors": engine.n_agents,
                "reward": episode.total_reward,
            }
        finally:
            runner.close()
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("SMARTGRID_RUNTIME_OK " + json.dumps(report))


if __name__ == "__main__":
    main()
