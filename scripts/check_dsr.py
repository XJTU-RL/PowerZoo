"""小网络/IEEE13恢复上限、规则全轨迹、Windows worker及真实训练回放验收。"""

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def core_for(profile, split="train"):
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass, environment_class
    from envs.dsr.core.dsr_core import DSRCoreEnv

    cfg = ConfigLoader().load(
        experiment=profile, values={"env": {"data": {"split": split}}}
    )
    return DSRCoreEnv(
        build_dataclass(environment_class(cfg.main_args["env"]), cfg.env_args)
    )


def greedy_frame(core):
    """每次至多一个实际操作；使用零时间AC试算，不推进原生能量。"""
    import numpy as np

    best_frame = [0] * len(core.agent_spaces)
    score = lambda info: sum(
        m["qualified_served_kw"] * m["priority"] for m in info["loads"].values()
    )
    best = score(core._info())
    for agent, mask in enumerate(core.available_actions()):
        for action in np.flatnonzero(mask)[1:]:
            element, value = core.choices[agent][action]
            trial = core.evaluate_positions({element: value})
            candidate = score(trial)
            if candidate > best + 1e-4:
                best = candidate
                best_frame = [0] * len(core.agent_spaces)
                best_frame[agent] = int(action)
    return best_frame


def check_native(profile):
    import numpy as np

    records = []
    for split in ("train", "validation", "test"):
        core = core_for(profile, split)
        try:
            for episode in range(len(core.dataset.episodes)):
                core.reset(load_profile_idx=episode)
                assert len(core.controls) <= 12, "精确参考只适用于已声明的小规模动作集"
                upper = 0.0
                for values in product((False, True), repeat=len(core.controls)):
                    trial = core.evaluate_positions(dict(zip(core.controls, values)))
                    if trial is not None:
                        upper = max(upper, trial["restored_ratio"])
                assert core.circuit.time_s == 0
                for policy in ("wait", "greedy_ac"):
                    core.reset(load_profile_idx=episode)
                    samples = []
                    total_reward = 0.0
                    for _ in range(core.horizon):
                        action = (
                            greedy_frame(core)
                            if policy == "greedy_ac"
                            else [0] * len(core.agent_spaces)
                        )
                        _, reward, done, info = core.step(action)
                        assert np.isfinite(reward) and info["dss_converged"]
                        assert abs(info["active_balance_error_kw"]) < 0.05
                        assert info["restored_ratio"] <= upper + 1e-6
                        assert all(
                            not core.circuit.connection(e)["enabled"]
                            for e in core.unavailable
                        )
                        total_reward += reward
                        samples.append(info)
                    assert (
                        done
                        and core.circuit.time_s
                        == core.horizon * core.config.data.step_seconds
                    )
                    records.append(
                        {
                            "split": split,
                            "scenario": core.entry["id"],
                            "controller": policy,
                            "steps": core.horizon,
                            "clock_end_s": core.circuit.time_s,
                            "reward": total_reward,
                            "unserved_kwh": sum(s["unserved_kwh"] for s in samples),
                            "weighted_service_unserved_kwh": sum(
                                s["weighted_service_unserved_kwh"] for s in samples
                            ),
                            "operations": sum(s["operations"] for s in samples),
                            "final_restored_ratio": samples[-1]["restored_ratio"],
                            "static_ac_restoration_upper": upper,
                            "voltage_min_pu": min(s["voltage_min_pu"] for s in samples),
                            "max_loading_pu": max(
                                s["max_branch_loading_pu"] for s in samples
                            ),
                            "max_balance_error_kw": max(
                                abs(s["active_balance_error_kw"]) for s in samples
                            ),
                        }
                    )
            manifest = core.get_physics_manifest()
            dimensions = [int(s.n) for s in core.agent_spaces]
            obs = core.observation_space.shape[0]
        finally:
            core.close()
    return {
        "model_sha256": manifest["model_sha256"],
        "observation_dim": obs,
        "action_sizes": dimensions,
        "cases": records,
    }


def check_training(profile, output, cuda):
    import numpy as np
    import torch
    from scripts.check_vvc import run_logged
    from envs.powergrid.checkpoint import load_checkpoint
    from envs.dsr.render.engine.episode_runner import DSREpisodeRunner

    began = time.time_ns()
    report = {}
    for device in ["cpu", "cuda"] if cuda else ["cpu"]:
        run_logged(
            [
                "scripts/train.py",
                "--config",
                profile,
                "--exp-name",
                "dsr_" + device,
                "--set",
                "device.cuda=" + str(device == "cuda").lower(),
                "--set",
                "logger.log_dir=" + (output / "runs").as_posix(),
                "--set",
                "eval.use_eval=true",
                "--set",
                "eval.eval_episodes=4",
                "--set",
                "train.eval_interval=5",
            ],
            output / f"happo_{device}.log",
        )
        report["happo_" + device] = {
            "steps": 240,
            "updates": 10,
            "validation_episodes": 8,
        }
    created = [
        p for p in (output / "runs").rglob("*.pt") if p.stat().st_mtime_ns >= began
    ]
    assert created
    for path in created:

        def finite(value):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), path
            elif isinstance(value, dict):
                for child in value.values():
                    finite(child)

        finite(torch.load(path, map_location="cpu", weights_only=True))
    checkpoint = max(
        (p.parent for p in created if p.name == "actor_agent0.pt"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    engine = load_checkpoint(checkpoint, expected_family="dsr")
    runner = DSREpisodeRunner(engine.config, engine, project_root=str(ROOT))
    try:
        episode = runner.run_episode(seed=5, max_steps=12)
        assert episode.episode_length == 12
        assert np.isclose(
            episode.total_reward,
            sum(s["info"]["reward_total"] for s in episode.snapshots[1:]),
            rtol=1e-6,
        )
        assert all(s["info"]["invalid_actions"] == 0 for s in episode.snapshots[1:])
        report["happo_replay"] = {
            "steps": 12,
            "actors": engine.n_agents,
            "observation_dims": engine.obs_dims,
            "finite_model_files": len(created),
        }
    finally:
        runner.close()
    model = output / "ppo"
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
        output / "ppo.log",
    )
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env
    from utils.centralized_cli import load_model
    from envs.powergrid.adapters import CentralizedAdapter

    cfg = ConfigLoader().load(experiment=model.with_suffix(".environment.yaml"))
    env = CentralizedAdapter(
        make_power_env(cfg.main_args["env"], cfg.env_args, 0, 0).env,
        action_space_type="discrete",
    )
    try:
        policy = load_model(model, env)
        assert policy._n_updates == 4 and policy.num_timesteps == 96
        obs, _ = env.reset(seed=3)
        for _ in range(12):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            assert np.isfinite(reward) and info["dss_converged"]
        assert done and not truncated
        report["ppo"] = {"steps": 96, "updates": 4, "replay_steps": 12}
    finally:
        env.close()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    from scripts.check_smartgrid import check_spawn

    root = ROOT / "results/validation/dsr"
    root.mkdir(parents=True, exist_ok=True)
    report = {}
    for case, profile in [
        ("small", "configs/experiments/dsr_small_smoke.yaml"),
        ("ieee13", "configs/experiments/dsr_13bus_smoke.yaml"),
    ]:
        output = root / case
        output.mkdir(exist_ok=True)
        entry = {"native": check_native(profile), "spawn": check_spawn(profile)}
        report[case] = entry
        (root / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        if args.train:
            entry["training"] = check_training(profile, output, args.require_cuda)
        report[case] = entry
        (root / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("DSR_CASE_OK", case, flush=True)
    print("DSR_RUNTIME_OK", flush=True)


if __name__ == "__main__":
    main()
