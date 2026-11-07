"""区域原生计量、全天规则对照、Windows worker及真实训练/回放验收。"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE = "configs/experiments/district_dispatch_smoke.yaml"


def baseline_action(core, rule=False):
    import numpy as np

    price, _ = core.config.market.price_at(core.circuit.time_s)
    mean_price = core.config.market.base_price_yuan_kwh * float(
        np.mean(core.config.market.tou_multipliers)
    )
    parts = []
    for i, region in enumerate(core.config.regions):
        voltage = float(
            np.mean(
                [
                    v
                    for b in region.buses
                    for v in core.physical_state.voltages_pu[b].values()
                ]
            )
        )
        q = float(np.clip((1 - voltage) * 5, -1, 1)) if rule else 0.0
        for element in region.controls:
            if element.startswith("PVSystem."):
                parts.extend([1.0, q])
            elif element.startswith("Storage."):
                # 公开TOU+期末能量恢复规则；仅为可核算对照，不宣称最优。
                value = (
                    (
                        1.0
                        if price > mean_price * 1.15
                        else -1.0 if price < mean_price * 0.85 else 0.0
                    )
                    if rule
                    else 0.0
                )
                if rule and core.t >= core.horizon - 16:
                    name = element.split(".", 1)[1]
                    dev = core.circuit.batteries[name]
                    delta = core._energy()[name] - core.initial_energy[name]
                    remaining = (
                        (core.horizon - core.t) * core.config.data.step_seconds / 3600
                    )
                    value = float(
                        np.clip(
                            (
                                delta * dev.eta_discharge / (remaining * dev.max_kw)
                                if delta > 0
                                else delta / (dev.eta_charge * remaining * dev.max_kw)
                            ),
                            -1,
                            1,
                        )
                    )
                parts.append(value)
            else:
                parts.append(q)
    return parts


def check_native():
    import numpy as np
    from utils.config_pipeline import ConfigLoader
    from utils.config_validation import build_dataclass, environment_class
    from envs.district_dispatch.core.dispatch_core import DistrictDispatchCore

    cfg = ConfigLoader().load(
        "happo",
        "district_dispatch",
        values={"env": {"data": {"episodes_per_split": 6}}},
    )
    core = DistrictDispatchCore(
        build_dataclass(environment_class("district_dispatch"), cfg.env_args)
    )
    records = []
    try:
        for controller in ("fixed", "tou_voltage_rule"):
            for episode in range(6):
                core.reset(load_profile_idx=episode)
                samples = []
                for _ in range(core.horizon):
                    _, reward, done, info = core.step(
                        baseline_action(core, controller != "fixed")
                    )
                    assert np.isfinite(reward) and info["dss_converged"]
                    assert abs(info["active_balance_error_kw"]) < 0.05
                    assert (
                        max(
                            abs(v)
                            for r in info["regions"].values()
                            for v in r["balance_error"]
                        )
                        < 0.05
                    )
                    samples.append(info)
                assert done and core.circuit.time_s == 86400
                dt = core.config.data.step_seconds / 3600
                costs = {key: float(sum(s[key] for s in samples)) for key in core.costs}
                records.append(
                    {
                        "controller": controller,
                        "scenario": info["scenario"],
                        "steps": core.horizon,
                        "clock_end_s": core.circuit.time_s,
                        "costs_yuan": costs,
                        "total_cost_yuan": sum(costs.values()),
                        "grid_import_kwh": sum(
                            max(s["total_power_kw"], 0) * dt for s in samples
                        ),
                        "grid_export_kwh": sum(
                            max(-s["total_power_kw"], 0) * dt for s in samples
                        ),
                        "loss_kwh": sum(s["power_loss_kw"] * dt for s in samples),
                        "pv_kwh": sum(s["pv_power_kw"] * dt for s in samples),
                        "peak_import_kw": max(s["total_power_kw"] for s in samples),
                        "phase_violation_minutes": sum(
                            s["voltage_violation_count"] * dt * 60 for s in samples
                        ),
                        "voltage_min_pu": min(s["voltage_min_pu"] for s in samples),
                        "voltage_max_pu": max(s["voltage_max_pu"] for s in samples),
                        "max_loading_pu": max(
                            s["max_branch_loading_pu"] for s in samples
                        ),
                        "max_balance_error_kw": max(
                            abs(s["active_balance_error_kw"]) for s in samples
                        ),
                        "max_region_balance_kw_kvar": max(
                            abs(v)
                            for s in samples
                            for r in s["regions"].values()
                            for v in r["balance_error"]
                        ),
                        "final_energy_kwh": core._energy(),
                        "final_soc": info["storage_soc"],
                    }
                )
                print(
                    "DISTRICT_NATIVE_CASE_OK", controller, info["scenario"], flush=True
                )
        return {
            "model_sha256": core.circuit.model_sha256,
            "dataset_sha256": core.dataset.sha256,
            "step_seconds": core.config.data.step_seconds,
            "action_dimensions": [s.shape[0] for s in core.agent_spaces],
            "actor_observation_dim": core.local_observation_space.shape[0],
            "critic_observation_dim": core.observation_space.shape[0],
            "cases": records,
        }
    finally:
        core.close()


def check_training(output, require_cuda):
    import numpy as np
    import torch
    from scripts.check_vvc import run_logged

    report = {}
    began = time.time_ns()
    for device in ["cpu", "cuda"] if require_cuda else ["cpu"]:
        run_logged(
            [
                "scripts/train.py",
                "--config",
                PROFILE,
                "--exp-name",
                "district_" + device,
                "--set",
                "device.cuda=" + str(device == "cuda").lower(),
                "--set",
                "logger.log_dir=" + (output / "runs").as_posix(),
                "--set",
                "eval.use_eval=true",
                "--set",
                "eval.n_eval_rollout_threads=1",
                "--set",
                "eval.eval_episodes=6",
                "--set",
                "train.eval_interval=5",
            ],
            output / f"happo_{device}.log",
        )
        report["happo_" + device] = {
            "steps": 240,
            "updates": 10,
            "validation_episodes": 12,
        }
    created = [
        p for p in (output / "runs").rglob("*.pt") if p.stat().st_mtime_ns >= began
    ]
    assert created and any("checkpoint_episode_10" in p.parts for p in created)
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
    from envs.powergrid.checkpoint import load_checkpoint
    from envs.district_dispatch.render.engine.episode_runner import EpisodeRunner

    engine = load_checkpoint(checkpoint, expected_family="district_dispatch")
    runner = EpisodeRunner(engine.config, engine, project_root=str(ROOT))
    try:
        episode = runner.run_episode(seed=4, max_steps=24)
        assert episode.episode_length == 24
        assert np.isclose(
            episode.total_reward,
            sum(s["info"]["reward_total"] for s in episode.snapshots[1:]),
            rtol=1e-6,
        )
        report["happo_replay"] = {
            "actors": engine.n_agents,
            "obs_dims": engine.obs_dims,
            "steps": 24,
            "finite_model_files": len(created),
        }
    finally:
        runner.close()
    model = output / "ppo"
    run_logged(
        [
            "scripts/train_centralized.py",
            "--config",
            PROFILE,
            "--steps",
            "96",
            "--output",
            str(model),
        ],
        output / "sb3.log",
    )
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env
    from envs.powergrid.adapters import CentralizedAdapter
    from utils.centralized_cli import load_model

    cfg = ConfigLoader().load(experiment=model.with_suffix(".environment.yaml"))
    wrapper = make_power_env("district_dispatch", cfg.env_args, 0, cfg.env_args["seed"])
    central = CentralizedAdapter(wrapper.env)
    try:
        policy = load_model(model, central)
        assert policy.num_timesteps == 96 and policy._n_updates == 4
        obs, _ = central.reset(seed=7)
        for _ in range(24):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = central.step(action)
            assert np.isfinite(reward) and info["dss_converged"]
        assert done and not truncated
        report["ppo"] = {"steps": 96, "updates": 4, "replay_steps": 24}
    finally:
        central.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    from scripts.check_smartgrid import check_spawn

    output = ROOT / "results/validation/district"
    output.mkdir(parents=True, exist_ok=True)
    report = {"native": check_native(), "spawn": check_spawn(PROFILE)}
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if args.train:
        report["training"] = check_training(output, args.require_cuda)
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    print("DISTRICT_RUNTIME_OK", flush=True)


if __name__ == "__main__":
    main()
