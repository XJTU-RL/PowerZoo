"""在南洋的隔离仓库副本运行，使用合成 CBESS 剖面及真实 OpenDSS 求解。"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, error=None):
    result = subprocess.run(
        [sys.executable, "scripts/train_centralized.py", *map(str, args)],
        cwd=ROOT,
        check=False,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
    )
    if error:
        assert result.returncode != 0 and error in result.stderr, (
            result.stdout + result.stderr
        )
        return
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def profiles(root):
    sys.path.insert(0, str(ROOT))
    from envs.cbess.profile_store import ProfileStore

    case = ROOT / "node_systems" / "33Bus_CBESS"
    meta = json.loads((case / "case_meta.json").read_text(encoding="utf-8"))
    days = {"train": "2000-01-01", "validation": "2000-01-02"}
    for split, day in days.items():
        folder = root / split
        folder.mkdir(parents=True)
        loads = np.full((1, 4, meta["num_buses"]), 20.0)
        np.savez(
            folder / "profiles.npz",
            day_keys=[day],
            p_load_kw=loads,
            q_load_kvar=loads * 0.5,
            pv_available_kw=np.full((1, 4, len(meta["der_buses"])), 10.0),
            price_aud_per_mwh=np.full((1, 4), 85.0 if split == "train" else 95.0),
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "origin": "synthetic integration smoke",
                    "split": {day: split},
                    "pv_peak_kw": {str(bus): 30.0 for bus in meta["der_buses"]},
                }
            ),
            encoding="utf-8",
        )
    return days, ProfileStore(root / "validation", split="validation").fingerprint()


def cbess_smoke(output):
    days, eval_digest = profiles(output / "profiles")
    for backend in ("ldcppo", "sb3_sac", "sb3_td3"):
        config = output / f"{backend}.yaml"
        model = output / backend
        run = {
            "run": {"algo": backend, "env": "cbess"},
            "env": {"steps_per_day": 4, "data_dir": str(output / "profiles")},
        }
        if backend == "ldcppo":
            run.update(
                train={
                    "total_episodes": 2,
                    "episode_length": 4,
                    "budget_estimate_episodes": 1,
                    "checkpoint_every": 1,
                },
                algo={"ppo_epochs": 1},
            )
        else:
            run["algo"] = {"buffer_size": 16, "learning_starts": 0, "batch_size": 2}
        config.write_text(yaml.safe_dump(run), encoding="utf-8")
        base = ("--backend", backend, "--output", model)
        training = (*base, "--config", config)
        run_cli(*training, "--steps", 4, "--split", "train")
        snapshot = model.with_suffix(".environment.yaml")
        if backend == "ldcppo":
            run_cli(*base, "--config", snapshot, "--steps", 8, "--resume")
            statistics = json.loads(
                (model / "run.ldcppo.json").read_text(encoding="utf-8")
            )
            assert statistics["episodes"] == 2 and statistics["steps"] == 8
        else:
            from utils.single_agent.backends import sb3_class

            loaded = sb3_class(backend).load(model.with_suffix(".zip"), device="cpu")
            assert loaded.num_timesteps == 4 and loaded._n_updates > 0

        evaluation = (*base, "--config", snapshot, "--evaluate")
        for split in ("train", "validation"):
            options = (
                ("--eval-scenario-sha256", eval_digest) if split == "validation" else ()
            )
            result = json.loads(
                run_cli(*evaluation, "--split", split, *options).splitlines()[-1]
            )
            assert result["scenario"]["split"] == split
            episode = result["episodes"][0]
            assert episode["day_key"] == days[split] and episode["steps"] == 4
            assert episode["terminated"] and not episode["truncated"]
            assert set(episode["costs"]) == {"VD", "VN", "LO"}
        print(
            f"{backend}: CBESS optimizer update and bound train/validation evaluation passed"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_cli(
        "--backend",
        "ldcppo",
        "--config",
        "configs/experiments/dsr_small_smoke.yaml",
        "--output",
        output / "discrete_rejection",
        error="LDCPPO 需要 CBESS",
    )
    assert not (output / "discrete_rejection").exists()
    cbess_smoke(output)

    # 原 PPO 的量化 VVC 路线及两参数加载入口继续可用。
    model = output / "vvc_ppo"
    run_cli("--steps", 24, "--output", model)
    from envs.powergrid.adapters import CentralizedAdapter
    from utils.centralized_cli import load_model
    from utils.config_pipeline import ConfigLoader
    from utils.envs_tools import make_power_env

    cfg = ConfigLoader().load(experiment=model.with_suffix(".environment.yaml"))
    env = CentralizedAdapter(
        make_power_env("vvc", cfg.env_args, 0, cfg.env_args["seed"]).env
    )
    try:
        policy = load_model(model, env)
        assert policy.num_timesteps == 24 and policy._n_updates == 1
        obs, _ = env.reset(seed=0)
        action, _ = policy.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        assert np.isfinite(reward) and info["dss_converged"]
    finally:
        env.close()
    print("sb3_ppo: VVC quantized path and existing model loader passed")


if __name__ == "__main__":
    main()
