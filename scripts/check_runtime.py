"""Validate native DSS, a real VVC rollout, Windows spawn, and optional training.

Run from the repository root with ``python scripts/check_runtime.py --train``.
Checks use a temporary copy of the checked-in IEEE13 model and load profiles.
They never replace research inputs with generated synthetic profiles.
"""

import argparse
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_native_dss():
    """Load the bundled native DLL and solve through both installed DSS APIs."""
    import dss
    import opendssdirect

    command = (
        "clear\nnew circuit.runtime_check basekv=12.47 pu=1 phases=3 bus1=source\n"
        "new load.test bus1=source phases=3 conn=wye kv=12.47 kw=100 kvar=20\n"
        "set voltagebases=[12.47]\ncalcvoltagebases\nsolve"
    )
    engine = dss.DSS.NewContext()
    engine.Text.Commands(command)
    if not engine.ActiveCircuit.Solution.Converged:
        raise RuntimeError("DSS-Python native solve did not converge")
    direct = opendssdirect.NewContext()
    direct.Text.Commands(command)
    if not direct.Solution.Converged():
        raise RuntimeError("OpenDSSDirect native solve did not converge")
    return {"dss_version": engine.Version, "direct_version": opendssdirect.__version__}


def check_torch(require_cuda):
    """Run an actual tensor operation on CPU and on CUDA when available."""
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif require_cuda:
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    for device in devices:
        value = torch.ones((16, 16), device=device) @ torch.ones(
            (16, 16), device=device
        )
        if value[0, 0].item() != 16:
            raise RuntimeError(f"Tensor operation failed on {device}")
    return {"torch": torch.__version__, "devices": devices, "cuda": torch.version.cuda}


def check_vvc(system_path, workers):
    """Step the real project vector environment, including spawned workers."""
    import numpy as np

    from utils.configs_tools import get_defaults_yaml_args
    from utils.envs_tools import make_train_env

    _, env_args = get_defaults_yaml_args("happo", "vvc")
    env_args.update(system_name=str(system_path), dss_folder=str(system_path), useS=False, use_render=False)
    env = make_train_env("vvc", 42, workers, env_args)
    try:
        obs, _, _ = env.reset()
        for _ in range(4):
            actions = np.asarray(
                [[space.sample() for space in env.action_space] for _ in range(workers)]
            )[..., None]
            obs, _, rewards, _, infos, _ = env.step(actions)
            if not np.isfinite(np.asarray(obs, dtype=float)).all():
                raise RuntimeError("Non-finite VVC observations")
            if not np.isfinite(rewards).all():
                raise RuntimeError("Non-finite VVC rewards")
            if any(info.get("error") for worker in infos for info in worker):
                raise RuntimeError(f"VVC returned an error transition: {infos}")
        return {"workers": workers, "observation_shape": list(np.shape(obs))}
    finally:
        env.close()


def check_training(system_path, output_path, cuda=False):
    """Complete one small HAPPO update using the project's normal runner."""
    from runners import RUNNER_REGISTRY
    from utils.configs_tools import get_defaults_yaml_args

    algo_args, env_args = get_defaults_yaml_args("happo", "vvc")
    env_args.update(system_name=str(system_path), dss_folder=str(system_path), useS=False, use_render=False)
    algo_args["device"].update(cuda=cuda, torch_threads=1)
    algo_args["train"].update(
        n_rollout_threads=1, num_env_steps=24, episode_length=24, eval_interval=1
    )
    algo_args["eval"]["use_eval"] = False
    algo_args["model"].update(use_recurrent_policy=False, hidden_sizes=[32, 32])
    algo_args["algo"].update(
        ppo_epoch=1, critic_epoch=1, actor_num_mini_batch=1, critic_num_mini_batch=1
    )
    algo_args["logger"]["log_dir"] = str(output_path)
    args = {"algo": "happo", "env": "vvc", "exp_name": "windows_runtime_check"}
    runner = RUNNER_REGISTRY["happo"](args, algo_args, env_args)
    try:
        runner.run()
    finally:
        runner.close()
    return {"algorithm": "happo", "steps": 24, "device": str(runner.device)}


def main():
    """Report checks as JSON and return a nonzero exit code on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", action="store_true", help="also perform a HAPPO update"
    )
    parser.add_argument(
        "--require-cuda", action="store_true", help="fail if CUDA is unavailable"
    )
    args = parser.parse_args()
    results = {"python": sys.version, "executable": sys.executable, "checks": {}}
    for module in (
        "dss",
        "opendssdirect",
        "gym",
        "gymnasium",
        "stable_baselines3",
        "runners",
    ):
        importlib.import_module(module)
    results["checks"]["torch"] = check_torch(args.require_cuda)
    results["checks"]["native_dss"] = check_native_dss()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="powerzoo smoke ") as directory:
        temporary = Path(directory)
        system = temporary / "13Bus"
        shutil.copytree(ROOT / "node_systems" / "13Bus", system)
        try:
            results["checks"]["vvc_single_worker"] = check_vvc(system, 1)
            results["checks"]["vvc_spawn_workers"] = check_vvc(system, 2)
            if args.train:
                results["checks"]["training"] = check_training(
                    system, temporary / "results"
                )
                if args.require_cuda:
                    results["checks"]["training_cuda"] = check_training(
                        system, temporary / "results", cuda=True
                    )
        finally:
            os.chdir(original_cwd)
    print("POWERZOO_RUNTIME_OK " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
