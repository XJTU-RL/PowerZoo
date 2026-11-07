"""Validate native DSS, a real VVC rollout, Windows spawn, and optional training.

Run from the repository root with ``python scripts/check_runtime.py --train``.
Use ``--preflight`` for import, tensor and DSS checks only.
Rollout and training checks use a temporary copy of the checked-in IEEE13 model
and load profiles.
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

RUNTIME_IMPORTS = (
    "dss",
    "opendssdirect",
    "gym",
    "gymnasium",
    "stable_baselines3",
    "runners",
)


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


def check_preflight(require_cuda):
    """Report all baseline import errors and run the independent native probes."""
    failed = set()

    def report(label, check):
        try:
            details = check()
        except Exception as exc:
            failed.add(label)
            problem = "缺少模块 " if isinstance(exc, ModuleNotFoundError) else ""
            print(f"[失败] {label}  {problem}{type(exc).__name__}: {exc}")
        else:
            print(f"[通过] {label}" + (f"  {details}" if details else ""))

    def dss_details():
        info = check_native_dss()
        engine = "\n    ".join(info["dss_version"].strip().splitlines())
        return f"OpenDSSDirect.py {info['direct_version']}\n    {engine}"

    print(f"Python  {sys.executable}（{sys.version.split()[0]}）")
    print(f"工作目录  {Path.cwd()}\n")
    for module in RUNTIME_IMPORTS:
        report(
            module,
            lambda name=module: getattr(importlib.import_module(name), "__version__", ""),
        )

    def torch_details():
        info = check_torch(require_cuda)
        devices = ", ".join(device.upper() for device in info["devices"])
        cuda_status = "可用" if "cuda" in info["devices"] else "不可用"
        return f"{info['torch']}  实测设备 {devices}  CUDA {cuda_status}（构建 {info['cuda'] or '无'}）"

    report("PyTorch 张量运算", torch_details)
    if failed.intersection(("dss", "opendssdirect")):
        print("[略过] OpenDSS 原生求解  请先修复 DSS API 导入")
    else:
        report("OpenDSS 原生求解", dss_details)
    return not failed


def check_vvc(system_path, workers):
    """Step the real project vector environment, including spawned workers."""
    import numpy as np

    from utils.configs_tools import get_defaults_yaml_args
    from utils.envs_tools import make_train_env

    _, env_args = get_defaults_yaml_args("happo", "vvc")
    env_args.update(system_name=str(system_path), dss_folder=str(system_path))
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
    env_args.update(system_name=str(system_path), dss_folder=str(system_path))
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="also perform a HAPPO update")
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="check imports, tensor operations and DSS without starting VVC workers",
    )
    parser.add_argument(
        "--require-cuda", action="store_true", help="fail if CUDA is unavailable"
    )
    args = parser.parse_args()
    if args.preflight:
        sys.exit(0 if check_preflight(args.require_cuda) else 1)
    results = {"python": sys.version, "executable": sys.executable, "checks": {}}
    for module in RUNTIME_IMPORTS:
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
