"""五类电力环境从公共 CLI 启动真实短训练/评估；使用模型副本，失败汇总后返回非零。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = {
    "vvc": "vvc_13bus_smoke",
    "smartgrid": "smartgrid_synthetic_smoke",
    "dsr": "dsr_13bus_smoke",
    "stackelberg": "stackelberg_13bus_smoke",
    "district_dispatch": "district_dispatch_smoke",
}


def check_sequential_run(text, output, family, profile):
    import torch
    from envs.stackelberg.training import load_checkpoint, collect_episode

    marker = "SEQUENTIAL_PPO_OK "
    run = Path(
        next(line.split(marker, 1)[1] for line in text.splitlines() if marker in line)
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["physical_steps"] == 240 and summary["policy_updates"] == 10
    assert summary["validation_episodes"] == 4
    env, models = load_checkpoint(run / "checkpoint")
    try:
        _, records = collect_episode(env, models, training=False)
        assert len(records) == 24
        assert all(
            torch.isfinite(p).all() for m in models for p in m.policy.parameters()
        )
    finally:
        env.close()
        for model in models:
            model.get_env().close()
    return {
        "profile": profile,
        "steps": 240,
        "updates": 10,
        "evaluation_calls": 2,
        "finite_model_files": len(models),
        "replay_steps": 24,
        "log": str(output / f"{family}.log"),
    }


def check(family, output):
    from utils.config_pipeline import ConfigLoader

    profile = f"configs/experiments/{PROFILES[family]}.yaml"
    cfg = ConfigLoader().load(experiment=profile)
    source = ROOT / (
        cfg.env_args.get("dss_folder") or f"node_systems/{cfg.env_args['system_name']}"
    )
    began = time.time_ns()
    with tempfile.TemporaryDirectory(prefix=f"powerzoo-{family}-") as tmp:
        copy = Path(tmp) / "model"
        shutil.copytree(
            source, copy, ignore=shutil.ignore_patterns("__pycache__", "logs", "*.lock")
        )
        cmd = [
            sys.executable,
            "-u",
            "scripts/train.py",
            "--config",
            profile,
            "--exp-name",
            f"check_{family}",
            "--set",
            f"env.dss_folder={copy.as_posix()}",
            "--set",
            "logger.log_dir=results/entrypoints/runs",
            "--set",
            "eval.use_eval=true",
            "--set",
            "eval.n_eval_rollout_threads=1",
            "--set",
            "eval.eval_episodes=2",
            "--set",
            "train.eval_interval=5",
        ]
        with (output / f"{family}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=240,
            )
        if family == "stackelberg":
            sequential_result = check_sequential_run(
                (output / f"{family}.log").read_text(encoding="utf-8"),
                output,
                family,
                profile,
            )
    import torch

    text = (output / f"{family}.log").read_text(encoding="utf-8")
    if any(
        marker in text
        for marker in (
            "Traceback (most recent call last)",
            "| ERROR |",
            "ERROR:",
            "观测构建失败",
        )
    ):
        raise RuntimeError("训练日志记录了运行错误，请查看对应日志")

    if family == "stackelberg":
        return sequential_result

    checkpoints = [
        p
        for p in (ROOT / "results/entrypoints/runs").rglob("*.pt")
        if p.stat().st_mtime_ns >= began and f"check_{family}" in p.parts
    ]
    assert checkpoints, "训练没有保存模型"
    assert any(
        "checkpoint_episode_10" in p.parts for p in checkpoints
    ), "训练未完成第10次更新"
    for p in checkpoints:
        state = torch.load(p, map_location="cpu", weights_only=True)

        def visit(value):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"非有限模型参数：{p}"
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)

        visit(state)
    return {
        "profile": profile,
        "steps": 240,
        "updates": 10,
        "evaluation_calls": 2,
        "finite_model_files": len(checkpoints),
        "native_nonconvergence_warnings": text.count("潮流计算未收敛"),
        "log": str(output / f"{family}.log"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs", nargs="+", choices=list(PROFILES), default=list(PROFILES)
    )
    args = parser.parse_args()
    output = ROOT / "results/validation/entrypoints"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for family in args.envs:
        print(f"CHECK {family}", flush=True)
        try:
            results[family] = {"status": "passed", **check(family, output)}
        except Exception as exc:
            results[family] = {
                "status": "failed",
                "error": str(exc),
                "log": str(output / f"{family}.log"),
            }
        print(json.dumps({family: results[family]}, ensure_ascii=False), flush=True)
        (output / "report.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if any(item["status"] != "passed" for item in results.values()):
        raise SystemExit(1)
    print("TRAINING_ENTRYPOINTS_OK")


if __name__ == "__main__":
    main()
