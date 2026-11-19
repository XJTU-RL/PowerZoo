"""Replay a frozen HAPPO checkpoint on explicit VVC validation scenarios."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

from scripts.vvc_benchmark_metrics import summarize_episode

FIELDS = (
    "dss_converged",
    "step_seconds",
    "power_loss_kw",
    "voltage_violation_count",
    "voltage_min_pu",
    "voltage_max_pu",
)


def _runtime():
    from envs.powergrid.checkpoint import load_checkpoint
    from envs.vvc.render.engine.episode_runner import VVCEpisodeRunner

    return load_checkpoint, VVCEpisodeRunner


def _normalize(snapshots, identity):
    if not snapshots or snapshots[0].get("step") != 0:
        raise ValueError("native replay must start with an initial snapshot")
    records = []
    for step, snapshot in enumerate(snapshots[1:], 1):
        if snapshot.get("step") != step:
            raise ValueError("noncontiguous snapshot steps")
        if snapshot.get("physical_failure") is True:
            if step != len(snapshots) - 1:
                raise ValueError("native replay continued after physical failure")
            records.append({"physical_failure": True})
            continue
        info = snapshot.get("info")
        if not isinstance(info, dict):
            raise ValueError("successful native step missing info")
        for key in ("scenario_id", "dataset_sha256"):
            if info.get(key) != identity[key]:
                raise ValueError(f"native step {key} disagrees with selected scenario")
        records.append({key: info[key] for key in FIELDS if key in info})
    return records


def export_validation(
    checkpoint, manifest, indices, *, seed, device="cpu", split="validation"
):
    if split != "validation":
        raise ValueError("only validation is enabled")
    indices = tuple(indices)
    if (
        not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("provide distinct nonnegative scenario indices")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    manifest = Path(manifest).resolve(strict=True)
    digest = sha256(manifest.read_bytes()).hexdigest()
    load_checkpoint, runner_class = _runtime()
    engine = load_checkpoint(checkpoint, device=device, expected_family="vvc")
    config = engine.config
    if config.smartds_series:
        raise ValueError("this exporter only supports ScenarioDataset-based VVC")
    data = config.data
    if data.split != "train" or not data.path or data.manifest_sha256 != digest:
        raise ValueError(
            "checkpoint must bind a frozen train manifest with matching SHA256"
        )
    config = replace(
        config,
        data=replace(
            data, split="validation", path=str(manifest), manifest_sha256=digest
        ),
    )
    episodes = []
    seen = set()
    for index in indices:
        runner = runner_class(config, engine, scenario_index=index)
        try:
            replay = runner.run_episode(seed=seed, max_steps=config.max_episode_steps)
            identity = runner.benchmark_identity
            if (
                identity["split"] != split
                or identity["dataset_sha256"] != digest
                or identity["scenario_index"] != index
                or not identity["scenario_id"]
                or identity["scenario_id"] in seen
            ):
                raise ValueError(
                    "native scenario identity, split, or manifest mismatch"
                )
            seen.add(identity["scenario_id"])
            records = _normalize(replay.snapshots, identity)
            if replay.episode_length != len(records):
                raise ValueError("native snapshot count disagrees with replay length")
            episodes.append(
                {
                    **identity,
                    "seed": seed,
                    "records": records,
                    "metrics": summarize_episode(records, config.max_episode_steps),
                }
            )
        finally:
            runner.close()
    return {"split": split, "dataset_sha256": digest, "episodes": episodes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-index", type=int, required=True, action="append")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["validation"], default="validation")
    args = parser.parse_args(argv)
    try:
        output = export_validation(
            args.checkpoint,
            args.manifest,
            args.scenario_index,
            seed=args.seed,
            device=args.device,
            split=args.split,
        )
        print(json.dumps(output, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"VVC validation export failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
