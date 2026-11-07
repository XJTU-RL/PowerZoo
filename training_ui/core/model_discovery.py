"""Discover models written by the common MARL and sequential PPO runners."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class ModelRecord:
    """Metadata for a single training run."""

    env: str
    system: str
    algo: str
    exp_name: str
    run_id: str
    run_dir: str
    config: dict = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)


def scan_models(results_dir: str | None = None) -> list[ModelRecord]:
    """Read known runner directory layouts, newest run first."""
    root = Path(results_dir).resolve() if results_dir else DEFAULT_RESULTS_DIR
    records = []

    def add_run(run_dir, env, system, algo, exp_name, config):
        checkpoints = get_checkpoints(str(run_dir))
        if config or checkpoints:
            records.append(ModelRecord(
                env, system, algo, exp_name, run_dir.name, str(run_dir), config, checkpoints
            ))

    # Synchronous runners: env / system / algo / experiment / seed-*.
    for env_dir in _iter_dirs(root):
        for system_dir in _iter_dirs(env_dir):
            for algo_dir in _iter_dirs(system_dir):
                if env_dir.name == "stackelberg" and algo_dir.name.startswith("run_"):
                    continue  # Sequential PPO layout handled below.
                for exp_dir in _iter_dirs(algo_dir):
                    for run_dir in _iter_dirs(exp_dir):
                        add_run(
                            run_dir, env_dir.name, system_dir.name, algo_dir.name,
                            exp_dir.name, get_model_config(str(run_dir)),
                        )

    # Sequential PPO: stackelberg / experiment / run_* / checkpoint.
    for exp_dir in _iter_dirs(root / "stackelberg"):
        for run_dir in _iter_dirs(exp_dir):
            if not run_dir.name.startswith("run_"):
                continue
            config = get_model_config(str(run_dir))
            env_config = config.get("env", {})
            system = env_config.get("system_name", "") if isinstance(env_config, dict) else ""
            add_run(run_dir, "stackelberg", system, "sequential_ppo", exp_dir.name, config)

    return sorted(records, key=lambda record: _modified_at(Path(record.run_dir)), reverse=True)


def get_model_config(run_dir: str) -> dict:
    """Read the resolved YAML snapshot or the older runner JSON export."""
    root = Path(run_dir)
    for path in (
        root / "config.resolved.yaml",
        root / "checkpoint" / "config.resolved.yaml",
        root / "config.json",
    ):
        try:
            raw = path.read_text(encoding="utf-8")
            config = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError):
            continue
        if isinstance(config, dict):
            return config
    return {}


def get_checkpoints(run_dir: str) -> list[str]:
    """List recognized model files relative to the run directory."""
    root = Path(run_dir)
    models = root / "models"
    stages = [
        path for path in _iter_dirs(models)
        if path.name.startswith(("checkpoint_episode_", "policy_")) or path.name == "best_model"
    ]
    files = [
        path
        for directory in [models, *stages]
        for path in directory.glob("*.pt")
        if path.is_file()
    ]
    files.extend(
        path for path in (root / "checkpoint").glob("agent_*.zip") if path.is_file()
    )
    return sorted(path.relative_to(root).as_posix() for path in files)


def filter_models(
    records: list[ModelRecord],
    env: str | None = None,
    algo: str | None = None,
) -> list[ModelRecord]:
    """Filter model records while preserving the discovery order."""
    return [
        record
        for record in records
        if (not env or record.env == env) and (not algo or record.algo == algo)
    ]


def _iter_dirs(parent: Path):
    if parent.is_dir():
        yield from (
            child for child in parent.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )


def _modified_at(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0
