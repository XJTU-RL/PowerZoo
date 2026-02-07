"""Scan the results/ directory tree for trained models.

Directory convention::

	results/{env}/{system}/{algo}/{exp_name}/{seed-XXXXX-timestamp}/
		config.json
		models/
			actor_agent0.pt
			actor_agent1.pt
			critic.pt
			...
		logs/
			events.out.tfevents.*

This module walks the tree, collects ``ModelRecord`` objects, and provides
filtering and sorting utilities for the Gradio UI model browser.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default directory where PowerZoo writes training outputs
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class ModelRecord:
	"""Metadata for a single training run."""

	env: str
	system: str
	algo: str
	exp_name: str
	run_id: str        # e.g. "seed-00001-2025-06-01"
	run_dir: str       # absolute path to the run directory
	config: dict = field(default_factory=dict)
	checkpoints: list[str] = field(default_factory=list)


def scan_models(results_dir: str | None = None) -> list[ModelRecord]:
	"""Scan the results directory and return discovered model records.

	The expected hierarchy is::

		results/{env}/{system}/{algo}/{exp_name}/{run_id}/

	Where *run_id* is typically ``seed-XXXXX-timestamp``.

	Args:
		results_dir: Override for the results root.  Defaults to
			``PROJECT_ROOT / "results"``.

	Returns:
		List of ``ModelRecord``, sorted by run directory modification time
		(most recent first).
	"""
	root = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR

	if not root.exists() or not root.is_dir():
		return []

	records: list[ModelRecord] = []

	# Walk: env / system / algo / exp_name / run_id
	for env_dir in _iter_dirs(root):
		env_name = env_dir.name
		for system_dir in _iter_dirs(env_dir):
			system_name = system_dir.name
			for algo_dir in _iter_dirs(system_dir):
				algo_name = algo_dir.name
				for exp_dir in _iter_dirs(algo_dir):
					exp_name = exp_dir.name
					for run_dir in _iter_dirs(exp_dir):
						run_id = run_dir.name
						record = ModelRecord(
							env=env_name,
							system=system_name,
							algo=algo_name,
							exp_name=exp_name,
							run_id=run_id,
							run_dir=str(run_dir),
							config=get_model_config(str(run_dir)),
							checkpoints=get_checkpoints(str(run_dir)),
						)
						records.append(record)

	# Sort by directory modification time, newest first
	records.sort(
		key=lambda r: Path(r.run_dir).stat().st_mtime
		if Path(r.run_dir).exists() else 0,
		reverse=True,
	)
	return records


def get_model_config(run_dir: str) -> dict:
	"""Load ``config.json`` from a run directory.

	Args:
		run_dir: Absolute path to a training run directory.

	Returns:
		Parsed config dict, or empty dict if the file is missing or invalid.
	"""
	config_path = Path(run_dir) / "config.json"
	if not config_path.exists():
		return {}
	try:
		with open(config_path, "r", encoding="utf-8") as f:
			return json.load(f)
	except (json.JSONDecodeError, OSError):
		return {}


def get_checkpoints(run_dir: str) -> list[str]:
	"""List checkpoint files (``*.pt``) inside ``run_dir/models/``.

	Args:
		run_dir: Absolute path to a training run directory.

	Returns:
		Sorted list of checkpoint filenames (not full paths).
	"""
	models_dir = Path(run_dir) / "models"
	if not models_dir.exists() or not models_dir.is_dir():
		return []
	return sorted(p.name for p in models_dir.iterdir() if p.suffix == ".pt")


def filter_models(
	records: list[ModelRecord],
	env: str | None = None,
	algo: str | None = None,
) -> list[ModelRecord]:
	"""Filter model records by environment and/or algorithm name.

	Args:
		records: Full list of ModelRecord to filter.
		env: If provided, keep only records matching this env name.
		algo: If provided, keep only records matching this algo name.

	Returns:
		Filtered list (preserves original ordering).
	"""
	filtered = records
	if env:
		filtered = [r for r in filtered if r.env == env]
	if algo:
		filtered = [r for r in filtered if r.algo == algo]
	return filtered


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_dirs(parent: Path):
	"""Yield immediate child directories of *parent*, skipping hidden dirs."""
	if not parent.is_dir():
		return
	for child in sorted(parent.iterdir()):
		if child.is_dir() and not child.name.startswith("."):
			yield child
