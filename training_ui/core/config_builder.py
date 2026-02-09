"""Build training configuration YAML from UI parameters.

Reads base algorithm and environment YAML configs from the PowerZoo project,
deep-merges user overrides from the Gradio UI, and writes a complete config
file to the cache directory for consumption by train.py.
"""

import uuid
from copy import deepcopy
from pathlib import Path

import yaml

# Project root is two levels up from this file:
#   training_ui/core/config_builder.py -> PowerZoo-gradio-ui/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Base config directories inside the PowerZoo project tree
ALGO_CFG_DIR = PROJECT_ROOT / "configs" / "algos_cfgs"
ENV_CFG_DIR = PROJECT_ROOT / "configs" / "envs_cfgs"

# Cache directory for generated task configs
CACHE_DIR = PROJECT_ROOT / "training_ui" / ".cache"


def _ensure_cache_dir() -> Path:
	"""Create the cache directory if it does not exist.

	Returns:
		Resolved cache directory path.
	"""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	return CACHE_DIR


def load_base_config(algo: str, env: str) -> tuple[dict, dict]:
	"""Load base algorithm and environment YAML configs.

	Args:
		algo: Algorithm name, e.g. "happo".  Resolved to
			``configs/algos_cfgs/{algo}.yaml``.
		env: Environment name, e.g. "vvc".  Resolved to
			``configs/envs_cfgs/{env}.yaml``.

	Returns:
		(algo_args, env_args) tuple of parsed dicts.

	Raises:
		FileNotFoundError: If either YAML file is missing.
	"""
	algo_path = ALGO_CFG_DIR / f"{algo}.yaml"
	env_path = ENV_CFG_DIR / f"{env}.yaml"

	if not algo_path.exists():
		raise FileNotFoundError(f"Algorithm config not found: {algo_path}")
	if not env_path.exists():
		raise FileNotFoundError(f"Environment config not found: {env_path}")

	with open(algo_path, "r", encoding="utf-8") as f:
		algo_args = yaml.safe_load(f) or {}

	with open(env_path, "r", encoding="utf-8") as f:
		env_args = yaml.safe_load(f) or {}

	return algo_args, env_args


def deep_merge(base: dict, override: dict) -> dict:
	"""Recursively merge *override* into a deep copy of *base*.

	- Dict values are merged recursively.
	- Non-dict values in *override* replace those in *base*.
	- Keys present only in *override* are added.

	Args:
		base: Base configuration dict (not mutated).
		override: Override dict whose values take precedence.

	Returns:
		New merged dict.
	"""
	merged = deepcopy(base)
	for key, value in override.items():
		if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
			merged[key] = deep_merge(merged[key], value)
		else:
			merged[key] = deepcopy(value)
	return merged


def build_config(
	algo: str,
	env: str,
	exp_name: str,
	ui_overrides: dict,
) -> tuple[str, dict]:
	"""Build a complete training config and persist it to disk.

	Workflow:
		1. Load base algo and env YAML configs.
		2. Deep-merge ``ui_overrides`` section-by-section into algo_args.
		   Recognised top-level keys in *ui_overrides*: ``seed``, ``device``,
		   ``train``, ``eval``, ``render``, ``model``, ``algo``, ``logger``,
		   ``env``.  The ``env`` key merges into env_args; all others merge
		   into algo_args.
		3. Write the combined config to
		   ``training_ui/.cache/task_{short_uuid}.yaml``.

	Args:
		algo: Algorithm name.
		env: Environment name.
		exp_name: Human-readable experiment name.
		ui_overrides: Nested dict of ``{section: {key: value}}`` overrides.
			Example::

				{
					"train": {"num_env_steps": 100000},
					"algo": {"lr": 0.001},
					"env": {"episode_length": 24},
				}

	Returns:
		(config_path, full_config) where *config_path* is the absolute path
		to the saved YAML file, and *full_config* is the complete dict.
	"""
	algo_args, env_args = load_base_config(algo, env)

	# Separate env-level overrides from algo-level overrides
	overrides = deepcopy(ui_overrides) if ui_overrides else {}
	env_overrides = overrides.pop("env", {})

	# Merge algo-level overrides (train, model, algo, seed, device, ...)
	if overrides:
		algo_args = deep_merge(algo_args, overrides)

	# Merge env-level overrides
	if env_overrides:
		env_args = deep_merge(env_args, env_overrides)

	# Assemble full config
	full_config = {
		"algo_name": algo,
		"env_name": env,
		"exp_name": exp_name,
		"algo_args": algo_args,
		"env_args": env_args,
	}

	# Persist to cache
	_ensure_cache_dir()
	short_id = uuid.uuid4().hex[:8]
	config_path = CACHE_DIR / f"task_{short_id}.yaml"

	with open(config_path, "w", encoding="utf-8") as f:
		yaml.dump(
			full_config,
			f,
			default_flow_style=False,
			allow_unicode=True,
			sort_keys=False,
		)

	return str(config_path), full_config


def config_to_yaml_str(config: dict) -> str:
	"""Render a config dict as a human-readable YAML string.

	Useful for the UI preview panel.

	Args:
		config: Configuration dict to render.

	Returns:
		Formatted YAML string.
	"""
	return yaml.dump(
		config,
		default_flow_style=False,
		allow_unicode=True,
		sort_keys=False,
	)
