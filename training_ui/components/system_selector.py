"""IEEE system selector component.

Provides a dropdown for selecting an IEEE test feeder system along with a
markdown info card that displays metadata (node count, PV info, etc.) loaded
from ``configs/systems/_registry.yaml``.
"""

import gradio as gr
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

_REGISTRY_PATH = PROJECT_ROOT / "configs" / "systems" / "_registry.yaml"

# Module-level cache so the YAML is only read once.
_system_metadata_cache: dict | None = None


def load_system_metadata() -> dict:
	"""Load system metadata from the registry YAML.

	Reads ``configs/systems/_registry.yaml`` and returns the ``metadata``
	section as a dict keyed by system name.

	Returns:
		Dict mapping system name -> metadata dict.  Returns an empty dict
		if the file is missing or unparseable.
	"""
	global _system_metadata_cache
	if _system_metadata_cache is not None:
		return _system_metadata_cache

	if not _REGISTRY_PATH.exists():
		_system_metadata_cache = {}
		return _system_metadata_cache

	try:
		with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f) or {}
		_system_metadata_cache = data.get("metadata", {})
	except Exception:
		_system_metadata_cache = {}

	return _system_metadata_cache


def _format_system_info(system_name: str) -> str:
	"""Format system metadata as a markdown info card.

	Displays node_count, description, and optional PV / district info.

	Args:
		system_name: Name of the IEEE system, e.g. "34Bus_PV".

	Returns:
		Markdown string.
	"""
	metadata = load_system_metadata()
	meta = metadata.get(system_name)

	if not meta:
		return f"*No metadata available for `{system_name}`.*"

	lines = [
		f"**{system_name}**",
		"",
		f"> {meta.get('description', 'N/A')}",
		"",
		f"| Property | Value |",
		f"|----------|-------|",
		f"| Nodes | {meta.get('node_count', 'N/A')} |",
	]

	if "pv_count" in meta:
		lines.append(f"| PV Units | {meta['pv_count']} |")
	if "penetration_rate" in meta:
		rate_pct = f"{meta['penetration_rate'] * 100:.1f}%"
		lines.append(f"| PV Penetration | {rate_pct} |")
	if "district_count" in meta:
		lines.append(f"| Districts | {meta['district_count']} |")
	if "typical_episode_length" in meta:
		lines.append(f"| Episode Length | {meta['typical_episode_length']} |")

	return "\n".join(lines)


def build_system_selector(
	available_systems: list[str],
	default: str,
) -> tuple[gr.Dropdown, gr.Markdown]:
	"""Build system selector with info card.

	Args:
		available_systems: List of system names to show in the dropdown.
		default: Default selected system name.

	Returns:
		Tuple of (system_dropdown, system_info_card).
	"""
	system_dropdown = gr.Dropdown(
		label="IEEE System",
		choices=available_systems,
		value=default,
	)
	system_info = gr.Markdown(value=_format_system_info(default))
	return system_dropdown, system_info


def on_system_change(system_name: str) -> str:
	"""Callback when system selection changes.

	Args:
		system_name: Newly selected system name.

	Returns:
		Updated markdown string for the info card.
	"""
	if not system_name:
		return "*Select a system to see its details.*"
	return _format_system_info(system_name)


def bind_events(
	system_dropdown: gr.Dropdown,
	system_info: gr.Markdown,
) -> None:
	"""Bind change events for the system selector.

	Args:
		system_dropdown: The IEEE system dropdown.
		system_info: The system info markdown card.
	"""
	system_dropdown.change(
		fn=on_system_change,
		inputs=[system_dropdown],
		outputs=[system_info],
	)
