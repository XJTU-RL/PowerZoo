"""Algorithm family and algorithm selection component.

Provides a two-level selector: first pick an algorithm family (On-Policy HA,
Off-Policy MA, etc.), then pick a specific algorithm within that family.
An info card displays the selected algorithm's description.
"""

import gradio as gr


def _lazy_registry():
	"""Lazy-import registry to avoid circular/path issues at module level."""
	from training_ui.core.registry import (
		ALGO_FAMILIES,
		ALGO_REGISTRY,
		get_algo_choices,
		get_family_choices,
	)
	return ALGO_FAMILIES, ALGO_REGISTRY, get_algo_choices, get_family_choices


def _format_algo_description(algo_display_name: str) -> str:
	"""Format algorithm description as a markdown info card.

	Args:
		algo_display_name: Display name of the algorithm, e.g. "HAPPO".

	Returns:
		Markdown string with algorithm metadata.
	"""
	_, ALGO_REGISTRY, _, _ = _lazy_registry()
	# Reverse lookup: display_name -> AlgoMeta
	for meta in ALGO_REGISTRY.values():
		if meta.display_name == algo_display_name:
			lines = [
				f"**{meta.display_name}** (`{meta.name}`)",
				f"",
				f"> {meta.description}",
				f"",
				f"- **Family**: `{meta.family}`",
				f"- **Config**: `{meta.config_file}`",
			]
			if meta.compatible_envs:
				envs_str = ", ".join(meta.compatible_envs)
				lines.append(f"- **Compatible Envs**: {envs_str}")
			else:
				lines.append("- **Compatible Envs**: All")
			return "\n".join(lines)
	return "*Select an algorithm to see its description.*"


def build_algo_selector() -> tuple[gr.Radio, gr.Dropdown, gr.Markdown]:
	"""Build algorithm selection component.

	Creates a two-level selector with family radio buttons, algorithm dropdown,
	and a markdown description card.

	Returns:
		Tuple of (family_radio, algo_dropdown, algo_description).
	"""
	_, _, get_algo_choices, get_family_choices = _lazy_registry()

	families = get_family_choices()
	default_family = families[0]  # "On-Policy HA"
	default_algos = get_algo_choices(default_family)
	default_algo = default_algos[0] if default_algos else None
	default_desc = _format_algo_description(default_algo) if default_algo else ""

	family_radio = gr.Radio(
		label="Algorithm Family",
		choices=families,
		value=default_family,
	)
	algo_dropdown = gr.Dropdown(
		label="Algorithm",
		choices=default_algos,
		value=default_algo,
	)
	algo_description = gr.Markdown(value=default_desc)

	return family_radio, algo_dropdown, algo_description


def on_family_change(family: str) -> dict:
	"""Callback when algorithm family selection changes.

	Updates the algorithm dropdown's choices and value to reflect the
	newly selected family.

	Args:
		family: Selected family display name, e.g. "On-Policy HA".

	Returns:
		gr.update dict for the algo_dropdown component.
	"""
	_, _, get_algo_choices, _ = _lazy_registry()
	choices = get_algo_choices(family)
	new_value = choices[0] if choices else None
	return gr.Dropdown(choices=choices, value=new_value)


def on_algo_change(algo_display_name: str) -> str:
	"""Callback when algorithm selection changes.

	Args:
		algo_display_name: Display name of the selected algorithm.

	Returns:
		Markdown string for the description card.
	"""
	if not algo_display_name:
		return "*Select an algorithm to see its description.*"
	return _format_algo_description(algo_display_name)


def _on_family_change_desc(family: str) -> str:
	"""Get the description for the first algorithm in a family.

	Used as a callback when the family radio changes, so the description
	card updates to reflect the new default algorithm.

	Args:
		family: Selected family display name.

	Returns:
		Markdown description string.
	"""
	_, _, get_algo_choices, _ = _lazy_registry()
	choices = get_algo_choices(family)
	if choices:
		return _format_algo_description(choices[0])
	return "*No algorithms in this family.*"


def bind_events(
	family_radio: gr.Radio,
	algo_dropdown: gr.Dropdown,
	algo_desc: gr.Markdown,
) -> None:
	"""Bind change events between the algorithm selector components.

	Wires up:
	- family_radio.change -> update algo_dropdown choices/value + update desc
	- algo_dropdown.change -> update desc

	Args:
		family_radio: The algorithm family radio button group.
		algo_dropdown: The algorithm dropdown.
		algo_desc: The algorithm description markdown.
	"""
	family_radio.change(
		fn=on_family_change,
		inputs=[family_radio],
		outputs=[algo_dropdown],
	)
	family_radio.change(
		fn=_on_family_change_desc,
		inputs=[family_radio],
		outputs=[algo_desc],
	)
	algo_dropdown.change(
		fn=on_algo_change,
		inputs=[algo_dropdown],
		outputs=[algo_desc],
	)
