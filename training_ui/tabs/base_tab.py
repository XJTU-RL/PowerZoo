"""Abstract base class for environment configuration tabs.

Provides the standard layout shared by all environment tabs:
	Row 1: Algorithm selector (family radio + algo dropdown)
	Sub-tabs:
		1. Environment Config - env-specific parameters
		2. Algorithm Config - network / on-policy / off-policy panels
		3. Training Settings - training params + seed/device + evaluation
		4. Launch - experiment name, config preview, validation, start button

Subclasses implement ``build_env_params()`` to populate the Environment Config
sub-tab with env-specific Gradio components.
"""

import json
from abc import ABC, abstractmethod

import gradio as gr
import yaml


class BaseEnvironmentTab(ABC):
	"""Base class for all environment configuration tabs.

	Concrete subclasses must define ``env_name``, ``tab_label``, and
	``build_env_params()``.  Everything else (algorithm selection, algorithm
	config, training settings, launch panel) is provided by the base class.
	"""

	# ------------------------------------------------------------------
	# Abstract interface
	# ------------------------------------------------------------------

	@property
	@abstractmethod
	def env_name(self) -> str:
		"""Environment identifier used for registry lookup and config files."""

	@property
	@abstractmethod
	def tab_label(self) -> str:
		"""Display label rendered on the Gradio tab header."""

	@abstractmethod
	def build_env_params(self) -> dict[str, gr.components.Component]:
		"""Build environment-specific parameter panels.

		Returns:
			Dict mapping parameter names to Gradio components.  The keys
			should match the YAML config keys consumed by ``config_builder``.
		"""

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	def build(self) -> gr.Tab:
		"""Build the complete environment tab and wire up events.

		Returns:
			The constructed ``gr.Tab`` instance.
		"""
		with gr.Tab(label=self.tab_label) as tab:
			# --- Algorithm selection row ---
			with gr.Row():
				with gr.Column(scale=1):
					from training_ui.components.algo_selector import (
						bind_events as bind_algo_events,
						build_algo_selector,
					)
					family_radio, algo_dropdown, algo_desc = build_algo_selector()
					bind_algo_events(family_radio, algo_dropdown, algo_desc)

			# --- Sub-tabs ---
			with gr.Tabs():
				# Sub-tab 1: Environment Config
				with gr.Tab("Environment Config"):
					env_params = self.build_env_params()

				# Sub-tab 2: Algorithm Config
				with gr.Tab("Algorithm Config"):
					algo_params = self._build_algo_config()

				# Sub-tab 3: Training Settings
				with gr.Tab("Training Settings"):
					training_params = self._build_training_config()

				# Sub-tab 4: Launch
				with gr.Tab("Launch"):
					launch = self._build_launch_panel()

			# --- Bind launch-panel events ---
			self._bind_launch_events(
				launch=launch,
				algo_dropdown=algo_dropdown,
				env_params=env_params,
				algo_params=algo_params,
				training_params=training_params,
			)

		return tab

	# ------------------------------------------------------------------
	# Shared sub-tab builders
	# ------------------------------------------------------------------

	def _build_algo_config(self) -> dict:
		"""Build algorithm configuration panels (network + on/off-policy)."""
		from training_ui.components.param_panels import (
			build_network_panel,
			build_off_policy_algo_panel,
			build_on_policy_algo_panel,
		)
		network = build_network_panel()
		on_policy = build_on_policy_algo_panel()
		off_policy = build_off_policy_algo_panel()
		return {"network": network, "on_policy": on_policy, "off_policy": off_policy}

	def _build_training_config(self) -> dict:
		"""Build training settings panels (seed/device + training + eval)."""
		from training_ui.components.param_panels import (
			build_eval_panel,
			build_seed_device_panel,
			build_training_panel,
		)
		seed_device = build_seed_device_panel()
		training = build_training_panel()
		evaluation = build_eval_panel()
		return {"seed_device": seed_device, "training": training, "eval": evaluation}

	def _build_launch_panel(self) -> dict:
		"""Build launch panel with config preview and start button.

		Returns:
			Dict of launch-related Gradio components.
		"""
		with gr.Row():
			with gr.Column(scale=1):
				exp_name = gr.Textbox(
					label="Experiment Name",
					value="test",
					placeholder="Enter experiment name...",
				)
				model_dir = gr.Textbox(
					label="Model Directory (resume training)",
					value="",
					placeholder="Leave empty for new training",
				)
				log_dir = gr.Textbox(
					label="Log Directory",
					value="./results",
				)
			with gr.Column(scale=1):
				config_preview = gr.Code(
					label="Config Preview (YAML)",
					language="yaml",
					interactive=False,
					lines=20,
				)

		with gr.Row():
			preview_btn = gr.Button("Preview Config", variant="secondary")
			validate_btn = gr.Button("Validate", variant="secondary")
			launch_btn = gr.Button("Start Training", variant="primary", size="lg")

		status_msg = gr.Markdown("")

		return {
			"exp_name": exp_name,
			"model_dir": model_dir,
			"log_dir": log_dir,
			"config_preview": config_preview,
			"preview_btn": preview_btn,
			"validate_btn": validate_btn,
			"launch_btn": launch_btn,
			"status_msg": status_msg,
		}

	# ------------------------------------------------------------------
	# Event wiring
	# ------------------------------------------------------------------

	def _bind_launch_events(
		self,
		launch: dict,
		algo_dropdown: gr.Dropdown,
		env_params: dict,
		algo_params: dict,
		training_params: dict,
	) -> None:
		"""Wire up preview / validate / launch button callbacks.

		This collects all Gradio component references from the various panels
		so they can be passed as ``inputs`` to ``gr.Button.click()``.
		"""
		# Flatten all leaf components (skip nested dicts and internal keys)
		env_components = self._flatten_components(env_params)
		algo_components = self._flatten_components(algo_params)
		training_components = self._flatten_components(training_params)

		all_inputs = [
			algo_dropdown,
			launch["exp_name"],
			launch["model_dir"],
			launch["log_dir"],
			*env_components,
			*algo_components,
			*training_components,
		]

		# Build component key maps for reconstruction
		env_keys = self._flatten_keys(env_params)
		algo_keys = self._flatten_keys(algo_params)
		training_keys = self._flatten_keys(training_params)

		# --- Preview callback ---
		def _on_preview(*values):
			return self._preview_callback(
				values, env_keys, algo_keys, training_keys,
			)

		launch["preview_btn"].click(
			fn=_on_preview,
			inputs=all_inputs,
			outputs=[launch["config_preview"]],
		)

		# --- Validate callback ---
		def _on_validate(*values):
			return self._validate_callback(
				values, env_keys, algo_keys, training_keys,
			)

		launch["validate_btn"].click(
			fn=_on_validate,
			inputs=all_inputs,
			outputs=[launch["status_msg"]],
		)

		# --- Launch callback ---
		def _on_launch(*values):
			return self._launch_callback(
				values, env_keys, algo_keys, training_keys,
			)

		launch["launch_btn"].click(
			fn=_on_launch,
			inputs=all_inputs,
			outputs=[launch["status_msg"]],
		)

	# ------------------------------------------------------------------
	# Callback implementations
	# ------------------------------------------------------------------

	def _preview_callback(
		self,
		values: tuple,
		env_keys: list[str],
		algo_keys: list[str],
		training_keys: list[str],
	) -> str:
		"""Generate a YAML preview string from current component values."""
		try:
			config = self._values_to_config(values, env_keys, algo_keys, training_keys)
			return yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
		except Exception as exc:
			return f"# Error generating preview:\n# {exc}"

	def _validate_callback(
		self,
		values: tuple,
		env_keys: list[str],
		algo_keys: list[str],
		training_keys: list[str],
	) -> str:
		"""Run basic validation on the assembled config."""
		try:
			config = self._values_to_config(values, env_keys, algo_keys, training_keys)
			errors: list[str] = []

			if not config.get("algo_name"):
				errors.append("No algorithm selected.")
			if not config.get("exp_name"):
				errors.append("Experiment name is empty.")

			env_args = config.get("env_args", {})
			episode_length = env_args.get("episode_length")
			if episode_length is not None and episode_length <= 0:
				errors.append("Episode length must be positive.")

			if errors:
				return "**Validation Failed**\n\n" + "\n".join(f"- {e}" for e in errors)
			return "**Validation Passed** -- config looks good."
		except Exception as exc:
			return f"**Validation Error**: {exc}"

	def _launch_callback(
		self,
		values: tuple,
		env_keys: list[str],
		algo_keys: list[str],
		training_keys: list[str],
	) -> str:
		"""Build config, persist to disk, and start a training subprocess."""
		try:
			config = self._values_to_config(values, env_keys, algo_keys, training_keys)

			algo_name = config.get("algo_name", "")
			env_name = config.get("env_name", self.env_name)
			exp_name = config.get("exp_name", "test")

			if not algo_name:
				return "**Error**: No algorithm selected."

			from training_ui.core.config_builder import build_config
			config_path, full_config = build_config(
				algo=algo_name,
				env=env_name,
				exp_name=exp_name,
				ui_overrides=config.get("overrides", {}),
			)

			from training_ui.core.process_manager import process_manager
			task_id = process_manager.start(
				algo=algo_name,
				env=env_name,
				exp_name=exp_name,
				config_path=config_path,
			)

			return (
				f"**Training Started**\n\n"
				f"- Task ID: `{task_id}`\n"
				f"- Algorithm: `{algo_name}`\n"
				f"- Environment: `{env_name}`\n"
				f"- Config: `{config_path}`"
			)
		except Exception as exc:
			return f"**Launch Error**: {exc}"

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _resolve_env_name(self, env_params_values: dict) -> str:
		"""Resolve the actual env name, handling dynamic variants.

		Subclasses with a ``_env_variant`` component (Stackelberg, DSR) will
		use the dropdown value; others fall back to ``self.env_name``.
		"""
		return env_params_values.get("_env_variant", self.env_name)

	def _values_to_config(
		self,
		values: tuple,
		env_keys: list[str],
		algo_keys: list[str],
		training_keys: list[str],
	) -> dict:
		"""Reassemble a flat tuple of component values into a config dict.

		The value ordering mirrors: [algo_dropdown, exp_name, model_dir,
		log_dir, *env_values, *algo_values, *training_values].
		"""
		idx = 0

		algo_display = values[idx]; idx += 1
		exp_name = values[idx]; idx += 1
		model_dir = values[idx]; idx += 1
		log_dir = values[idx]; idx += 1

		env_vals = {}
		for key in env_keys:
			env_vals[key] = values[idx]; idx += 1

		algo_vals = {}
		for key in algo_keys:
			algo_vals[key] = values[idx]; idx += 1

		training_vals = {}
		for key in training_keys:
			training_vals[key] = values[idx]; idx += 1

		# Resolve algorithm internal name from display name
		algo_name = self._resolve_algo_name(algo_display)
		resolved_env = self._resolve_env_name(env_vals)

		return {
			"algo_name": algo_name,
			"env_name": resolved_env,
			"exp_name": exp_name,
			"model_dir": model_dir,
			"log_dir": log_dir,
			"env_args": env_vals,
			"overrides": {
				"algo": algo_vals,
				"train": training_vals,
			},
		}

	@staticmethod
	def _resolve_algo_name(display_name: str) -> str:
		"""Map an algorithm display name back to its internal name."""
		from training_ui.core.registry import ALGO_REGISTRY
		for meta in ALGO_REGISTRY.values():
			if meta.display_name == display_name:
				return meta.name
		return display_name.lower() if display_name else ""

	@staticmethod
	def _flatten_components(params: dict) -> list[gr.components.Component]:
		"""Extract leaf Gradio components from a possibly nested dict.

		Skips keys starting with ``_`` (internal references like ``_panel``).
		"""
		components: list[gr.components.Component] = []
		for key, val in params.items():
			if key.startswith("_"):
				continue
			if isinstance(val, dict):
				components.extend(BaseEnvironmentTab._flatten_components(val))
			else:
				components.append(val)
		return components

	@staticmethod
	def _flatten_keys(params: dict) -> list[str]:
		"""Extract leaf key names from a possibly nested dict.

		Nested dicts produce dot-separated keys, e.g. ``"network.lr"``.
		Skips keys starting with ``_``.
		"""
		keys: list[str] = []
		for key, val in params.items():
			if key.startswith("_"):
				continue
			if isinstance(val, dict):
				for sub_key in BaseEnvironmentTab._flatten_keys(val):
					keys.append(f"{key}.{sub_key}")
			else:
				keys.append(key)
		return keys
