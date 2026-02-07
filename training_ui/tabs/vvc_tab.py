"""VVC (Volt-VAR Control) environment tab.

Corresponds to ``configs/envs_cfgs/vvc.yaml``.  Provides controls for the
base VVC environment including system selection, episode settings, runtime
flags, and logging configuration.
"""

import gradio as gr

from training_ui.tabs.base_tab import BaseEnvironmentTab


class VVCTab(BaseEnvironmentTab):
	"""Configuration tab for the VVC environment."""

	@property
	def env_name(self) -> str:
		return "vvc"

	@property
	def tab_label(self) -> str:
		return "VVC"

	def build_env_params(self) -> dict[str, gr.components.Component]:
		"""Build VVC-specific environment parameters.

		Layout:
			- System selector (dropdown + info card)
			- VVC environment settings (episode length, seed, mode, SHOM flags)
			- Runtime flags (render, plot, testing, etc.)
			- Logging configuration
		"""
		from training_ui.components.system_selector import (
			build_system_selector,
			bind_events as bind_system_events,
		)

		# --- System selector ---
		system_dropdown, system_info = build_system_selector(
			available_systems=["13Bus", "34Bus", "34Bus_PV", "123Bus"],
			default="13Bus",
		)
		bind_system_events(system_dropdown, system_info)

		# --- VVC Environment Settings ---
		with gr.Accordion("VVC Environment Settings", open=True):
			with gr.Row():
				episode_length = gr.Number(
					label="Episode Length",
					value=24,
					precision=0,
					info="Maximum steps per episode",
				)
				seed = gr.Number(
					label="Env Seed",
					value=123456,
					precision=0,
				)
			with gr.Row():
				mode = gr.Dropdown(
					label="Mode",
					choices=["single", "parallel", "episodic", "dss"],
					value="single",
					info="Environment execution mode",
				)
				useS = gr.Checkbox(
					label="Use S Matrix (SHOM)",
					value=False,
				)
				big2small = gr.Checkbox(
					label="Big to Small (SHOM)",
					value=False,
				)

		# --- Runtime Flags ---
		with gr.Accordion("Runtime Flags", open=False):
			with gr.Row():
				use_render = gr.Checkbox(label="Render", value=False)
				use_plot = gr.Checkbox(label="Plot", value=False)
				do_testing = gr.Checkbox(label="Testing Mode", value=False)
			with gr.Row():
				record_node = gr.Checkbox(label="Record Node Info", value=False)
				dss_act = gr.Checkbox(
					label="DSS Auto Control",
					value=False,
					info="If enabled, OpenDSS controls override RL actions",
				)

		# --- Logging ---
		with gr.Accordion("Logging", open=False):
			with gr.Row():
				enable_system_logging = gr.Checkbox(
					label="System Logging",
					value=True,
				)
				enable_realtime_log = gr.Checkbox(
					label="Realtime Log",
					value=True,
				)
			system_log_dir = gr.Textbox(
				label="Log Directory",
				value="./logs/system_params",
			)
			with gr.Row():
				log_buffer_size = gr.Number(
					label="Log Buffer Size",
					value=5000,
					precision=0,
				)
				log_save_interval = gr.Number(
					label="Log Save Interval",
					value=50,
					precision=0,
				)

		return {
			"system_ref": system_dropdown,
			"episode_length": episode_length,
			"seed": seed,
			"mode": mode,
			"useS": useS,
			"big2small": big2small,
			"use_render": use_render,
			"use_plot": use_plot,
			"do_testing": do_testing,
			"record_node": record_node,
			"dss_act": dss_act,
			"enable_system_logging": enable_system_logging,
			"enable_realtime_log": enable_realtime_log,
			"system_log_dir": system_log_dir,
			"log_buffer_size": log_buffer_size,
			"log_save_interval": log_save_interval,
		}
