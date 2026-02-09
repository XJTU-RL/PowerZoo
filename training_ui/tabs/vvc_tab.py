"""VVC (Volt-VAR Control) environment tab.

Corresponds to ``configs/envs_cfgs/vvc.yaml``.  Provides controls for the
base VVC environment including system selection, episode settings, runtime
flags, and logging configuration.
"""

import gradio as gr

from training_ui.i18n import t
from training_ui.tabs.base_tab import BaseEnvironmentTab


class VVCTab(BaseEnvironmentTab):
	"""Configuration tab for the VVC environment."""

	@property
	def env_name(self) -> str:
		return "vvc"

	@property
	def tab_label(self) -> str:
		return t("tab_vvc")

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
		with gr.Accordion(t("vvc_accordion_settings"), open=True):
			with gr.Row():
				episode_length = gr.Number(
					label=t("label_episode_length"),
					value=24,
					precision=0,
					info=t("info_episode_length"),
				)
				seed = gr.Number(
					label=t("label_env_seed"),
					value=123456,
					precision=0,
				)
			with gr.Row():
				mode = gr.Dropdown(
					label=t("label_mode"),
					choices=["single", "parallel", "episodic", "dss"],
					value="single",
					info=t("info_mode"),
				)
				useS = gr.Checkbox(
					label=t("label_use_s_matrix"),
					value=False,
				)
				big2small = gr.Checkbox(
					label=t("label_big2small"),
					value=False,
				)

		# --- Runtime Flags ---
		with gr.Accordion(t("accordion_runtime_flags"), open=False):
			with gr.Row():
				use_render = gr.Checkbox(label=t("label_render"), value=False)
				use_plot = gr.Checkbox(label=t("label_plot"), value=False)
				do_testing = gr.Checkbox(label=t("label_testing_mode"), value=False)
			with gr.Row():
				record_node = gr.Checkbox(label=t("label_record_node"), value=False)
				dss_act = gr.Checkbox(
					label=t("label_dss_auto_control"),
					value=False,
					info=t("info_dss_auto_control"),
				)

		# --- Logging ---
		with gr.Accordion(t("accordion_logging"), open=False):
			with gr.Row():
				enable_system_logging = gr.Checkbox(
					label=t("label_system_logging"),
					value=True,
				)
				enable_realtime_log = gr.Checkbox(
					label=t("label_realtime_log"),
					value=True,
				)
			system_log_dir = gr.Textbox(
				label=t("label_log_directory"),
				value="./logs/system_params",
			)
			with gr.Row():
				log_buffer_size = gr.Number(
					label=t("label_log_buffer_size"),
					value=5000,
					precision=0,
				)
				log_save_interval = gr.Number(
					label=t("label_log_save_interval"),
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
