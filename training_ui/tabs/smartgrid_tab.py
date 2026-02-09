"""SmartGrid environment tab.

Corresponds to ``configs/envs_cfgs/smartgrid.yaml``.  Extends the base VVC
controls with reward weight overrides, voltage constraint overrides, curriculum
learning settings, and LLM-enhanced features.
"""

import gradio as gr

from training_ui.i18n import t
from training_ui.tabs.base_tab import BaseEnvironmentTab


class SmartGridTab(BaseEnvironmentTab):
	"""Configuration tab for the SmartGrid environment."""

	@property
	def env_name(self) -> str:
		return "smartgrid"

	@property
	def tab_label(self) -> str:
		return t("tab_smartgrid")

	def build_env_params(self) -> dict[str, gr.components.Component]:
		"""Build SmartGrid-specific environment parameters.

		Layout:
			- System selector (with PV variant support)
			- Base environment settings
			- Reward weights (sliders matching environment_specific defaults)
			- Voltage constraints
			- Curriculum learning
			- Runtime flags (including LLM-enhanced)
			- Logging
		"""
		from training_ui.components.system_selector import (
			build_system_selector,
			bind_events as bind_system_events,
		)

		# --- System selector ---
		system_dropdown, system_info = build_system_selector(
			available_systems=[
				"13Bus", "34Bus", "34Bus_PV",
				"34Bus_PV_Aggressive", "34Bus_PV_Conservative", "34Bus_PV_Optimized",
				"123Bus", "8500Node",
			],
			default="34Bus_PV_Aggressive",
		)
		bind_system_events(system_dropdown, system_info)

		# --- Base Environment Settings ---
		with gr.Accordion(t("smartgrid_accordion_settings"), open=True):
			with gr.Row():
				episode_length = gr.Number(
					label=t("label_episode_length"),
					value=360,
					precision=0,
					info=t("info_smartgrid_episode"),
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
				)
				useS = gr.Checkbox(label=t("label_use_s_matrix"), value=False)
				big2small = gr.Checkbox(label=t("label_big2small"), value=False)

		# --- Reward Weights ---
		with gr.Accordion(t("accordion_reward_weights"), open=True):
			gr.Markdown(t("reward_weights_desc"))
			with gr.Row():
				rw_power_loss = gr.Slider(
					label=t("label_power_loss"),
					minimum=0.0,
					maximum=10.0,
					step=0.1,
					value=1.0,
				)
				rw_capacitor = gr.Slider(
					label=t("label_capacitor"),
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0303,
				)
				rw_regulator = gr.Slider(
					label=t("label_regulator"),
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0303,
				)
			with gr.Row():
				rw_battery_soc = gr.Slider(
					label=t("label_battery_soc"),
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0,
				)
				rw_battery_discharge = gr.Slider(
					label=t("label_battery_discharge"),
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.303,
				)
				rw_pv_control = gr.Slider(
					label=t("label_pv_control"),
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0606,
				)

		# --- Voltage Constraints ---
		with gr.Accordion(t("accordion_voltage_constraints"), open=False):
			with gr.Row():
				voltage_min = gr.Slider(
					label=t("label_voltage_min"),
					minimum=0.90,
					maximum=1.00,
					step=0.01,
					value=0.95,
				)
				voltage_max = gr.Slider(
					label=t("label_voltage_max"),
					minimum=1.00,
					maximum=1.10,
					step=0.01,
					value=1.05,
				)
			with gr.Row():
				voltage_penalty_scale = gr.Slider(
					label=t("label_voltage_penalty_scale"),
					minimum=0.0,
					maximum=5.0,
					step=0.1,
					value=1.0,
				)
				constraint_aware = gr.Checkbox(
					label=t("label_constraint_aware"),
					value=True,
				)

		# --- Curriculum Learning ---
		with gr.Accordion(t("accordion_curriculum"), open=False):
			curriculum_learning = gr.Checkbox(
				label=t("label_enable_curriculum"),
				value=True,
			)
			with gr.Row():
				voltage_violation_penalty = gr.Number(
					label=t("label_voltage_violation_penalty"),
					value=50.0,
				)
				power_loss_weight = gr.Number(
					label=t("label_power_loss_weight"),
					value=5.0,
				)
				control_cost_weight = gr.Number(
					label=t("label_control_cost_weight"),
					value=0.05,
				)

		# --- Runtime Flags ---
		with gr.Accordion(t("accordion_runtime_flags"), open=False):
			with gr.Row():
				use_render = gr.Checkbox(label=t("label_render"), value=False)
				record_node = gr.Checkbox(label=t("label_record_node"), value=False)
				dss_act = gr.Checkbox(
					label=t("label_dss_auto_control"),
					value=False,
					info=t("info_dss_auto_control"),
				)
			with gr.Row():
				llm_enhanced = gr.Checkbox(
					label=t("label_llm_enhanced"),
					value=False,
					info=t("info_llm_enhanced"),
				)

		# --- Logging ---
		with gr.Accordion(t("accordion_logging"), open=False):
			with gr.Row():
				enable_system_logging = gr.Checkbox(label=t("label_system_logging"), value=True)
				enable_realtime_log = gr.Checkbox(label=t("label_realtime_log"), value=True)
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
			# Reward weights
			"rw_power_loss": rw_power_loss,
			"rw_capacitor": rw_capacitor,
			"rw_regulator": rw_regulator,
			"rw_battery_soc": rw_battery_soc,
			"rw_battery_discharge": rw_battery_discharge,
			"rw_pv_control": rw_pv_control,
			# Voltage constraints
			"voltage_min": voltage_min,
			"voltage_max": voltage_max,
			"voltage_penalty_scale": voltage_penalty_scale,
			"constraint_aware": constraint_aware,
			# Curriculum learning
			"curriculum_learning": curriculum_learning,
			"voltage_violation_penalty": voltage_violation_penalty,
			"power_loss_weight": power_loss_weight,
			"control_cost_weight": control_cost_weight,
			# Runtime flags
			"use_render": use_render,
			"record_node": record_node,
			"dss_act": dss_act,
			"llm_enhanced": llm_enhanced,
			# Logging
			"enable_system_logging": enable_system_logging,
			"enable_realtime_log": enable_realtime_log,
			"system_log_dir": system_log_dir,
			"log_buffer_size": log_buffer_size,
			"log_save_interval": log_save_interval,
		}
