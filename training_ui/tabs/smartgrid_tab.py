"""SmartGrid environment tab.

Corresponds to ``configs/envs_cfgs/smartgrid.yaml``.  Extends the base VVC
controls with reward weight overrides, voltage constraint overrides, curriculum
learning settings, and LLM-enhanced features.
"""

import gradio as gr

from training_ui.tabs.base_tab import BaseEnvironmentTab


class SmartGridTab(BaseEnvironmentTab):
	"""Configuration tab for the SmartGrid environment."""

	@property
	def env_name(self) -> str:
		return "smartgrid"

	@property
	def tab_label(self) -> str:
		return "SmartGrid"

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
		with gr.Accordion("SmartGrid Environment Settings", open=True):
			with gr.Row():
				episode_length = gr.Number(
					label="Episode Length",
					value=360,
					precision=0,
					info="Maximum steps per episode (360 = 15min resolution over 24h)",
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
				)
				useS = gr.Checkbox(label="Use S Matrix (SHOM)", value=False)
				big2small = gr.Checkbox(label="Big to Small (SHOM)", value=False)

		# --- Reward Weights ---
		with gr.Accordion("Reward Weights", open=True):
			gr.Markdown(
				"Override system default reward weights. "
				"Values from `environment_specific.reward_weights` in YAML."
			)
			with gr.Row():
				rw_power_loss = gr.Slider(
					label="Power Loss",
					minimum=0.0,
					maximum=10.0,
					step=0.1,
					value=1.0,
				)
				rw_capacitor = gr.Slider(
					label="Capacitor",
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0303,
				)
				rw_regulator = gr.Slider(
					label="Regulator",
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0303,
				)
			with gr.Row():
				rw_battery_soc = gr.Slider(
					label="Battery SOC",
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0,
				)
				rw_battery_discharge = gr.Slider(
					label="Battery Discharge",
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.303,
				)
				rw_pv_control = gr.Slider(
					label="PV Control",
					minimum=0.0,
					maximum=1.0,
					step=0.01,
					value=0.0606,
				)

		# --- Voltage Constraints ---
		with gr.Accordion("Voltage Constraints", open=False):
			with gr.Row():
				voltage_min = gr.Slider(
					label="Voltage Min (p.u.)",
					minimum=0.90,
					maximum=1.00,
					step=0.01,
					value=0.95,
				)
				voltage_max = gr.Slider(
					label="Voltage Max (p.u.)",
					minimum=1.00,
					maximum=1.10,
					step=0.01,
					value=1.05,
				)
			with gr.Row():
				voltage_penalty_scale = gr.Slider(
					label="Voltage Penalty Scale",
					minimum=0.0,
					maximum=5.0,
					step=0.1,
					value=1.0,
				)
				constraint_aware = gr.Checkbox(
					label="Constraint Aware",
					value=True,
				)

		# --- Curriculum Learning ---
		with gr.Accordion("Curriculum Learning", open=False):
			curriculum_learning = gr.Checkbox(
				label="Enable Curriculum Learning",
				value=True,
			)
			with gr.Row():
				voltage_violation_penalty = gr.Number(
					label="Voltage Violation Penalty",
					value=50.0,
				)
				power_loss_weight = gr.Number(
					label="Power Loss Weight",
					value=5.0,
				)
				control_cost_weight = gr.Number(
					label="Control Cost Weight",
					value=0.05,
				)

		# --- Runtime Flags ---
		with gr.Accordion("Runtime Flags", open=False):
			with gr.Row():
				use_render = gr.Checkbox(label="Render", value=False)
				record_node = gr.Checkbox(label="Record Node Info", value=False)
				dss_act = gr.Checkbox(
					label="DSS Auto Control",
					value=False,
					info="If enabled, OpenDSS controls override RL actions",
				)
			with gr.Row():
				llm_enhanced = gr.Checkbox(
					label="LLM Enhanced",
					value=False,
					info="Enable LLM-enhanced observations and action explanations",
				)

		# --- Logging ---
		with gr.Accordion("Logging", open=False):
			with gr.Row():
				enable_system_logging = gr.Checkbox(label="System Logging", value=True)
				enable_realtime_log = gr.Checkbox(label="Realtime Log", value=True)
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
