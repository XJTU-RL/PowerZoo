"""DSR (Distribution System Restoration) environment tab.

Corresponds to ``configs/envs_cfgs/dsr*.yaml``.  Provides controls for device
configuration, load aggregation, fault scenarios, physical constraints, reward
weights, priority weights, and advanced features specific to the distribution
restoration problem.
"""

import gradio as gr

from training_ui.tabs.base_tab import BaseEnvironmentTab


class DSRTab(BaseEnvironmentTab):
	"""Configuration tab for DSR environments.

	The ``env_name`` is dynamic -- it depends on the variant dropdown value
	(dsr, dsr_13bus, or dsr_8500node).
	"""

	@property
	def env_name(self) -> str:
		# Default fallback; actual value resolved via _env_variant component
		return "dsr"

	@property
	def tab_label(self) -> str:
		return "DSR"

	def build_env_params(self) -> dict[str, gr.components.Component]:
		"""Build DSR-specific environment parameters.

		Layout:
			- Environment variant selector
			- Episode settings
			- Device configuration (DG, PV, switch, load levels)
			- Load aggregation
			- Fault configuration
			- Physical constraints
			- Reward weights
			- Priority weights
			- Advanced features (action mask, dynamic network, action space)
			- Runtime flags
		"""
		# --- Variant selector ---
		env_variant = gr.Dropdown(
			label="DSR Variant",
			choices=["dsr", "dsr_13bus", "dsr_8500node"],
			value="dsr",
			info="dsr=123Bus, dsr_13bus=13Bus, dsr_8500node=8500Node",
		)

		# --- Episode Settings ---
		with gr.Accordion("Episode Settings", open=True):
			with gr.Row():
				max_episode_steps = gr.Number(
					label="Max Episode Steps",
					value=15,
					precision=0,
					info="dsr=15, 13bus=10, 8500node=20",
				)
				seed = gr.Number(
					label="Seed",
					value=123456,
					precision=0,
				)

		# --- Device Configuration ---
		with gr.Accordion("Device Configuration", open=True):
			with gr.Row():
				n_dg = gr.Slider(
					label="Diesel Generators (n_dg)",
					minimum=1,
					maximum=20,
					step=1,
					value=7,
					info="Black-start power sources. 13bus=3, dsr=7, 8500node=15",
				)
				n_pv = gr.Slider(
					label="PV Units (n_pv)",
					minimum=1,
					maximum=20,
					step=1,
					value=9,
					info="13bus=3, dsr=9, 8500node=20",
				)
			with gr.Row():
				n_switch = gr.Slider(
					label="Switches (n_switch)",
					minimum=5,
					maximum=50,
					step=1,
					value=20,
					info="13bus=10, dsr=20, 8500node=50",
				)
				n_load_levels = gr.Slider(
					label="Load Priority Levels",
					minimum=1,
					maximum=5,
					step=1,
					value=3,
				)

		# --- Load Aggregation ---
		with gr.Accordion("Load Aggregation", open=False):
			use_load_aggregation = gr.Checkbox(
				label="Use Load Aggregation",
				value=True,
				info="Aggregate individual loads into agent groups",
			)
			load_aggregation_method = gr.Dropdown(
				label="Aggregation Method",
				choices=["zone", "cluster", "random"],
				value="zone",
			)

		# --- Fault Configuration ---
		with gr.Accordion("Fault Configuration", open=False):
			with gr.Row():
				fault_scenarios = gr.Slider(
					label="Fault Scenarios",
					minimum=1,
					maximum=20,
					step=1,
					value=5,
					info="13bus=3, dsr=5, 8500node=10",
				)
			with gr.Row():
				min_faults = gr.Slider(
					label="Min Faults per Scenario",
					minimum=1,
					maximum=10,
					step=1,
					value=3,
					info="13bus=1, dsr=3, 8500node=5",
				)
				max_faults = gr.Slider(
					label="Max Faults per Scenario",
					minimum=1,
					maximum=10,
					step=1,
					value=5,
					info="13bus=3, dsr=5, 8500node=10",
				)

		# --- Physical Constraints ---
		with gr.Accordion("Physical Constraints", open=False):
			with gr.Row():
				v_min = gr.Slider(
					label="Voltage Min (p.u.)",
					minimum=0.90,
					maximum=1.00,
					step=0.01,
					value=0.95,
				)
				v_max = gr.Slider(
					label="Voltage Max (p.u.)",
					minimum=1.00,
					maximum=1.10,
					step=0.01,
					value=1.05,
				)
			max_load_per_step = gr.Number(
				label="Max Load per Step (kW)",
				value=500.0,
				info="13bus=200, dsr=500, 8500node=1000",
			)

		# --- Reward Weights ---
		with gr.Accordion("Reward Weights", open=False):
			with gr.Row():
				reward_restore = gr.Slider(
					label="Restore Reward",
					minimum=0.0,
					maximum=50.0,
					step=1.0,
					value=20.0,
				)
				reward_voltage = gr.Slider(
					label="Voltage Reward",
					minimum=0.0,
					maximum=10.0,
					step=0.1,
					value=1.0,
				)
			with gr.Row():
				reward_overload = gr.Slider(
					label="Overload Penalty",
					minimum=0.0,
					maximum=10.0,
					step=0.1,
					value=1.0,
				)
				reward_done = gr.Number(
					label="Done Penalty",
					value=-5.0,
					info="Penalty when episode ends without full restoration",
				)

		# --- Priority Weights ---
		with gr.Accordion("Priority Weights", open=False):
			gr.Markdown("Load priority weights by level (higher = more important)")
			with gr.Row():
				priority_1 = gr.Slider(
					label="Priority 1 (Critical)",
					minimum=0.0,
					maximum=10.0,
					step=0.5,
					value=3.0,
				)
				priority_2 = gr.Slider(
					label="Priority 2 (Important)",
					minimum=0.0,
					maximum=10.0,
					step=0.5,
					value=2.0,
				)
				priority_3 = gr.Slider(
					label="Priority 3 (Normal)",
					minimum=0.0,
					maximum=10.0,
					step=0.5,
					value=1.0,
				)

		# --- Advanced Features ---
		with gr.Accordion("Advanced Features", open=False):
			with gr.Row():
				use_action_mask = gr.Checkbox(
					label="Use Action Mask",
					value=True,
					info="Mask invalid actions in the action space",
				)
				use_dynamic_network = gr.Checkbox(
					label="Use Dynamic Network",
					value=True,
					info="Enable dynamic network topology changes",
				)
			gr.Markdown("#### Action Space Configuration")
			with gr.Row():
				pv_power_levels = gr.Slider(
					label="PV Power Levels",
					minimum=3,
					maximum=21,
					step=2,
					value=11,
				)
				load_action_levels = gr.Slider(
					label="Load Action Levels",
					minimum=2,
					maximum=10,
					step=1,
					value=2,
				)
				pv_max_power = gr.Number(
					label="PV Max Power (kW)",
					value=150.0,
					info="8500node=200",
				)
			gr.Markdown("#### Overload Detection")
			with gr.Row():
				overload_threshold = gr.Number(
					label="Overload Threshold",
					value=1.0,
				)
				line_disconnect_prob = gr.Slider(
					label="Line Disconnect Probability",
					minimum=0.0,
					maximum=1.0,
					step=0.05,
					value=0.7,
					info="Initial probability for line disconnection. 8500node=0.8",
				)

		# --- Runtime Flags ---
		with gr.Accordion("Runtime Flags", open=False):
			with gr.Row():
				use_render = gr.Checkbox(
					label="Render",
					value=False,
					info="Large systems: rendering not recommended",
				)
				record_node = gr.Checkbox(
					label="Record Node Info",
					value=True,
					info="8500node: disabled by default to save memory",
				)

		return {
			"_env_variant": env_variant,
			# Episode
			"max_episode_steps": max_episode_steps,
			"seed": seed,
			# Device configuration
			"n_dg": n_dg,
			"n_pv": n_pv,
			"n_switch": n_switch,
			"n_load_levels": n_load_levels,
			# Load aggregation
			"use_load_aggregation": use_load_aggregation,
			"load_aggregation_method": load_aggregation_method,
			# Fault configuration
			"fault_scenarios": fault_scenarios,
			"min_faults": min_faults,
			"max_faults": max_faults,
			# Physical constraints
			"v_min": v_min,
			"v_max": v_max,
			"max_load_per_step": max_load_per_step,
			# Reward weights
			"reward_restore": reward_restore,
			"reward_voltage": reward_voltage,
			"reward_overload": reward_overload,
			"reward_done": reward_done,
			# Priority weights
			"priority_1": priority_1,
			"priority_2": priority_2,
			"priority_3": priority_3,
			# Advanced features
			"use_action_mask": use_action_mask,
			"use_dynamic_network": use_dynamic_network,
			"pv_power_levels": pv_power_levels,
			"load_action_levels": load_action_levels,
			"pv_max_power": pv_max_power,
			"overload_threshold": overload_threshold,
			"line_disconnect_prob": line_disconnect_prob,
			# Runtime
			"use_render": use_render,
			"record_node": record_node,
		}
