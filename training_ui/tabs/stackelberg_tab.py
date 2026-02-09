"""Stackelberg game environment tab.

Corresponds to ``configs/envs_cfgs/stackelberg_*.yaml``.  The most complex
environment tab, featuring dynamic variant selection (13bus / 34bus / 123bus)
and extensive configuration panels for agent setup, action spaces, reward
weights, physical constraints, market/TOU/ESS/DER/DR parameters, and
carbon/N-1 security settings.
"""

import json

import gradio as gr

from training_ui.i18n import t
from training_ui.tabs.base_tab import BaseEnvironmentTab


class StackelbergTab(BaseEnvironmentTab):
	"""Configuration tab for Stackelberg game environments.

	The ``env_name`` is dynamic -- it depends on the variant dropdown value
	(stackelberg_13bus, stackelberg_34bus, or stackelberg_123bus).
	"""

	@property
	def env_name(self) -> str:
		# Default fallback; actual value resolved via _env_variant component
		return "stackelberg_13bus"

	@property
	def tab_label(self) -> str:
		return t("tab_stackelberg")

	def build_env_params(self) -> dict[str, gr.components.Component]:
		"""Build Stackelberg-specific environment parameters.

		Layout:
			- Environment variant selector
			- Agent configuration
			- UC / Consumer action spaces
			- Reward weights (UC + Consumer)
			- Physical constraints
			- Market configuration
			- TOU configuration
			- ESS configuration
			- DER configuration
			- DR configuration
			- Carbon & N-1 security
			- Runtime flags
		"""
		# --- Variant selector ---
		env_variant = gr.Dropdown(
			label=t("label_stackelberg_variant"),
			choices=[
				"stackelberg_13bus",
				"stackelberg_34bus",
				"stackelberg_123bus",
			],
			value="stackelberg_13bus",
			info=t("info_stackelberg_variant"),
		)

		# --- Agent Configuration ---
		with gr.Accordion(t("accordion_agent_config"), open=True):
			n_consumer_agents = gr.Slider(
				label=t("label_n_consumer_agents"),
				minimum=1,
				maximum=50,
				step=1,
				value=8,
				info="13Bus=8, 34Bus=10, 123Bus=20",
			)
			max_episode_steps = gr.Number(
				label=t("label_max_episode_steps"),
				value=24,
				precision=0,
				info="24-hour episodes by default",
			)
			seed = gr.Number(
				label=t("label_seed"),
				value=42,
				precision=0,
			)
			consumer_bus_mapping = gr.Code(
				label=t("label_consumer_bus_mapping"),
				value=json.dumps(
					{
						"0": [671, 675],
						"1": [692, 611],
						"2": [652, 632],
						"3": [633, 634],
						"4": [645, 646],
						"5": [680],
						"6": [684],
						"7": [650],
					},
					indent=2,
				),
				language="json",
				lines=10,
			)

		# --- UC Action Space ---
		with gr.Accordion(t("accordion_uc_action"), open=False):
			with gr.Row():
				price_signal_low = gr.Number(label=t("label_price_signal_low"), value=0.5)
				price_signal_high = gr.Number(label=t("label_price_signal_high"), value=2.0)
			with gr.Row():
				dr_incentive_low = gr.Number(label=t("label_dr_incentive_low"), value=0.0)
				dr_incentive_high = gr.Number(label=t("label_dr_incentive_high"), value=0.5)
			with gr.Row():
				capacity_alloc_low = gr.Number(label=t("label_capacity_alloc_low"), value=0.0)
				capacity_alloc_high = gr.Number(label=t("label_capacity_alloc_high"), value=1.0)
			with gr.Row():
				ess_charge_low = gr.Number(label=t("label_ess_charge_low"), value=-1.0)
				ess_charge_high = gr.Number(label=t("label_ess_charge_high"), value=1.0)
			with gr.Row():
				der_curtail_low = gr.Number(label=t("label_der_curtail_low"), value=0.0)
				der_curtail_high = gr.Number(label=t("label_der_curtail_high"), value=1.0)

		# --- Consumer Action Space ---
		with gr.Accordion(t("accordion_consumer_action"), open=False):
			with gr.Row():
				load_adj_low = gr.Number(
					label=t("label_load_adj_low"),
					value=-0.3,
					info=t("info_load_adj_low"),
				)
				load_adj_high = gr.Number(
					label=t("label_load_adj_high"),
					value=0.1,
					info=t("info_load_adj_high"),
				)
			with gr.Row():
				der_output_low = gr.Number(label=t("label_der_output_low"), value=0.0)
				der_output_high = gr.Number(label=t("label_der_output_high"), value=1.0)

		# --- Reward Weights ---
		with gr.Accordion(t("accordion_reward_weights_stk"), open=False):
			gr.Markdown(t("heading_uc_rewards"))
			with gr.Row():
				uc_electricity_revenue = gr.Slider(
					label=t("label_electricity_revenue"),
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				uc_market_cost = gr.Slider(
					label=t("label_market_cost"),
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				uc_der_profit = gr.Slider(
					label=t("label_der_profit"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.8,
				)
			with gr.Row():
				uc_dr_cost = gr.Slider(
					label=t("label_dr_cost"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.6,
				)
				uc_system_loss = gr.Slider(
					label=t("label_system_loss"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.5,
				)
				uc_voltage_violation = gr.Slider(
					label=t("label_voltage_violation"),
					minimum=0.0, maximum=10.0, step=0.1, value=2.0,
				)
			with gr.Row():
				uc_carbon_reduction = gr.Slider(
					label=t("label_carbon_reduction"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.3,
				)

			gr.Markdown(t("heading_consumer_rewards"))
			with gr.Row():
				con_electricity_cost = gr.Slider(
					label=t("label_electricity_cost"),
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				con_comfort_loss = gr.Slider(
					label=t("label_comfort_loss"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.8,
				)
			with gr.Row():
				con_dr_revenue = gr.Slider(
					label=t("label_dr_revenue"),
					minimum=0.0, maximum=5.0, step=0.1, value=1.2,
				)
				con_voltage_quality = gr.Slider(
					label=t("label_voltage_quality"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.3,
				)

		# --- Physical Constraints ---
		with gr.Accordion(t("accordion_physical_constraints"), open=False):
			with gr.Row():
				voltage_min = gr.Slider(
					label=t("label_voltage_min"),
					minimum=0.90, maximum=1.00, step=0.01, value=0.95,
				)
				voltage_max = gr.Slider(
					label=t("label_voltage_max"),
					minimum=1.00, maximum=1.10, step=0.01, value=1.05,
				)
			with gr.Row():
				line_capacity_factor = gr.Slider(
					label=t("label_line_capacity_factor"),
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)
				max_load_change_rate = gr.Slider(
					label=t("label_max_load_change_rate"),
					minimum=0.01, maximum=0.5, step=0.01, value=0.1,
				)
				ess_ramp_rate = gr.Slider(
					label=t("label_ess_ramp_rate"),
					minimum=0.01, maximum=0.5, step=0.01, value=0.2,
				)

		# --- Market Configuration ---
		with gr.Accordion(t("accordion_market"), open=False):
			with gr.Row():
				base_price = gr.Number(
					label=t("label_base_price"),
					value=0.10,
				)
				peak_multiplier = gr.Number(
					label=t("label_peak_multiplier"),
					value=2.0,
				)
			with gr.Row():
				valley_multiplier = gr.Number(
					label=t("label_valley_multiplier"),
					value=0.5,
				)
				market_volatility = gr.Number(
					label=t("label_market_volatility"),
					value=0.03,
				)

		# --- TOU Configuration ---
		with gr.Accordion(t("accordion_tou"), open=False):
			peak_hours = gr.CheckboxGroup(
				label=t("label_peak_hours"),
				choices=[str(h) for h in range(24)],
				value=[str(h) for h in [8, 9, 10, 11, 17, 18, 19, 20]],
			)
			valley_hours = gr.CheckboxGroup(
				label=t("label_valley_hours"),
				choices=[str(h) for h in range(24)],
				value=[str(h) for h in [0, 1, 2, 3, 4, 5, 23]],
			)

		# --- ESS Configuration ---
		with gr.Accordion(t("accordion_ess"), open=False):
			with gr.Row():
				ess_total_capacity = gr.Number(
					label=t("label_ess_total_capacity"),
					value=2.0,
					info="13Bus=2.0, 34Bus=4.0, 123Bus=10.0",
				)
				ess_initial_soc = gr.Slider(
					label=t("label_ess_initial_soc"),
					minimum=0.0, maximum=1.0, step=0.05, value=0.5,
				)
			with gr.Row():
				ess_efficiency_charge = gr.Slider(
					label=t("label_ess_charge_efficiency"),
					minimum=0.80, maximum=1.00, step=0.01, value=0.95,
				)
				ess_efficiency_discharge = gr.Slider(
					label=t("label_ess_discharge_efficiency"),
					minimum=0.80, maximum=1.00, step=0.01, value=0.95,
				)
			with gr.Row():
				ess_self_discharge = gr.Number(
					label=t("label_ess_self_discharge"),
					value=0.001,
				)
				ess_max_power = gr.Number(
					label=t("label_ess_max_power"),
					value=0.5,
					info="13Bus=0.5, 34Bus=1.0, 123Bus=2.0",
				)
			with gr.Row():
				ess_min_soc = gr.Slider(
					label=t("label_ess_min_soc"),
					minimum=0.0, maximum=0.5, step=0.05, value=0.2,
				)
				ess_max_soc = gr.Slider(
					label=t("label_ess_max_soc"),
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)

		# --- DER Configuration ---
		with gr.Accordion(t("accordion_der"), open=False):
			with gr.Row():
				der_total_capacity = gr.Number(
					label=t("label_der_total_capacity"),
					value=3.0,
					info="13Bus=3.0, 34Bus=5.0, 123Bus=12.0",
				)
				der_availability = gr.Dropdown(
					label=t("label_der_availability"),
					choices=["solar", "wind", "constant"],
					value="solar",
				)
			with gr.Row():
				der_forecast_error = gr.Number(
					label=t("label_der_forecast_error"),
					value=0.1,
				)
				der_curtailment_cost = gr.Number(
					label=t("label_der_curtailment_cost"),
					value=0.02,
				)

		# --- DR Configuration ---
		with gr.Accordion(t("accordion_dr"), open=False):
			with gr.Row():
				dr_max_ratio = gr.Slider(
					label=t("label_dr_max_ratio"),
					minimum=0.0, maximum=1.0, step=0.05, value=0.3,
				)
				dr_min_response = gr.Slider(
					label=t("label_dr_min_response"),
					minimum=1, maximum=12, step=1, value=1,
				)
			with gr.Row():
				dr_fatigue = gr.Slider(
					label=t("label_dr_fatigue"),
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)
				dr_participation = gr.Slider(
					label=t("label_dr_participation"),
					minimum=0.0, maximum=1.0, step=0.05, value=0.8,
				)

		# --- Carbon & N-1 Security ---
		with gr.Accordion(t("accordion_carbon_n1"), open=False):
			gr.Markdown(t("heading_carbon_tracking"))
			with gr.Row():
				track_emissions = gr.Checkbox(label=t("label_track_emissions"), value=True)
				grid_carbon_intensity = gr.Number(
					label=t("label_grid_carbon_intensity"),
					value=0.5,
				)
				carbon_price = gr.Number(
					label=t("label_carbon_price"),
					value=0.02,
				)
			gr.Markdown(t("heading_n1_security"))
			with gr.Row():
				n1_enable = gr.Checkbox(label=t("label_n1_enable"), value=True)
				contingency_prob = gr.Number(
					label=t("label_contingency_prob"),
					value=0.001,
				)
				recovery_time = gr.Number(
					label=t("label_recovery_time"),
					value=4,
					precision=0,
				)

		# --- Runtime Flags ---
		with gr.Accordion(t("accordion_runtime_flags"), open=False):
			with gr.Row():
				debug = gr.Checkbox(label=t("label_debug"), value=False)
				verbose = gr.Checkbox(label=t("label_verbose"), value=True)
				render = gr.Checkbox(label=t("label_render"), value=False)

		return {
			"_env_variant": env_variant,
			# Agent config
			"n_consumer_agents": n_consumer_agents,
			"max_episode_steps": max_episode_steps,
			"seed": seed,
			"consumer_bus_mapping": consumer_bus_mapping,
			# UC action space
			"price_signal_low": price_signal_low,
			"price_signal_high": price_signal_high,
			"dr_incentive_low": dr_incentive_low,
			"dr_incentive_high": dr_incentive_high,
			"capacity_alloc_low": capacity_alloc_low,
			"capacity_alloc_high": capacity_alloc_high,
			"ess_charge_low": ess_charge_low,
			"ess_charge_high": ess_charge_high,
			"der_curtail_low": der_curtail_low,
			"der_curtail_high": der_curtail_high,
			# Consumer action space
			"load_adj_low": load_adj_low,
			"load_adj_high": load_adj_high,
			"der_output_low": der_output_low,
			"der_output_high": der_output_high,
			# UC reward weights
			"uc_electricity_revenue": uc_electricity_revenue,
			"uc_market_cost": uc_market_cost,
			"uc_der_profit": uc_der_profit,
			"uc_dr_cost": uc_dr_cost,
			"uc_system_loss": uc_system_loss,
			"uc_voltage_violation": uc_voltage_violation,
			"uc_carbon_reduction": uc_carbon_reduction,
			# Consumer reward weights
			"con_electricity_cost": con_electricity_cost,
			"con_comfort_loss": con_comfort_loss,
			"con_dr_revenue": con_dr_revenue,
			"con_voltage_quality": con_voltage_quality,
			# Physical constraints
			"voltage_min": voltage_min,
			"voltage_max": voltage_max,
			"line_capacity_factor": line_capacity_factor,
			"max_load_change_rate": max_load_change_rate,
			"ess_ramp_rate": ess_ramp_rate,
			# Market config
			"base_price": base_price,
			"peak_multiplier": peak_multiplier,
			"valley_multiplier": valley_multiplier,
			"market_volatility": market_volatility,
			# TOU config
			"peak_hours": peak_hours,
			"valley_hours": valley_hours,
			# ESS config
			"ess_total_capacity": ess_total_capacity,
			"ess_initial_soc": ess_initial_soc,
			"ess_efficiency_charge": ess_efficiency_charge,
			"ess_efficiency_discharge": ess_efficiency_discharge,
			"ess_self_discharge": ess_self_discharge,
			"ess_max_power": ess_max_power,
			"ess_min_soc": ess_min_soc,
			"ess_max_soc": ess_max_soc,
			# DER config
			"der_total_capacity": der_total_capacity,
			"der_availability": der_availability,
			"der_forecast_error": der_forecast_error,
			"der_curtailment_cost": der_curtailment_cost,
			# DR config
			"dr_max_ratio": dr_max_ratio,
			"dr_min_response": dr_min_response,
			"dr_fatigue": dr_fatigue,
			"dr_participation": dr_participation,
			# Carbon
			"track_emissions": track_emissions,
			"grid_carbon_intensity": grid_carbon_intensity,
			"carbon_price": carbon_price,
			# N-1 security
			"n1_enable": n1_enable,
			"contingency_prob": contingency_prob,
			"recovery_time": recovery_time,
			# Runtime
			"debug": debug,
			"verbose": verbose,
			"render": render,
		}
