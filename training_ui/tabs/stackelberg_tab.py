"""Stackelberg game environment tab.

Corresponds to ``configs/envs_cfgs/stackelberg_*.yaml``.  The most complex
environment tab, featuring dynamic variant selection (13bus / 34bus / 123bus)
and extensive configuration panels for agent setup, action spaces, reward
weights, physical constraints, market/TOU/ESS/DER/DR parameters, and
carbon/N-1 security settings.
"""

import json

import gradio as gr

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
		return "Stackelberg"

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
			label="Stackelberg Variant",
			choices=[
				"stackelberg_13bus",
				"stackelberg_34bus",
				"stackelberg_123bus",
			],
			value="stackelberg_13bus",
			info="Selects the IEEE bus system and default agent count",
		)

		# --- Agent Configuration ---
		with gr.Accordion("Agent Configuration", open=True):
			n_consumer_agents = gr.Slider(
				label="Number of Consumer Agents",
				minimum=1,
				maximum=50,
				step=1,
				value=8,
				info="13Bus=8, 34Bus=10, 123Bus=20",
			)
			max_episode_steps = gr.Number(
				label="Max Episode Steps",
				value=24,
				precision=0,
				info="24-hour episodes by default",
			)
			seed = gr.Number(
				label="Seed",
				value=42,
				precision=0,
			)
			consumer_bus_mapping = gr.Code(
				label="Consumer Bus Mapping (JSON)",
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
		with gr.Accordion("UC Action Space", open=False):
			with gr.Row():
				price_signal_low = gr.Number(label="Price Signal Low", value=0.5)
				price_signal_high = gr.Number(label="Price Signal High", value=2.0)
			with gr.Row():
				dr_incentive_low = gr.Number(label="DR Incentive Low", value=0.0)
				dr_incentive_high = gr.Number(label="DR Incentive High", value=0.5)
			with gr.Row():
				capacity_alloc_low = gr.Number(label="Capacity Allocation Low", value=0.0)
				capacity_alloc_high = gr.Number(label="Capacity Allocation High", value=1.0)
			with gr.Row():
				ess_charge_low = gr.Number(label="ESS Charge Low", value=-1.0)
				ess_charge_high = gr.Number(label="ESS Charge High", value=1.0)
			with gr.Row():
				der_curtail_low = gr.Number(label="DER Curtailment Low", value=0.0)
				der_curtail_high = gr.Number(label="DER Curtailment High", value=1.0)

		# --- Consumer Action Space ---
		with gr.Accordion("Consumer Action Space", open=False):
			with gr.Row():
				load_adj_low = gr.Number(
					label="Load Adjustment Low",
					value=-0.3,
					info="Maximum load reduction (30%)",
				)
				load_adj_high = gr.Number(
					label="Load Adjustment High",
					value=0.1,
					info="Maximum load increase (10%)",
				)
			with gr.Row():
				der_output_low = gr.Number(label="DER Output Low", value=0.0)
				der_output_high = gr.Number(label="DER Output High", value=1.0)

		# --- Reward Weights ---
		with gr.Accordion("Reward Weights", open=False):
			gr.Markdown("#### UC Rewards")
			with gr.Row():
				uc_electricity_revenue = gr.Slider(
					label="Electricity Revenue",
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				uc_market_cost = gr.Slider(
					label="Market Cost",
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				uc_der_profit = gr.Slider(
					label="DER Profit",
					minimum=0.0, maximum=5.0, step=0.1, value=0.8,
				)
			with gr.Row():
				uc_dr_cost = gr.Slider(
					label="DR Cost",
					minimum=0.0, maximum=5.0, step=0.1, value=0.6,
				)
				uc_system_loss = gr.Slider(
					label="System Loss",
					minimum=0.0, maximum=5.0, step=0.1, value=0.5,
				)
				uc_voltage_violation = gr.Slider(
					label="Voltage Violation",
					minimum=0.0, maximum=10.0, step=0.1, value=2.0,
				)
			with gr.Row():
				uc_carbon_reduction = gr.Slider(
					label="Carbon Reduction",
					minimum=0.0, maximum=5.0, step=0.1, value=0.3,
				)

			gr.Markdown("#### Consumer Rewards")
			with gr.Row():
				con_electricity_cost = gr.Slider(
					label="Electricity Cost",
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				con_comfort_loss = gr.Slider(
					label="Comfort Loss",
					minimum=0.0, maximum=5.0, step=0.1, value=0.8,
				)
			with gr.Row():
				con_dr_revenue = gr.Slider(
					label="DR Revenue",
					minimum=0.0, maximum=5.0, step=0.1, value=1.2,
				)
				con_voltage_quality = gr.Slider(
					label="Voltage Quality",
					minimum=0.0, maximum=5.0, step=0.1, value=0.3,
				)

		# --- Physical Constraints ---
		with gr.Accordion("Physical Constraints", open=False):
			with gr.Row():
				voltage_min = gr.Slider(
					label="Voltage Min (p.u.)",
					minimum=0.90, maximum=1.00, step=0.01, value=0.95,
				)
				voltage_max = gr.Slider(
					label="Voltage Max (p.u.)",
					minimum=1.00, maximum=1.10, step=0.01, value=1.05,
				)
			with gr.Row():
				line_capacity_factor = gr.Slider(
					label="Line Capacity Factor",
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)
				max_load_change_rate = gr.Slider(
					label="Max Load Change Rate",
					minimum=0.01, maximum=0.5, step=0.01, value=0.1,
				)
				ess_ramp_rate = gr.Slider(
					label="ESS Ramp Rate",
					minimum=0.01, maximum=0.5, step=0.01, value=0.2,
				)

		# --- Market Configuration ---
		with gr.Accordion("Market Configuration", open=False):
			with gr.Row():
				base_price = gr.Number(
					label="Base Price ($/kWh)",
					value=0.10,
				)
				peak_multiplier = gr.Number(
					label="Peak Multiplier",
					value=2.0,
				)
			with gr.Row():
				valley_multiplier = gr.Number(
					label="Valley Multiplier",
					value=0.5,
				)
				market_volatility = gr.Number(
					label="Market Volatility",
					value=0.03,
				)

		# --- TOU Configuration ---
		with gr.Accordion("TOU Configuration", open=False):
			peak_hours = gr.CheckboxGroup(
				label="Peak Hours",
				choices=[str(h) for h in range(24)],
				value=[str(h) for h in [8, 9, 10, 11, 17, 18, 19, 20]],
			)
			valley_hours = gr.CheckboxGroup(
				label="Valley Hours",
				choices=[str(h) for h in range(24)],
				value=[str(h) for h in [0, 1, 2, 3, 4, 5, 23]],
			)

		# --- ESS Configuration ---
		with gr.Accordion("ESS Configuration", open=False):
			with gr.Row():
				ess_total_capacity = gr.Number(
					label="Total Capacity (MWh)",
					value=2.0,
					info="13Bus=2.0, 34Bus=4.0, 123Bus=10.0",
				)
				ess_initial_soc = gr.Slider(
					label="Initial SOC",
					minimum=0.0, maximum=1.0, step=0.05, value=0.5,
				)
			with gr.Row():
				ess_efficiency_charge = gr.Slider(
					label="Charge Efficiency",
					minimum=0.80, maximum=1.00, step=0.01, value=0.95,
				)
				ess_efficiency_discharge = gr.Slider(
					label="Discharge Efficiency",
					minimum=0.80, maximum=1.00, step=0.01, value=0.95,
				)
			with gr.Row():
				ess_self_discharge = gr.Number(
					label="Self Discharge Rate",
					value=0.001,
				)
				ess_max_power = gr.Number(
					label="Max Power (MW)",
					value=0.5,
					info="13Bus=0.5, 34Bus=1.0, 123Bus=2.0",
				)
			with gr.Row():
				ess_min_soc = gr.Slider(
					label="Min SOC",
					minimum=0.0, maximum=0.5, step=0.05, value=0.2,
				)
				ess_max_soc = gr.Slider(
					label="Max SOC",
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)

		# --- DER Configuration ---
		with gr.Accordion("DER Configuration", open=False):
			with gr.Row():
				der_total_capacity = gr.Number(
					label="Total Capacity (MW)",
					value=3.0,
					info="13Bus=3.0, 34Bus=5.0, 123Bus=12.0",
				)
				der_availability = gr.Dropdown(
					label="Availability Profile",
					choices=["solar", "wind", "constant"],
					value="solar",
				)
			with gr.Row():
				der_forecast_error = gr.Number(
					label="Forecast Error Std",
					value=0.1,
				)
				der_curtailment_cost = gr.Number(
					label="Curtailment Cost ($/kWh)",
					value=0.02,
				)

		# --- DR Configuration ---
		with gr.Accordion("DR Configuration", open=False):
			with gr.Row():
				dr_max_ratio = gr.Slider(
					label="Max DR Ratio",
					minimum=0.0, maximum=1.0, step=0.05, value=0.3,
				)
				dr_min_response = gr.Slider(
					label="Min Response Time (h)",
					minimum=1, maximum=12, step=1, value=1,
				)
			with gr.Row():
				dr_fatigue = gr.Slider(
					label="Fatigue Factor",
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)
				dr_participation = gr.Slider(
					label="Participation Rate",
					minimum=0.0, maximum=1.0, step=0.05, value=0.8,
				)

		# --- Carbon & N-1 Security ---
		with gr.Accordion("Carbon & N-1 Security", open=False):
			gr.Markdown("#### Carbon Tracking")
			with gr.Row():
				track_emissions = gr.Checkbox(label="Track Emissions", value=True)
				grid_carbon_intensity = gr.Number(
					label="Grid Carbon Intensity (kg CO2/kWh)",
					value=0.5,
				)
				carbon_price = gr.Number(
					label="Carbon Price ($/kg CO2)",
					value=0.02,
				)
			gr.Markdown("#### N-1 Security")
			with gr.Row():
				n1_enable = gr.Checkbox(label="Enable N-1 Security", value=True)
				contingency_prob = gr.Number(
					label="Contingency Probability",
					value=0.001,
				)
				recovery_time = gr.Number(
					label="Recovery Time (h)",
					value=4,
					precision=0,
				)

		# --- Runtime Flags ---
		with gr.Accordion("Runtime Flags", open=False):
			with gr.Row():
				debug = gr.Checkbox(label="Debug", value=False)
				verbose = gr.Checkbox(label="Verbose", value=True)
				render = gr.Checkbox(label="Render", value=False)

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
