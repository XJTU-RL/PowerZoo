"""District Dispatch environment tab.

Corresponds to ``configs/envs_cfgs/district_dispatch.yaml``.  Provides controls
for the multi-zone coordinated dispatch environment including zone configuration,
reward weights, physical constraints, and market parameters.
"""

import gradio as gr

from training_ui.i18n import t
from training_ui.tabs.base_tab import BaseEnvironmentTab


class DistrictDispatchTab(BaseEnvironmentTab):
	"""Configuration tab for the District Dispatch environment."""

	@property
	def env_name(self) -> str:
		return "district_dispatch"

	@property
	def tab_label(self) -> str:
		return t("tab_district_dispatch")

	def build_env_params(self) -> dict[str, gr.components.Component]:
		"""Build District Dispatch-specific environment parameters.

		Layout:
			- Zone configuration (districts, connection, episode)
			- Observation settings
			- Physical constraints (voltage, SOC)
			- Market parameters (price, carbon)
			- Reward weights (6 components)
			- Runtime flags
		"""
		# --- Zone Configuration ---
		with gr.Accordion(t("dd_accordion_zone_config"), open=True):
			with gr.Row():
				n_districts = gr.Number(
					label=t("dd_label_n_districts"),
					value=3,
					precision=0,
					info=t("dd_info_n_districts"),
				)
				connection_mode = gr.Dropdown(
					label=t("dd_label_connection_mode"),
					choices=["transformer", "tieline", "mixed"],
					value="mixed",
					info=t("dd_info_connection_mode"),
				)
			with gr.Row():
				max_episode_steps = gr.Number(
					label=t("label_max_episode_steps"),
					value=96,
					precision=0,
					info=t("dd_info_episode_steps"),
				)
				seed = gr.Number(
					label=t("label_env_seed"),
					value=42,
					precision=0,
				)

		# --- Observation Settings ---
		with gr.Accordion(t("dd_accordion_observation"), open=False):
			with gr.Row():
				use_neighbor_obs = gr.Checkbox(
					label=t("dd_label_use_neighbor_obs"),
					value=True,
					info=t("dd_info_use_neighbor_obs"),
				)
				max_neighbors = gr.Number(
					label=t("dd_label_max_neighbors"),
					value=2,
					precision=0,
				)

		# --- Physical Constraints ---
		with gr.Accordion(t("dd_accordion_physical"), open=False):
			gr.Markdown(t("dd_heading_voltage"))
			with gr.Row():
				v_min = gr.Slider(
					label=t("label_voltage_min"),
					minimum=0.90, maximum=1.00, step=0.01, value=0.95,
				)
				v_max = gr.Slider(
					label=t("label_voltage_max"),
					minimum=1.00, maximum=1.10, step=0.01, value=1.05,
				)
			gr.Markdown(t("dd_heading_soc"))
			with gr.Row():
				soc_min = gr.Slider(
					label=t("dd_label_soc_min"),
					minimum=0.0, maximum=0.5, step=0.05, value=0.1,
				)
				soc_max = gr.Slider(
					label=t("dd_label_soc_max"),
					minimum=0.5, maximum=1.0, step=0.05, value=0.9,
				)
			with gr.Row():
				soc_init = gr.Slider(
					label=t("dd_label_soc_init"),
					minimum=0.0, maximum=1.0, step=0.05, value=0.5,
				)
				charge_efficiency = gr.Slider(
					label=t("dd_label_charge_eff"),
					minimum=0.8, maximum=1.0, step=0.01, value=0.95,
				)

		# --- Market Parameters ---
		with gr.Accordion(t("dd_accordion_market"), open=False):
			with gr.Row():
				base_price = gr.Number(
					label=t("dd_label_base_price"),
					value=0.5,
				)
				carbon_intensity = gr.Number(
					label=t("dd_label_carbon_intensity"),
					value=0.6,
				)
				carbon_price = gr.Number(
					label=t("dd_label_carbon_price"),
					value=0.05,
				)

		# --- Reward Weights ---
		with gr.Accordion(t("dd_accordion_reward_weights"), open=True):
			gr.Markdown(t("dd_reward_weights_desc"))
			with gr.Row():
				rw_economic = gr.Slider(
					label=t("dd_label_rw_economic"),
					minimum=0.0, maximum=5.0, step=0.1, value=1.0,
				)
				rw_voltage = gr.Slider(
					label=t("dd_label_rw_voltage"),
					minimum=0.0, maximum=5.0, step=0.1, value=2.0,
				)
				rw_loss = gr.Slider(
					label=t("dd_label_rw_loss"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.5,
				)
			with gr.Row():
				rw_carbon = gr.Slider(
					label=t("dd_label_rw_carbon"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.3,
				)
				rw_exchange = gr.Slider(
					label=t("dd_label_rw_exchange"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.2,
				)
				rw_storage = gr.Slider(
					label=t("dd_label_rw_storage"),
					minimum=0.0, maximum=5.0, step=0.1, value=0.1,
				)

		# --- Runtime Flags ---
		with gr.Accordion(t("accordion_runtime_flags"), open=False):
			with gr.Row():
				load_noise = gr.Checkbox(
					label=t("dd_label_load_noise"),
					value=True,
				)
				noise_std = gr.Slider(
					label=t("dd_label_noise_std"),
					minimum=0.0, maximum=0.2, step=0.01, value=0.05,
				)
			with gr.Row():
				use_render = gr.Checkbox(label=t("label_render"), value=False)
				debug_mode = gr.Checkbox(label=t("label_debug"), value=False)

		return {
			"n_districts": n_districts,
			"connection_mode": connection_mode,
			"max_episode_steps": max_episode_steps,
			"seed": seed,
			"use_neighbor_obs": use_neighbor_obs,
			"max_neighbors": max_neighbors,
			"v_min": v_min,
			"v_max": v_max,
			"soc_min": soc_min,
			"soc_max": soc_max,
			"soc_init": soc_init,
			"charge_efficiency": charge_efficiency,
			"base_price": base_price,
			"carbon_intensity": carbon_intensity,
			"carbon_price": carbon_price,
			"rw_economic": rw_economic,
			"rw_voltage": rw_voltage,
			"rw_loss": rw_loss,
			"rw_carbon": rw_carbon,
			"rw_exchange": rw_exchange,
			"rw_storage": rw_storage,
			"load_noise": load_noise,
			"noise_std": noise_std,
			"use_render": use_render,
			"debug_mode": debug_mode,
		}
