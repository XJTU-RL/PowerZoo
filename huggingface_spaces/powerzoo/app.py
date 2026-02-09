"""
PowerZoo: Interactive Web Showcase
HuggingFace Spaces application with Gradio + Plotly.

5 Tabs: Project Overview | Power System Explorer | Data Visualization | Training Dashboard | Algorithm Comparison
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gradio as gr

# === Monkey-patch: fix Gradio additionalProperties schema error with Plotly ===
_original_plot_init = gr.Plot.__init__


def _patched_plot_init(self, *args, **kwargs):
	_original_plot_init(self, *args, **kwargs)
	if hasattr(self, "schema") and isinstance(self.schema, dict):
		self.schema.pop("additionalProperties", None)


gr.Plot.__init__ = _patched_plot_init

# === Data Loading ===
DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"

with open(DATA_DIR / "loadshapes.json") as f:
	LOADSHAPES = json.load(f)

with open(DATA_DIR / "pv_sample.json") as f:
	PV_DATA = json.load(f)

with open(DATA_DIR / "environments.json") as f:
	ENVIRONMENTS = json.load(f)

with open(DATA_DIR / "algorithms.json") as f:
	ALGORITHMS = json.load(f)

with open(DATA_DIR / "sample_training.json") as f:
	TRAINING = json.load(f)

# Architecture diagram JSON data (Plotly figures)
ARCH_FIGS = {}
_ARCH_NAMES = [
	"algorithm_hierarchy", "training_pipeline", "runner_algorithm_matrix",
	"happo_family", "mappo_family", "dan_happo",
	"ddpg_family", "hasac", "value_decomposition", "twots_vvc",
]
for fig_name in _ARCH_NAMES:
	fig_path = DATA_DIR / f"{fig_name}.json"
	if fig_path.exists():
		ARCH_FIGS[fig_name] = go.Figure(json.loads(fig_path.read_text()))

# === Color Palette ===
COLORS = {
	"primary": "#1565c0",
	"secondary": "#5e35b1",
	"accent": "#2e7d32",
	"warning": "#e65100",
	"agents": ["#1565c0", "#5e35b1", "#2e7d32", "#e65100", "#c62828", "#00838f"],
	"loadshapes": ["#1565c0", "#e65100", "#2e7d32"],
}

# === Init figures (required for gr.Plot(value=fig) pattern) ===
_INIT_FIG = go.Figure()
_INIT_FIG.update_layout(
	template="plotly_white",
	height=400,
	margin=dict(l=40, r=40, t=40, b=40),
)


# ============================================================
# Plot Factory Functions
# ============================================================

def plot_load_profiles(selected_shapes: list[str]) -> go.Figure:
	"""Plot annual load curves for selected LoadShapes."""
	if not selected_shapes:
		fig = go.Figure()
		fig.update_layout(template="plotly_white", height=450)
		fig.add_annotation(text="Select at least one load shape", showarrow=False, font=dict(size=16))
		return fig

	fig = go.Figure()
	for i, name in enumerate(selected_shapes):
		if name in LOADSHAPES:
			data = LOADSHAPES[name]
			# X-axis: approximate hours across the year (730 points, each ~12h apart)
			x = np.linspace(0, 8760, len(data))
			fig.add_trace(go.Scatter(
				x=x, y=data,
				mode="lines",
				name=name,
				line=dict(color=COLORS["loadshapes"][i % 3], width=1.5),
				opacity=0.85,
			))

	fig.update_layout(
		template="plotly_white",
		height=450,
		title="Annual Load Profiles (8760 hours)",
		xaxis_title="Hour of Year",
		yaxis_title="Load (p.u.)",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		hovermode="x unified",
	)
	return fig


def plot_load_daily_stats(shape_name: str) -> go.Figure:
	"""Plot daily mean +/- std band for a single LoadShape (reshape 365x24)."""
	if shape_name not in LOADSHAPES:
		fig = go.Figure()
		fig.update_layout(template="plotly_white", height=450)
		fig.add_annotation(text="Select a load shape", showarrow=False, font=dict(size=16))
		return fig

	raw = LOADSHAPES[shape_name]
	# Upsample back to ~8760 via linear interpolation for reshape
	full = np.interp(np.arange(8760), np.linspace(0, 8759, len(raw)), raw)
	daily = full.reshape(365, 24)
	mean = daily.mean(axis=0)
	std = daily.std(axis=0)
	hours = list(range(24))

	fig = go.Figure()
	# Std band
	fig.add_trace(go.Scatter(
		x=hours + hours[::-1],
		y=np.concatenate([mean + std, (mean - std)[::-1]]).tolist(),
		fill="toself",
		fillcolor="rgba(21, 101, 192, 0.15)",
		line=dict(color="rgba(0,0,0,0)"),
		showlegend=True,
		name="Std Dev Band",
	))
	# Mean line
	fig.add_trace(go.Scatter(
		x=hours, y=mean.tolist(),
		mode="lines+markers",
		name="Daily Mean",
		line=dict(color=COLORS["primary"], width=2.5),
		marker=dict(size=5),
	))

	fig.update_layout(
		template="plotly_white",
		height=450,
		title=f"{shape_name} - Daily Average Pattern",
		xaxis_title="Hour of Day",
		yaxis_title="Load (p.u.)",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_pv_timeseries() -> go.Figure:
	"""Plot 7-day PV irradiance time series."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=PV_DATA["timestamps"],
		y=PV_DATA["irradiance_wm2"],
		mode="lines",
		name="Irradiance",
		line=dict(color=COLORS["warning"], width=1.2),
		fill="tozeroy",
		fillcolor="rgba(230, 81, 0, 0.1)",
	))
	fig.update_layout(
		template="plotly_white",
		height=450,
		title="7-Day Solar Irradiance (Jan 2025, 10-min resolution)",
		xaxis_title="Timestamp",
		yaxis_title="Irradiance (W/m²)",
	)
	return fig


def plot_pv_scatter() -> go.Figure:
	"""Plot irradiance vs temperature scatter, colored by hour of day."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=PV_DATA["temperature_c"],
		y=PV_DATA["irradiance_wm2"],
		mode="markers",
		marker=dict(
			size=4,
			color=PV_DATA["hours"],
			colorscale="Viridis",
			colorbar=dict(title="Hour"),
			opacity=0.6,
		),
		text=[f"Hour: {h:.1f}" for h in PV_DATA["hours"]],
		hovertemplate="Temp: %{x:.1f}°C<br>Irradiance: %{y:.0f} W/m²<br>%{text}<extra></extra>",
	))
	fig.update_layout(
		template="plotly_white",
		height=450,
		title="Irradiance vs Temperature (colored by hour of day)",
		xaxis_title="Temperature (°C)",
		yaxis_title="Irradiance (W/m²)",
	)
	return fig


def plot_training_rewards() -> go.Figure:
	"""Plot episode rewards training curve."""
	if "episode_rewards" not in TRAINING:
		return _INIT_FIG

	data = TRAINING["episode_rewards"]
	steps = [pt[0] for pt in data]
	values = [pt[1] for pt in data]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=steps, y=values,
		mode="lines+markers",
		name="Episode Rewards",
		line=dict(color=COLORS["primary"], width=2),
		marker=dict(size=4),
	))
	fig.update_layout(
		template="plotly_white",
		height=450,
		title="HAPPO on 13Bus - Episode Rewards",
		xaxis_title="Training Steps",
		yaxis_title="Total Reward",
	)
	return fig


def plot_training_metric(metric_name: str) -> go.Figure:
	"""Plot per-agent comparison for a selected metric."""
	fig = go.Figure()
	for agent_id in range(6):
		key = f"agent{agent_id}_{metric_name}"
		if key in TRAINING:
			data = TRAINING[key]
			steps = [pt[0] for pt in data]
			values = [pt[1] for pt in data]
			fig.add_trace(go.Scatter(
				x=steps, y=values,
				mode="lines+markers",
				name=f"Agent {agent_id}",
				line=dict(color=COLORS["agents"][agent_id], width=1.8),
				marker=dict(size=4),
			))

	display_name = metric_name.replace("_", " ").title()
	fig.update_layout(
		template="plotly_white",
		height=450,
		title=f"Per-Agent {display_name}",
		xaxis_title="Training Steps",
		yaxis_title=display_name,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_power_metrics() -> go.Figure:
	"""Plot 2x2 subplot: power_loss_kw, power_loss_kvar, total_power_kw, total_power_kvar."""
	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"Power Loss (kW)", "Power Loss (kVar)",
			"Total Power (kW)", "Total Power (kVar)",
		),
		vertical_spacing=0.12,
		horizontal_spacing=0.1,
	)

	metrics = [
		("power_loss_kw", 1, 1, COLORS["primary"]),
		("power_loss_kvar", 1, 2, COLORS["secondary"]),
		("total_power_kw", 2, 1, COLORS["accent"]),
		("total_power_kvar", 2, 2, COLORS["warning"]),
	]

	for key, row, col, color in metrics:
		if key in TRAINING:
			data = TRAINING[key]
			steps = [pt[0] for pt in data]
			values = [pt[1] for pt in data]
			fig.add_trace(
				go.Scatter(
					x=steps, y=values,
					mode="lines+markers",
					line=dict(color=color, width=2),
					marker=dict(size=4),
					showlegend=False,
				),
				row=row, col=col,
			)

	fig.update_layout(
		template="plotly_white",
		height=550,
		title_text="Power System Metrics During Training",
	)
	return fig


# ============================================================
# Environment Explorer Helpers
# ============================================================

def get_env_names() -> list[str]:
	"""Return list of environment names."""
	return list(ENVIRONMENTS.keys())


def get_system_names(env_name: str) -> gr.Dropdown:
	"""Update system dropdown based on selected environment."""
	if env_name and env_name in ENVIRONMENTS:
		systems = list(ENVIRONMENTS[env_name]["systems"].keys())
		return gr.Dropdown(choices=systems, value=systems[0] if systems else None)
	return gr.Dropdown(choices=[], value=None)


def get_env_info(env_name: str) -> str:
	"""Return environment description as markdown."""
	if not env_name or env_name not in ENVIRONMENTS:
		return "Select an environment to view details."

	env = ENVIRONMENTS[env_name]
	md = f"### {env_name}\n\n"
	md += f"**Description**: {env['description']}\n\n"
	md += f"**Action Space**: {env['action_space']}\n\n"
	md += f"**Observation**: {env['observation']}\n\n"
	md += f"**Reward**: {env['reward']}\n\n"
	md += "**Key Features**:\n"
	for feat in env["features"]:
		md += f"- {feat}\n"
	return md


def get_system_table(env_name: str, system_name: str) -> pd.DataFrame:
	"""Return system configuration as a DataFrame."""
	if not env_name or not system_name:
		return pd.DataFrame({"Property": ["Select environment and system"], "Value": ["-"]})

	env = ENVIRONMENTS.get(env_name, {})
	system = env.get("systems", {}).get(system_name, {})
	if not system:
		return pd.DataFrame({"Property": ["System not found"], "Value": ["-"]})

	rows = []
	for key, val in system.items():
		if key == "name":
			continue
		display_key = key.replace("_", " ").title()
		rows.append({"Property": display_key, "Value": str(val)})
	return pd.DataFrame(rows)


# ============================================================
# Algorithm Table
# ============================================================

def get_algorithm_df() -> pd.DataFrame:
	"""Return algorithm comparison DataFrame."""
	return pd.DataFrame([
		{
			"Algorithm": a["name"],
			"Type": a["type"],
			"Policy": a["policy"],
			"Action Space": a["action_space"],
			"Key Feature": a["key_feature"],
		}
		for a in ALGORITHMS
	])


# ============================================================
# Build Gradio App
# ============================================================

def build_app() -> gr.Blocks:
	"""Construct the Gradio Blocks application with 5 tabs."""
	with gr.Blocks(
		title="PowerZoo: MARL for Power Systems",
		theme=gr.themes.Soft(
			primary_hue="blue",
			secondary_hue="purple",
		),
	) as app:
		# Header
		gr.Markdown(
			"""
			# ⚡ PowerZoo: A Universal MARL Platform for Power System Control
			**4 Environments** · **15 Algorithms** · **9 IEEE Test Systems** · **IEEE TSG 2025**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# Tab 1: Project Overview
			# --------------------------------------------------------
			with gr.Tab("Project Overview"):
				gr.Markdown(
					"""
					## About PowerZoo

					PowerZoo is a comprehensive multi-agent reinforcement learning (MARL) platform
					designed for intelligent power system control. It provides a unified interface
					for training and evaluating MARL algorithms across diverse power system environments.

					### Key Highlights
					"""
				)

				with gr.Row():
					with gr.Column(scale=1):
						gr.Markdown(
							"""
							**🏗️ 4 Environments**
							- PowerZoo VVC (Volt-VAR Control)
							- SmartGrid (PV Integration)
							- Stackelberg Game (Market)
							- DSR (Fault Recovery)
							"""
						)
					with gr.Column(scale=1):
						gr.Markdown(
							"""
							**🤖 15 MARL Algorithms**
							- On-Policy: HAPPO, HATRPO, MAPPO, etc.
							- Off-Policy: HADDPG, HASAC, MADDPG, etc.
							- Value-Based: QMIX, HAD3QN
							- Special: 2TS-VVC, DAN-HAPPO
							"""
						)
					with gr.Column(scale=1):
						gr.Markdown(
							"""
							**🔌 9 IEEE Test Systems**
							- 13-Bus, 34-Bus, 123-Bus, 8500-Node
							- PV variants (Conservative/Optimized/Aggressive)
							- From rapid prototyping to scalability testing
							"""
						)

				# Architecture SVG
				svg_path = ASSETS_DIR / "architecture.svg"
				if svg_path.exists():
					svg_content = svg_path.read_text()
					gr.HTML(
						f'<div style="text-align:center; margin:20px 0; overflow-x:auto;">'
						f'{svg_content}'
						f'</div>'
					)

				gr.Markdown(
					"""
					### Citation

					> PowerZoo: A Universal Multi-Agent Reinforcement Learning Platform for Power System Control.
					> *IEEE Transactions on Smart Grid*, 2025.

					**Links**: [GitHub](https://github.com/XJTU-RL/PowerZoo) ·
					[Paper](https://ieeexplore.ieee.org/)
					"""
				)

			# --------------------------------------------------------
			# Tab 2: Power System Explorer
			# --------------------------------------------------------
			with gr.Tab("Power System Explorer"):
				gr.Markdown("## Explore Environments & IEEE Test Systems")

				with gr.Row():
					env_dropdown = gr.Dropdown(
						choices=get_env_names(),
						label="Select Environment",
						value=get_env_names()[0],
					)
					system_dropdown = gr.Dropdown(
						choices=[],
						label="Select IEEE System",
					)

				env_info = gr.Markdown(value="Select an environment to view details.")
				system_table = gr.Dataframe(
					headers=["Property", "Value"],
					label="System Configuration",
				)

				# Wire events
				env_dropdown.change(
					fn=get_system_names,
					inputs=env_dropdown,
					outputs=system_dropdown,
				)
				env_dropdown.change(
					fn=get_env_info,
					inputs=env_dropdown,
					outputs=env_info,
				)
				system_dropdown.change(
					fn=get_system_table,
					inputs=[env_dropdown, system_dropdown],
					outputs=system_table,
				)

				# Trigger initial load
				app.load(
					fn=get_system_names,
					inputs=env_dropdown,
					outputs=system_dropdown,
				)
				app.load(
					fn=get_env_info,
					inputs=env_dropdown,
					outputs=env_info,
				)

			# --------------------------------------------------------
			# Tab 3: Data Visualization
			# --------------------------------------------------------
			with gr.Tab("Data Visualization"):
				with gr.Tabs():
					# Sub-tab: Load Profiles
					with gr.Tab("Load Profiles"):
						gr.Markdown("### Annual Load Shape Visualization")
						load_checkbox = gr.CheckboxGroup(
							choices=list(LOADSHAPES.keys()),
							value=list(LOADSHAPES.keys()),
							label="Select Load Shapes",
						)
						load_annual_plot = gr.Plot(value=plot_load_profiles(list(LOADSHAPES.keys())))

						gr.Markdown("### Daily Average Pattern")
						load_radio = gr.Radio(
							choices=list(LOADSHAPES.keys()),
							value="LoadShape1",
							label="Select Load Shape for Daily Analysis",
						)
						load_daily_plot = gr.Plot(value=plot_load_daily_stats("LoadShape1"))

						load_checkbox.change(
							fn=plot_load_profiles,
							inputs=load_checkbox,
							outputs=load_annual_plot,
						)
						load_radio.change(
							fn=plot_load_daily_stats,
							inputs=load_radio,
							outputs=load_daily_plot,
						)

					# Sub-tab: PV Data
					with gr.Tab("PV Generation"):
						gr.Markdown("### Solar PV Data (January 2025, First 7 Days)")
						pv_ts_plot = gr.Plot(value=plot_pv_timeseries())
						gr.Markdown("### Irradiance-Temperature Correlation")
						pv_scatter_plot = gr.Plot(value=plot_pv_scatter())

			# --------------------------------------------------------
			# Tab 4: Training Dashboard
			# --------------------------------------------------------
			with gr.Tab("Training Dashboard"):
				gr.Markdown(
					"""
					## HAPPO Training on IEEE 13-Bus System
					Sample training metrics from a HAPPO experiment on the PowerZoo VVC environment.
					"""
				)

				rewards_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### Per-Agent Metrics")
				metric_dropdown = gr.Dropdown(
					choices=["policy_loss", "dist_entropy"],
					value="policy_loss",
					label="Select Metric",
				)
				agent_plot = gr.Plot(value=plot_training_metric("policy_loss"))

				metric_dropdown.change(
					fn=plot_training_metric,
					inputs=metric_dropdown,
					outputs=agent_plot,
				)

				gr.Markdown("### Power System Metrics")
				power_plot = gr.Plot(value=plot_power_metrics())

			# --------------------------------------------------------
			# Tab 5: Algorithm Comparison
			# --------------------------------------------------------
			with gr.Tab("Algorithm Comparison"):
				gr.Markdown(
					"""
					## 15 MARL Algorithms
					PowerZoo supports a comprehensive suite of multi-agent reinforcement learning algorithms
					spanning on-policy, off-policy, value-based, and specialized methods.
					"""
				)
				gr.Dataframe(
					value=get_algorithm_df(),
					label="Algorithm Feature Matrix",
					interactive=False,
				)

			# --------------------------------------------------------
			# Tab 6: Architecture Diagrams
			# --------------------------------------------------------
			with gr.Tab("Architecture Diagrams"):
				gr.Markdown(
					"""
					## Interactive Architecture Diagrams
					Explore the algorithm inheritance hierarchy, training pipeline flow,
					and runner-algorithm compatibility matrix.
					"""
				)

				if "algorithm_hierarchy" in ARCH_FIGS:
					gr.Markdown("### Algorithm Inheritance Hierarchy")
					gr.Plot(value=ARCH_FIGS["algorithm_hierarchy"])

				if "training_pipeline" in ARCH_FIGS:
					gr.Markdown("### Training Pipeline Flow")
					gr.Plot(value=ARCH_FIGS["training_pipeline"])

				if "runner_algorithm_matrix" in ARCH_FIGS:
					gr.Markdown("### Runner-Algorithm Compatibility Matrix")
					gr.Plot(value=ARCH_FIGS["runner_algorithm_matrix"])

				gr.Markdown("---\n## Algorithm Internal Architectures")

				_algo_details = [
					("happo_family", "HAPPO / HATRPO / HAA2C"),
					("mappo_family", "MAPPO / SN-MAPPO"),
					("dan_happo", "DAN-HAPPO"),
					("ddpg_family", "DDPG Family (HADDPG / HATD3 / MADDPG / MATD3)"),
					("hasac", "HASAC"),
					("value_decomposition", "QMIX / HAD3QN"),
					("twots_vvc", "2TS-VVC"),
				]
				for key, label in _algo_details:
					if key in ARCH_FIGS:
						gr.Markdown(f"### {label}")
						gr.Plot(value=ARCH_FIGS[key])

		# Footer
		gr.Markdown(
			"""
			---
			**PowerZoo** · MIT License · [XJTU-RL](https://github.com/XJTU-RL)
			· IEEE Transactions on Smart Grid, 2025
			"""
		)

	return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(
		server_name="0.0.0.0",
		server_port=7860,
		share=False,
		factory_reboot=True,
		allowed_paths=["assets"],
	)
