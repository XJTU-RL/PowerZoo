"""
PowerZoo District Dispatch: Multi-Zone Coordination Demo
HuggingFace Spaces application with Gradio + Plotly.

Self-contained demo for the District Dispatch MARL environment.
3 zone agents coordinate power dispatch across IEEE 34-Bus system.

5 Tabs: Overview | Zone Power Dispatch | Bus Voltage by Zone | Power Flow Sankey | Training Dashboard
"""
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

# === Color Palette ===
COLORS = {
	"primary": "#6366F1",
	"zone_a": "#6366F1",
	"zone_b": "#10B981",
	"zone_c": "#F59E0B",
	"accent": "#8B5CF6",
	"bg_dark": "#1a1a2e",
	"grid": "#2a2a4a",
	"text": "#e0e0e0",
	"safety_band": "rgba(16, 185, 129, 0.12)",
}

ZONE_NAMES = ["Zone A (Upstream)", "Zone B (Midstream)", "Zone C (Downstream)"]
ZONE_COLORS = [COLORS["zone_a"], COLORS["zone_b"], COLORS["zone_c"]]

# Plotly dark template defaults
LAYOUT_DEFAULTS = dict(
	template="plotly_dark",
	paper_bgcolor="rgba(0,0,0,0)",
	plot_bgcolor="rgba(26,26,46,0.8)",
	font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
	margin=dict(l=60, r=30, t=50, b=50),
)

HOURS = list(range(24))


# ============================================================
# Demo Data Generation
# ============================================================

def _make_rng(seed: int = 42) -> np.random.Generator:
	"""Create a seeded RNG for reproducible demo data."""
	return np.random.default_rng(seed)


def generate_load_profiles() -> dict[str, dict[str, np.ndarray]]:
	"""Generate realistic 24-hour load profiles for 3 zones across 3 scenarios.

	Scenarios: summer_peak, winter, mild_day
	Each zone has: load_kw, generation_kw, transfer_kw (inter-zone)
	"""
	rng = _make_rng(42)
	hours = np.arange(24, dtype=np.float64)

	# Base load shapes (normalized)
	# Summer: high midday AC load
	summer_base = 0.4 + 0.3 * np.sin(np.pi * (hours - 6) / 12)
	summer_base = np.where((hours >= 6) & (hours <= 20), summer_base + 0.2, summer_base)

	# Winter: morning/evening heating peaks
	winter_base = 0.5 + 0.25 * np.cos(np.pi * (hours - 18) / 6)
	winter_base = np.where((hours >= 6) & (hours <= 9), winter_base + 0.15, winter_base)
	winter_base = np.where((hours >= 17) & (hours <= 21), winter_base + 0.2, winter_base)

	# Mild: relatively flat
	mild_base = 0.35 + 0.15 * np.sin(np.pi * (hours - 8) / 14)

	# PV generation profile (bell curve centered at noon)
	pv_shape = np.exp(-0.5 * ((hours - 12.5) / 2.8) ** 2)
	pv_shape = np.clip(pv_shape, 0, 1)

	# Zone-specific scaling
	zone_load_scales = {
		"summer_peak": [850, 1100, 250],
		"winter": [700, 900, 200],
		"mild_day": [550, 750, 180],
	}
	zone_gen_scales = {
		"summer_peak": [380, 420, 140],
		"winter": [180, 200, 70],
		"mild_day": [300, 340, 110],
	}

	scenarios = {}
	for scenario, base in [("summer_peak", summer_base), ("winter", winter_base), ("mild_day", mild_base)]:
		zone_data = {}
		load_scales = zone_load_scales[scenario]
		gen_scales = zone_gen_scales[scenario]

		for z in range(3):
			noise = rng.normal(0, 0.02, 24)
			load_kw = base * load_scales[z] * (1 + noise + 0.05 * z * np.sin(np.pi * hours / 12))
			gen_kw = pv_shape * gen_scales[z] * (1 + rng.normal(0, 0.03, 24))
			gen_kw = np.clip(gen_kw, 0, None)

			# Inter-zone transfer: net surplus/deficit drives exchange
			net = gen_kw - load_kw
			# Zone A tends to export midday, Zone C imports
			transfer_scale = [0.15, 0.08, -0.2][z]
			transfer_kw = transfer_scale * load_kw + 0.3 * net + rng.normal(0, 10, 24)

			zone_data[f"zone_{z}"] = {
				"load_kw": load_kw,
				"generation_kw": gen_kw,
				"transfer_kw": transfer_kw,
			}
		scenarios[scenario] = zone_data

	return scenarios


def generate_bus_voltage_data() -> dict[str, list[dict]]:
	"""Generate 24-step voltage data for all 34-bus grouped by zone.

	Returns mapping: step -> list of {bus, zone, zone_idx, voltage}
	"""
	rng = _make_rng(123)

	zone_buses = {
		0: ["sourcebus", "800", "802", "806", "808", "810", "812", "814",
			"816", "818", "820", "822", "824", "826", "828", "830"],
		1: ["832", "834", "836", "838", "840", "842", "844", "846",
			"848", "850", "852", "854", "856", "858", "860", "862", "864"],
		2: ["888", "890"],
	}

	data = {}
	for step in range(24):
		records = []
		hour_factor = np.sin(np.pi * step / 24)

		for z_idx, buses in zone_buses.items():
			# Zone-level voltage drift: downstream zones have lower voltage
			zone_base = 1.01 - 0.012 * z_idx - 0.008 * hour_factor
			for i, bus in enumerate(buses):
				# Voltage drop along feeder
				position_drop = 0.001 * i
				noise = rng.normal(0, 0.004)
				# Midday PV injection raises voltage slightly
				pv_boost = 0.005 * np.exp(-0.5 * ((step - 12) / 3) ** 2) if z_idx < 2 else 0.002
				v = zone_base - position_drop + pv_boost + noise
				v = float(np.clip(v, 0.92, 1.08))
				records.append({
					"bus": bus,
					"zone": ZONE_NAMES[z_idx],
					"zone_idx": z_idx,
					"voltage": v,
				})
		data[str(step)] = records

	return data


def generate_sankey_data() -> dict[str, dict]:
	"""Generate power flow Sankey data for each step (0-23).

	Nodes: Substation, PV_A, PV_B, PV_C, Zone A, Zone B, Zone C, Load_A, Load_B, Load_C, Losses
	"""
	rng = _make_rng(77)
	hours = np.arange(24, dtype=np.float64)

	# PV output shape
	pv_shape = np.exp(-0.5 * ((hours - 12.5) / 2.8) ** 2)
	pv_shape = np.clip(pv_shape, 0, 1)

	# Load shape
	load_shape = 0.5 + 0.3 * np.sin(np.pi * (hours - 6) / 14)
	load_shape = np.where((hours >= 8) & (hours <= 20), load_shape + 0.15, load_shape)

	data = {}
	for step in range(24):
		pv_a = float(pv_shape[step] * 380 + rng.normal(0, 8))
		pv_b = float(pv_shape[step] * 420 + rng.normal(0, 10))
		pv_c = float(pv_shape[step] * 140 + rng.normal(0, 5))

		load_a = float(load_shape[step] * 850 + rng.normal(0, 15))
		load_b = float(load_shape[step] * 1100 + rng.normal(0, 20))
		load_c = float(load_shape[step] * 250 + rng.normal(0, 8))

		total_load = load_a + load_b + load_c
		total_pv = max(pv_a + pv_b + pv_c, 0)

		# Substation supplies the deficit
		substation = max(total_load - total_pv + rng.normal(30, 5), 50)

		# Losses (2-5% of total)
		loss_pct = 0.025 + 0.015 * load_shape[step]
		losses = float(total_load * loss_pct)

		# Ensure non-negative flows
		pv_a = max(pv_a, 0)
		pv_b = max(pv_b, 0)
		pv_c = max(pv_c, 0)
		load_a = max(load_a, 50)
		load_b = max(load_b, 50)
		load_c = max(load_c, 20)
		losses = max(losses, 5)

		# Zone inflows (from substation + PV)
		zone_a_in = substation * 0.35 + pv_a
		zone_b_in = substation * 0.45 + pv_b
		zone_c_in = substation * 0.20 + pv_c

		data[str(step)] = {
			"substation": float(substation),
			"pv_a": pv_a,
			"pv_b": pv_b,
			"pv_c": pv_c,
			"zone_a_in": float(zone_a_in),
			"zone_b_in": float(zone_b_in),
			"zone_c_in": float(zone_c_in),
			"load_a": load_a,
			"load_b": load_b,
			"load_c": load_c,
			"losses": losses,
		}

	return data


def generate_training_data() -> dict[str, np.ndarray]:
	"""Generate training dashboard data for ~1500 episodes, 3 agents.

	Returns:
		total_reward: (1500,) total team reward curve
		zone_rewards: (1500, 3) per-zone reward
		voltage_violation_rate: (1500,) violation rate over training
		power_balance_metric: (1500,) inter-zone balance metric
	"""
	rng = _make_rng(999)
	n_episodes = 1500
	episodes = np.arange(n_episodes)

	# Total reward: improves from ~-25 to ~-5 with noise
	progress = 1 - np.exp(-episodes / 400)
	total_reward = -25 + 20 * progress + rng.normal(0, 1.5, n_episodes)

	# Per-zone rewards (Zone A learns fastest, Zone C slowest)
	zone_rewards = np.zeros((n_episodes, 3))
	for z in range(3):
		speed = [350, 450, 600][z]
		base_bad = [-8, -10, -7][z]
		base_good = [-1.5, -2.0, -1.2][z]
		prog_z = 1 - np.exp(-episodes / speed)
		zone_rewards[:, z] = base_bad + (base_bad - base_good) * (-1) * prog_z + rng.normal(0, 0.6, n_episodes)

	# Voltage violation rate: drops from ~18% to ~2%
	vv_progress = 1 - np.exp(-episodes / 350)
	voltage_violation_rate = 0.18 - 0.16 * vv_progress + rng.normal(0, 0.01, n_episodes)
	voltage_violation_rate = np.clip(voltage_violation_rate, 0, 0.35)

	# Inter-zone power balance: improves from ~0.4 (unbalanced) to ~0.05 (balanced)
	pb_progress = 1 - np.exp(-episodes / 500)
	power_balance_metric = 0.4 - 0.35 * pb_progress + rng.normal(0, 0.02, n_episodes)
	power_balance_metric = np.clip(power_balance_metric, 0, 0.6)

	return {
		"episodes": episodes,
		"total_reward": total_reward,
		"zone_rewards": zone_rewards,
		"voltage_violation_rate": voltage_violation_rate,
		"power_balance_metric": power_balance_metric,
	}


# === Pre-generate all demo data ===
LOAD_PROFILES = generate_load_profiles()
BUS_VOLTAGES = generate_bus_voltage_data()
SANKEY_DATA = generate_sankey_data()
TRAINING_DATA = generate_training_data()

SCENARIO_LABELS = {
	"summer_peak": "Summer Peak",
	"winter": "Winter",
	"mild_day": "Mild Day",
}


# ============================================================
# Plot Factory Functions
# ============================================================

def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
	"""Simple moving average for training curves."""
	if len(arr) < window:
		return arr
	kernel = np.ones(window) / window
	return np.convolve(arr, kernel, mode="valid")


def plot_zone_power_dispatch(scenario: str) -> go.Figure:
	"""Create 3-row subplot for zone power dispatch.

	Each zone subplot has:
	- Area chart: zone load demand (kW)
	- Line plot: zone generation (kW)
	- Bar chart: inter-zone power transfer
	"""
	scenario_key = {v: k for k, v in SCENARIO_LABELS.items()}.get(scenario, "summer_peak")
	zone_data = LOAD_PROFILES.get(scenario_key, LOAD_PROFILES["summer_peak"])

	fig = make_subplots(
		rows=3, cols=1,
		subplot_titles=[f"{ZONE_NAMES[z]}" for z in range(3)],
		vertical_spacing=0.08,
		shared_xaxes=True,
	)

	for z in range(3):
		zd = zone_data[f"zone_{z}"]
		row = z + 1

		# Load demand (area fill)
		fig.add_trace(
			go.Scatter(
				x=HOURS, y=zd["load_kw"],
				mode="lines",
				name=f"Load ({ZONE_NAMES[z].split(' ')[0]} {ZONE_NAMES[z].split(' ')[1]})",
				fill="tozeroy",
				fillcolor=f"rgba({_hex_to_rgb(ZONE_COLORS[z])}, 0.15)",
				line=dict(color=ZONE_COLORS[z], width=2),
				legendgroup=f"zone_{z}",
				showlegend=(z == 0),
				hovertemplate="Hour %{x}: %{y:.0f} kW<extra>Load</extra>",
			),
			row=row, col=1,
		)

		# Generation (dashed line)
		fig.add_trace(
			go.Scatter(
				x=HOURS, y=zd["generation_kw"],
				mode="lines+markers",
				name="Generation" if z == 0 else None,
				line=dict(color="#F472B6", width=2, dash="dash"),
				marker=dict(size=4, color="#F472B6"),
				showlegend=(z == 0),
				hovertemplate="Hour %{x}: %{y:.0f} kW<extra>Generation</extra>",
			),
			row=row, col=1,
		)

		# Inter-zone transfer (bar)
		transfer = zd["transfer_kw"]
		bar_colors = [
			"rgba(16, 185, 129, 0.7)" if v >= 0 else "rgba(239, 68, 68, 0.7)"
			for v in transfer
		]
		fig.add_trace(
			go.Bar(
				x=HOURS, y=transfer,
				name="Transfer" if z == 0 else None,
				marker_color=bar_colors,
				showlegend=(z == 0),
				hovertemplate="Hour %{x}: %{y:+.0f} kW<extra>Transfer</extra>",
				opacity=0.7,
			),
			row=row, col=1,
		)

		fig.update_yaxes(title_text="Power (kW)", row=row, col=1)

	fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=900,
		title=f"Zone Power Dispatch - {scenario}",
		legend=dict(
			orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
			font=dict(size=11),
		),
		barmode="relative",
	)
	return fig


def plot_bus_voltage(step: int) -> go.Figure:
	"""Create grouped bar chart of bus voltages by zone."""
	records = BUS_VOLTAGES.get(str(step), BUS_VOLTAGES["0"])

	df = pd.DataFrame(records)

	fig = go.Figure()

	# Safety band
	fig.add_hrect(
		y0=0.95, y1=1.05,
		fillcolor=COLORS["safety_band"],
		line_width=0,
		annotation_text="Safe Band (0.95-1.05 pu)",
		annotation_position="top left",
		annotation_font=dict(color="rgba(16, 185, 129, 0.6)", size=10),
	)

	# Horizontal reference lines
	for y_val, label in [(0.95, "V_min"), (1.05, "V_max"), (1.0, "V_nom")]:
		fig.add_hline(
			y=y_val,
			line_dash="dot",
			line_color="rgba(255,255,255,0.2)",
			line_width=1,
		)

	# One trace per zone
	for z_idx in range(3):
		zone_df = df[df["zone_idx"] == z_idx].sort_values("bus")
		fig.add_trace(
			go.Bar(
				x=zone_df["bus"],
				y=zone_df["voltage"],
				name=ZONE_NAMES[z_idx],
				marker_color=ZONE_COLORS[z_idx],
				opacity=0.85,
				hovertemplate="Bus %{x}<br>V = %{y:.4f} pu<extra>" + ZONE_NAMES[z_idx] + "</extra>",
			)
		)

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=550,
		title=f"Bus Voltage Profile (Step {step}, Hour {step}:00)",
		xaxis_title="Bus Name (grouped by zone)",
		yaxis_title="Voltage Magnitude (pu)",
		yaxis=dict(range=[0.92, 1.08]),
		barmode="group",
		legend=dict(
			orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
		),
	)
	return fig


def plot_sankey(step: int) -> go.Figure:
	"""Create Sankey diagram for power flow at given step."""
	sd = SANKEY_DATA.get(str(step), SANKEY_DATA["0"])

	# Node indices:
	# 0=Substation, 1=PV_A, 2=PV_B, 3=PV_C, 4=Zone A, 5=Zone B, 6=Zone C,
	# 7=Load_A, 8=Load_B, 9=Load_C, 10=Losses
	node_labels = [
		"Substation",
		"PV_A", "PV_B", "PV_C",
		"Zone A", "Zone B", "Zone C",
		"Load A", "Load B", "Load C",
		"Losses",
	]
	node_colors = [
		"#818CF8",                              # Substation
		"#FCA5A5", "#FCA5A5", "#FCA5A5",        # PV sources
		COLORS["zone_a"], COLORS["zone_b"], COLORS["zone_c"],  # Zones
		"#93C5FD", "#93C5FD", "#93C5FD",        # Loads
		"#EF4444",                              # Losses
	]

	# Build links
	sources = []
	targets = []
	values = []
	link_colors = []

	# Substation -> Zones
	sub_to_a = sd["substation"] * 0.35
	sub_to_b = sd["substation"] * 0.45
	sub_to_c = sd["substation"] * 0.20

	for src, tgt, val, color in [
		(0, 4, sub_to_a, "rgba(129, 140, 248, 0.4)"),
		(0, 5, sub_to_b, "rgba(129, 140, 248, 0.4)"),
		(0, 6, sub_to_c, "rgba(129, 140, 248, 0.4)"),
		# PV -> Zones
		(1, 4, sd["pv_a"], "rgba(252, 165, 165, 0.4)"),
		(2, 5, sd["pv_b"], "rgba(252, 165, 165, 0.4)"),
		(3, 6, sd["pv_c"], "rgba(252, 165, 165, 0.4)"),
		# Zones -> Loads
		(4, 7, sd["load_a"], f"rgba({_hex_to_rgb(COLORS['zone_a'])}, 0.4)"),
		(5, 8, sd["load_b"], f"rgba({_hex_to_rgb(COLORS['zone_b'])}, 0.4)"),
		(6, 9, sd["load_c"], f"rgba({_hex_to_rgb(COLORS['zone_c'])}, 0.4)"),
		# Zones -> Losses (distributed proportionally)
		(4, 10, sd["losses"] * 0.30, "rgba(239, 68, 68, 0.3)"),
		(5, 10, sd["losses"] * 0.45, "rgba(239, 68, 68, 0.3)"),
		(6, 10, sd["losses"] * 0.25, "rgba(239, 68, 68, 0.3)"),
	]:
		if val > 0.5:  # Skip negligible flows
			sources.append(src)
			targets.append(tgt)
			values.append(float(val))
			link_colors.append(color)

	fig = go.Figure(data=[go.Sankey(
		node=dict(
			pad=20,
			thickness=25,
			line=dict(color="rgba(255,255,255,0.15)", width=1),
			label=node_labels,
			color=node_colors,
		),
		link=dict(
			source=sources,
			target=targets,
			value=values,
			color=link_colors,
		),
	)])

	total_load = sd["load_a"] + sd["load_b"] + sd["load_c"]
	total_pv = sd["pv_a"] + sd["pv_b"] + sd["pv_c"]

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=550,
		title=dict(
			text=(
				f"Power Flow Sankey (Step {step}, Hour {step}:00)"
				f"<br><sub>Substation: {sd['substation']:.0f} kW | "
				f"Total PV: {total_pv:.0f} kW | "
				f"Total Load: {total_load:.0f} kW | "
				f"Losses: {sd['losses']:.0f} kW</sub>"
			),
		),
	)
	return fig


def plot_training_rewards() -> go.Figure:
	"""Plot episode reward curve for 1500 episodes, 3 agents."""
	td = TRAINING_DATA
	episodes = td["episodes"]
	raw = td["total_reward"]
	smoothed = _smooth(raw, window=25)
	eps_smooth = episodes[:len(smoothed)]

	fig = go.Figure()

	# Raw reward (faint)
	fig.add_trace(go.Scatter(
		x=episodes, y=raw,
		mode="lines",
		name="Raw Reward",
		line=dict(color="rgba(99, 102, 241, 0.2)", width=1),
		hoverinfo="skip",
	))

	# Smoothed reward
	fig.add_trace(go.Scatter(
		x=eps_smooth, y=smoothed,
		mode="lines",
		name="Smoothed (25-ep)",
		line=dict(color=COLORS["primary"], width=2.5),
		hovertemplate="Episode %{x:.0f}<br>Reward: %{y:.2f}<extra></extra>",
	))

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=400,
		title="Total Team Reward (HAPPO on District_34Bus_3Zone)",
		xaxis_title="Episode",
		yaxis_title="Episode Reward",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_zone_rewards() -> go.Figure:
	"""Plot per-zone reward comparison (3 line plots)."""
	td = TRAINING_DATA
	episodes = td["episodes"]
	zone_rewards = td["zone_rewards"]

	fig = go.Figure()
	for z in range(3):
		smoothed = _smooth(zone_rewards[:, z], window=25)
		eps_smooth = episodes[:len(smoothed)]
		fig.add_trace(go.Scatter(
			x=eps_smooth, y=smoothed,
			mode="lines",
			name=ZONE_NAMES[z],
			line=dict(color=ZONE_COLORS[z], width=2),
			hovertemplate="Episode %{x:.0f}<br>Reward: %{y:.2f}<extra>" + ZONE_NAMES[z] + "</extra>",
		))

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=400,
		title="Per-Zone Reward Comparison",
		xaxis_title="Episode",
		yaxis_title="Zone Reward",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
	)
	return fig


def plot_voltage_violations() -> go.Figure:
	"""Plot voltage violation rate over training."""
	td = TRAINING_DATA
	episodes = td["episodes"]
	raw = td["voltage_violation_rate"]
	smoothed = _smooth(raw, window=30)
	eps_smooth = episodes[:len(smoothed)]

	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=episodes, y=raw * 100,
		mode="lines",
		name="Raw",
		line=dict(color="rgba(239, 68, 68, 0.15)", width=1),
		hoverinfo="skip",
	))

	fig.add_trace(go.Scatter(
		x=eps_smooth, y=smoothed * 100,
		mode="lines",
		name="Smoothed (30-ep)",
		line=dict(color="#EF4444", width=2.5),
		fill="tozeroy",
		fillcolor="rgba(239, 68, 68, 0.08)",
		hovertemplate="Episode %{x:.0f}<br>Violation: %{y:.1f}%<extra></extra>",
	))

	# Target line
	fig.add_hline(
		y=3, line_dash="dot", line_color="rgba(16, 185, 129, 0.5)",
		annotation_text="Target: 3%", annotation_position="bottom right",
		annotation_font=dict(color="#10B981", size=10),
	)

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=350,
		title="Voltage Violation Rate Over Training",
		xaxis_title="Episode",
		yaxis_title="Violation Rate (%)",
		yaxis=dict(range=[0, 25]),
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_power_balance() -> go.Figure:
	"""Plot inter-zone power balance metric over training."""
	td = TRAINING_DATA
	episodes = td["episodes"]
	raw = td["power_balance_metric"]
	smoothed = _smooth(raw, window=30)
	eps_smooth = episodes[:len(smoothed)]

	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=episodes, y=raw,
		mode="lines",
		name="Raw",
		line=dict(color="rgba(245, 158, 11, 0.15)", width=1),
		hoverinfo="skip",
	))

	fig.add_trace(go.Scatter(
		x=eps_smooth, y=smoothed,
		mode="lines",
		name="Smoothed (30-ep)",
		line=dict(color=COLORS["zone_c"], width=2.5),
		fill="tozeroy",
		fillcolor="rgba(245, 158, 11, 0.08)",
		hovertemplate="Episode %{x:.0f}<br>Imbalance: %{y:.3f}<extra></extra>",
	))

	fig.add_hline(
		y=0.05, line_dash="dot", line_color="rgba(16, 185, 129, 0.5)",
		annotation_text="Target: 0.05", annotation_position="bottom right",
		annotation_font=dict(color="#10B981", size=10),
	)

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=350,
		title="Inter-Zone Power Balance Metric",
		xaxis_title="Episode",
		yaxis_title="Imbalance Index",
		yaxis=dict(range=[0, 0.55]),
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


# ============================================================
# Utility
# ============================================================

def _hex_to_rgb(hex_color: str) -> str:
	"""Convert #RRGGBB to 'R, G, B' string for rgba()."""
	h = hex_color.lstrip("#")
	return f"{int(h[:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


# ============================================================
# Build Gradio App
# ============================================================

def build_app() -> gr.Blocks:
	"""Construct the Gradio Blocks application with 5 tabs."""
	with gr.Blocks(
		title="PowerZoo District Dispatch - Zone Coordination",
		theme=gr.themes.Soft(primary_hue="indigo"),
	) as app:

		# Header
		gr.Markdown(
			"""
			# PowerZoo: District Dispatch Environment
			**3-Zone Coordination** on IEEE 34-Bus System | **3 Dispatch Agents** | **24-Hour Episodes** | **Mixed Action Space**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# Tab 1: Overview
			# --------------------------------------------------------
			with gr.Tab("Overview"):
				gr.Markdown(
					"""
					## District Dispatch MARL Environment

					District Dispatch coordinates power distribution across **3 zones** in an IEEE 34-Bus system.
					Each zone has its own dispatch agent managing inter-zone power flow and local DER (Distributed Energy Resource) control.

					### Environment Architecture

					The IEEE 34-bus distribution feeder is partitioned into 3 operational zones:

					| Zone | Name | Buses | Boundary Bus | DER Assets |
					|------|------|-------|--------------|------------|
					| **Zone A** (Upstream) | Primary feeder head | sourcebus - 830 (16 buses) | Bus 830 | 2 PV (200kW each), 1 ESS (500kWh), 1 EV charger |
					| **Zone B** (Midstream) | Feeder midspan | 832 - 864 (17 buses) | Bus 854, 848 | 2 PV (200kW each), 1 ESS (500kWh), 1 EV charger |
					| **Zone C** (Downstream) | Remote branch | 888 - 890 (2 buses) | Bus 890 | 1 PV (150kW), 1 ESS (300kWh) |

					### Inter-Zone Tie Lines

					| Tie Line | From | To | Capacity | Type |
					|----------|------|----|----------|------|
					| `tie_0_1` | Zone A (Bus 830) | Zone B (Bus 854) | 500 kW | Transformer |
					| `tie_1_2` | Zone B (Bus 848) | Zone C (Bus 890) | 300 kW | Tieline |

					### Key Specifications

					| Parameter | Value |
					|-----------|-------|
					| Agents | 3 (one per zone) |
					| Episode Length | 24 steps (hourly) |
					| Action Space | Continuous [-1, 1]: inter-zone P/Q exchange, PV curtailment, storage control, EV modulation |
					| Observation | Local: voltage stats, load, PV/ESS/EV state, exchange, neighbor info, time/price |
					| Shared Observation | All local obs concatenated + system-level aggregates + market state |
					| Reward | Weighted: economic dispatch, voltage compliance, loss minimization, carbon, exchange balance, storage health |
					| Voltage Limits | 0.95 - 1.05 pu |
					| Supported Systems | IEEE 34-Bus 3-Zone |

					### MARL Formulation

					Each zone agent observes its local DER state, bus voltages, and neighbor zone summaries.
					The centralized critic sees the full system state. Agents coordinate through:

					1. **Tie-line power exchange** -- active and reactive power transfer decisions
					2. **PV curtailment** -- reduce solar output to prevent overvoltage
					3. **Storage dispatch** -- charge/discharge scheduling for peak shaving
					4. **EV load modulation** -- flexible EV charging rate adjustment

					The multi-objective reward balances economic efficiency, voltage quality, network losses,
					carbon emissions, exchange equilibrium, and battery health across all zones.
					"""
				)

			# --------------------------------------------------------
			# Tab 2: Zone Power Dispatch
			# --------------------------------------------------------
			with gr.Tab("Zone Power Dispatch"):
				gr.Markdown("## Zone-Level Load, Generation & Power Transfer")

				scenario_slider = gr.Radio(
					choices=list(SCENARIO_LABELS.values()),
					value="Summer Peak",
					label="Scenario",
				)

				dispatch_plot = gr.Plot(value=plot_zone_power_dispatch("Summer Peak"))

				scenario_slider.change(
					fn=plot_zone_power_dispatch,
					inputs=scenario_slider,
					outputs=dispatch_plot,
				)

				gr.Markdown(
					"""
					**Reading the chart**: Each row is one zone. The shaded area shows load demand,
					the dashed pink line shows PV generation, and bars show inter-zone power transfer
					(green = export, red = import). Midday PV surplus in Zone A/B feeds Zone C's deficit.
					"""
				)

			# --------------------------------------------------------
			# Tab 3: Bus Voltage by Zone
			# --------------------------------------------------------
			with gr.Tab("Bus Voltage by Zone"):
				gr.Markdown("## Per-Bus Voltage Magnitude Grouped by Zone")

				step_dropdown = gr.Dropdown(
					choices=[str(i) for i in range(24)],
					value="12",
					label="Step (Hour)",
				)

				voltage_plot = gr.Plot(value=plot_bus_voltage(12))

				step_dropdown.change(
					fn=lambda s: plot_bus_voltage(int(s)),
					inputs=step_dropdown,
					outputs=voltage_plot,
				)

				gr.Markdown(
					"""
					**Voltage safety band**: The green-shaded region marks the ANSI C84.1 acceptable
					range (0.95-1.05 pu). Downstream zones (Zone C) experience larger voltage drops,
					while midday PV injection provides a slight voltage boost in Zones A and B.
					"""
				)

			# --------------------------------------------------------
			# Tab 4: Power Flow Sankey
			# --------------------------------------------------------
			with gr.Tab("Power Flow Sankey"):
				gr.Markdown("## System Power Flow Diagram")

				sankey_slider = gr.Slider(
					minimum=0, maximum=23, step=1, value=12,
					label="Step (Hour)",
				)

				sankey_plot = gr.Plot(value=plot_sankey(12))

				sankey_slider.change(
					fn=lambda s: plot_sankey(int(s)),
					inputs=sankey_slider,
					outputs=sankey_plot,
				)

				gr.Markdown(
					"""
					**Flow interpretation**: Link thickness is proportional to power flow (kW).
					Substation provides grid power; PV sources inject into their respective zones.
					Each zone serves its local load with residual flowing to network losses.
					Slide through hours to see how PV generation shifts the Substation dependency
					-- at noon, PV covers a large share of zone loads.
					"""
				)

			# --------------------------------------------------------
			# Tab 5: Training Dashboard
			# --------------------------------------------------------
			with gr.Tab("Training Dashboard"):
				gr.Markdown(
					"""
					## HAPPO Training on District_34Bus_3Zone
					Synthetic training curves for 1500 episodes showing multi-zone coordination
					improvement. The agents learn to balance inter-zone power exchange while
					maintaining voltage compliance and minimizing losses.
					"""
				)

				reward_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### Per-Zone Reward Comparison")
				zone_reward_plot = gr.Plot(value=plot_zone_rewards())

				with gr.Row():
					with gr.Column():
						gr.Markdown("### Voltage Violation Rate")
						vv_plot = gr.Plot(value=plot_voltage_violations())
					with gr.Column():
						gr.Markdown("### Inter-Zone Power Balance")
						pb_plot = gr.Plot(value=plot_power_balance())

				gr.Markdown(
					"""
					**Training dynamics**: Zone A (upstream, close to substation) converges fastest
					due to stronger voltage support. Zone C (downstream, remote branch) takes longest
					to learn effective dispatch. The power balance metric shows agents progressively
					learning cooperative exchange strategies that reduce inter-zone imbalance.
					"""
				)

		# Footer
		gr.Markdown(
			"""
			---
			**PowerZoo** · MIT License · [XJTU-RL](https://github.com/XJTU-RL) · IEEE TSG 2025
			"""
		)

	return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(server_name="0.0.0.0", server_port=7860, share=False)
