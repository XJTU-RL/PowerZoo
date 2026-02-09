"""
PowerZoo 分区调度：多区域协调演示
HuggingFace Spaces 应用，基于 Gradio + Plotly。

分区调度 MARL 环境的独立演示应用。
3 个分区智能体在 IEEE 34 节点系统上协调功率调度。

5 个标签页：概览 | 分区功率调度 | 各区母线电压 | 功率流桑基图 | 训练仪表盘
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

ZONE_NAMES = ["A\u533a (\u4e0a\u6e38)", "B\u533a (\u4e2d\u6e38)", "C\u533a (\u4e0b\u6e38)"]
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
	"summer_peak": "\u590f\u5b63\u9ad8\u5cf0",
	"winter": "\u51ac\u5b63",
	"mild_day": "\u6e29\u548c\u5929\u6c14",
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
				name=f"\u8d1f\u8377\u9700\u6c42 ({ZONE_NAMES[z].split(' ')[0]})",
				fill="tozeroy",
				fillcolor=f"rgba({_hex_to_rgb(ZONE_COLORS[z])}, 0.15)",
				line=dict(color=ZONE_COLORS[z], width=2),
				legendgroup=f"zone_{z}",
				showlegend=(z == 0),
				hovertemplate="\u65f6\u523b %{x}: %{y:.0f} kW<extra>\u8d1f\u8377\u9700\u6c42</extra>",
			),
			row=row, col=1,
		)

		# Generation (dashed line)
		fig.add_trace(
			go.Scatter(
				x=HOURS, y=zd["generation_kw"],
				mode="lines+markers",
				name="\u53d1\u7535\u51fa\u529b" if z == 0 else None,
				line=dict(color="#F472B6", width=2, dash="dash"),
				marker=dict(size=4, color="#F472B6"),
				showlegend=(z == 0),
				hovertemplate="\u65f6\u523b %{x}: %{y:.0f} kW<extra>\u53d1\u7535\u51fa\u529b</extra>",
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
				name="\u533a\u95f4\u529f\u7387\u4ea4\u6362" if z == 0 else None,
				marker_color=bar_colors,
				showlegend=(z == 0),
				hovertemplate="\u65f6\u523b %{x}: %{y:+.0f} kW<extra>\u533a\u95f4\u529f\u7387\u4ea4\u6362</extra>",
				opacity=0.7,
			),
			row=row, col=1,
		)

		fig.update_yaxes(title_text="\u529f\u7387 (kW)", row=row, col=1)

	fig.update_xaxes(title_text="\u5c0f\u65f6", row=3, col=1)
	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=900,
		title=f"\u5206\u533a\u529f\u7387\u8c03\u5ea6 - {scenario}",
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
		annotation_text="\u5b89\u5168\u8303\u56f4 (0.95-1.05 pu)",
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
				hovertemplate="\u6bcd\u7ebf %{x}<br>V = %{y:.4f} pu<extra>" + ZONE_NAMES[z_idx] + "</extra>",
			)
		)

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=550,
		title=f"\u6bcd\u7ebf\u7535\u538b\u5206\u5e03 (\u65f6\u95f4\u6b65 {step}, {step}:00)",
		xaxis_title="\u6bcd\u7ebf\u540d\u79f0 (\u6309\u5206\u533a\u5206\u7ec4)",
		yaxis_title="\u7535\u538b (\u6807\u4e46\u503c)",
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
		"\u53d8\u7535\u7ad9",
		"PV_A", "PV_B", "PV_C",
		"A\u533a", "B\u533a", "C\u533a",
		"\u8d1f\u8377_A", "\u8d1f\u8377_B", "\u8d1f\u8377_C",
		"\u635f\u8017",
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
				f"\u529f\u7387\u6d41\u6851\u57fa\u56fe (\u65f6\u95f4\u6b65 {step}, {step}:00)"
				f"<br><sub>\u53d8\u7535\u7ad9: {sd['substation']:.0f} kW | "
				f"\u5149\u4f0f\u603b\u51fa\u529b: {total_pv:.0f} kW | "
				f"\u603b\u8d1f\u8377: {total_load:.0f} kW | "
				f"\u635f\u8017: {sd['losses']:.0f} kW</sub>"
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
		name="\u539f\u59cb\u5956\u52b1",
		line=dict(color="rgba(99, 102, 241, 0.2)", width=1),
		hoverinfo="skip",
	))

	# Smoothed reward
	fig.add_trace(go.Scatter(
		x=eps_smooth, y=smoothed,
		mode="lines",
		name="\u5e73\u6ed1 (25\u56de\u5408)",
		line=dict(color=COLORS["primary"], width=2.5),
		hovertemplate="\u56de\u5408 %{x:.0f}<br>\u5956\u52b1: %{y:.2f}<extra></extra>",
	))

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=400,
		title="\u56e2\u961f\u603b\u5956\u52b1 (HAPPO \u5728 District_34Bus_3Zone \u4e0a\u8bad\u7ec3)",
		xaxis_title="\u56de\u5408",
		yaxis_title="\u56de\u5408\u5956\u52b1",
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
			hovertemplate="\u56de\u5408 %{x:.0f}<br>\u5956\u52b1: %{y:.2f}<extra>" + ZONE_NAMES[z] + "</extra>",
		))

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=400,
		title="\u5404\u533a\u5956\u52b1\u5bf9\u6bd4",
		xaxis_title="\u56de\u5408",
		yaxis_title="\u5206\u533a\u5956\u52b1",
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
		name="\u539f\u59cb",
		line=dict(color="rgba(239, 68, 68, 0.15)", width=1),
		hoverinfo="skip",
	))

	fig.add_trace(go.Scatter(
		x=eps_smooth, y=smoothed * 100,
		mode="lines",
		name="\u5e73\u6ed1 (30\u56de\u5408)",
		line=dict(color="#EF4444", width=2.5),
		fill="tozeroy",
		fillcolor="rgba(239, 68, 68, 0.08)",
		hovertemplate="\u56de\u5408 %{x:.0f}<br>\u8d8a\u9650\u7387: %{y:.1f}%<extra></extra>",
	))

	# Target line
	fig.add_hline(
		y=3, line_dash="dot", line_color="rgba(16, 185, 129, 0.5)",
		annotation_text="\u76ee\u6807: 3%", annotation_position="bottom right",
		annotation_font=dict(color="#10B981", size=10),
	)

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=350,
		title="\u8bad\u7ec3\u8fc7\u7a0b\u7535\u538b\u8d8a\u9650\u7387",
		xaxis_title="\u56de\u5408",
		yaxis_title="\u8d8a\u9650\u7387 (%)",
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
		name="\u539f\u59cb",
		line=dict(color="rgba(245, 158, 11, 0.15)", width=1),
		hoverinfo="skip",
	))

	fig.add_trace(go.Scatter(
		x=eps_smooth, y=smoothed,
		mode="lines",
		name="\u5e73\u6ed1 (30\u56de\u5408)",
		line=dict(color=COLORS["zone_c"], width=2.5),
		fill="tozeroy",
		fillcolor="rgba(245, 158, 11, 0.08)",
		hovertemplate="\u56de\u5408 %{x:.0f}<br>\u4e0d\u5e73\u8861\u6307\u6570: %{y:.3f}<extra></extra>",
	))

	fig.add_hline(
		y=0.05, line_dash="dot", line_color="rgba(16, 185, 129, 0.5)",
		annotation_text="\u76ee\u6807: 0.05", annotation_position="bottom right",
		annotation_font=dict(color="#10B981", size=10),
	)

	fig.update_layout(
		**LAYOUT_DEFAULTS,
		height=350,
		title="\u533a\u95f4\u529f\u7387\u5e73\u8861\u6307\u6807",
		xaxis_title="\u56de\u5408",
		yaxis_title="\u4e0d\u5e73\u8861\u6307\u6570",
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
		title="PowerZoo \u5206\u533a\u8c03\u5ea6 \u591a\u533a\u57df\u534f\u8c03",
		theme=gr.themes.Soft(primary_hue="indigo"),
	) as app:

		# Header
		gr.Markdown(
			"""
			# PowerZoo \u5206\u533a\u8c03\u5ea6 \u591a\u533a\u57df\u534f\u8c03
			**3 \u533a\u57df\u534f\u8c03** \u57fa\u4e8e IEEE 34 \u8282\u70b9\u7cfb\u7edf | **3 \u4e2a\u8c03\u5ea6\u667a\u80fd\u4f53** | **24 \u5c0f\u65f6\u56de\u5408** | **\u6df7\u5408\u52a8\u4f5c\u7a7a\u95f4**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# Tab 1: Overview
			# --------------------------------------------------------
			with gr.Tab("\u6982\u89c8"):
				gr.Markdown(
					"""
					## \u5206\u533a\u8c03\u5ea6 MARL \u73af\u5883

					\u5206\u533a\u8c03\u5ea6\u73af\u5883\u5728 IEEE 34 \u8282\u70b9\u7cfb\u7edf\u4e0a\u534f\u8c03 **3 \u4e2a\u5206\u533a** \u7684\u7535\u529b\u5206\u914d\u3002
					\u6bcf\u4e2a\u5206\u533a\u62e5\u6709\u72ec\u7acb\u7684\u8c03\u5ea6\u667a\u80fd\u4f53\uff0c\u7ba1\u7406\u533a\u95f4\u529f\u7387\u6d41\u548c\u672c\u5730\u5206\u5e03\u5f0f\u80fd\u6e90 (DER) \u63a7\u5236\u3002

					### \u73af\u5883\u67b6\u6784

					IEEE 34 \u8282\u70b9\u914d\u7535\u9988\u7ebf\u88ab\u5212\u5206\u4e3a 3 \u4e2a\u8fd0\u884c\u533a\u57df\uff1a

					| \u533a\u57df | \u540d\u79f0 | \u6bcd\u7ebf | \u8fb9\u754c\u6bcd\u7ebf | DER \u8d44\u4ea7 |
					|------|------|-------|--------------|------------|
					| **A\u533a** (\u4e0a\u6e38) | \u9988\u7ebf\u9996\u7aef | sourcebus - 830 (16 \u4e2a\u6bcd\u7ebf) | Bus 830 | 2 \u5957\u5149\u4f0f (200kW/\u5957), 1 \u5957\u50a8\u80fd (500kWh), 1 \u4e2a\u5145\u7535\u6869 |
					| **B\u533a** (\u4e2d\u6e38) | \u9988\u7ebf\u4e2d\u6bb5 | 832 - 864 (17 \u4e2a\u6bcd\u7ebf) | Bus 854, 848 | 2 \u5957\u5149\u4f0f (200kW/\u5957), 1 \u5957\u50a8\u80fd (500kWh), 1 \u4e2a\u5145\u7535\u6869 |
					| **C\u533a** (\u4e0b\u6e38) | \u8fdc\u7aef\u652f\u8def | 888 - 890 (2 \u4e2a\u6bcd\u7ebf) | Bus 890 | 1 \u5957\u5149\u4f0f (150kW), 1 \u5957\u50a8\u80fd (300kWh) |

					### \u533a\u95f4\u8054\u7edc\u7ebf

					| \u8054\u7edc\u7ebf | \u8d77\u59cb | \u7ec8\u6b62 | \u5bb9\u91cf | \u7c7b\u578b |
					|----------|------|----|----------|------|
					| `tie_0_1` | A\u533a (Bus 830) | B\u533a (Bus 854) | 500 kW | \u53d8\u538b\u5668 |
					| `tie_1_2` | B\u533a (Bus 848) | C\u533a (Bus 890) | 300 kW | \u8054\u7edc\u7ebf |

					### \u5173\u952e\u53c2\u6570

					| \u53c2\u6570 | \u503c |
					|-----------|-------|
					| \u667a\u80fd\u4f53\u6570\u91cf | 3 (\u6bcf\u533a\u4e00\u4e2a) |
					| \u56de\u5408\u957f\u5ea6 | 24 \u6b65 (\u6bcf\u5c0f\u65f6\u4e00\u6b65) |
					| \u52a8\u4f5c\u7a7a\u95f4 | \u8fde\u7eed [-1, 1]: \u533a\u95f4\u6709\u529f/\u65e0\u529f\u4ea4\u6362\u3001\u5149\u4f0f\u5f03\u5149\u3001\u50a8\u80fd\u63a7\u5236\u3001\u5145\u7535\u8c03\u8282 |
					| \u89c2\u6d4b\u7a7a\u95f4 | \u5c40\u90e8: \u7535\u538b\u7edf\u8ba1\u3001\u8d1f\u8377\u3001\u5149\u4f0f/\u50a8\u80fd/\u5145\u7535\u6869\u72b6\u6001\u3001\u4ea4\u6362\u529f\u7387\u3001\u76f8\u90bb\u533a\u4fe1\u606f\u3001\u65f6\u95f4/\u7535\u4ef7 |
					| \u5171\u4eab\u89c2\u6d4b | \u6240\u6709\u5c40\u90e8\u89c2\u6d4b\u62fc\u63a5 + \u7cfb\u7edf\u7ea7\u6c47\u603b + \u5e02\u573a\u72b6\u6001 |
					| \u5956\u52b1\u51fd\u6570 | \u52a0\u6743: \u7ecf\u6d4e\u8c03\u5ea6\u3001\u7535\u538b\u5408\u89c4\u3001\u635f\u8017\u6700\u5c0f\u5316\u3001\u78b3\u6392\u653e\u3001\u4ea4\u6362\u5e73\u8861\u3001\u50a8\u80fd\u5065\u5eb7 |
					| \u7535\u538b\u9650\u503c | 0.95 - 1.05 pu |
					| \u652f\u6301\u7cfb\u7edf | IEEE 34 \u8282\u70b9 3 \u5206\u533a |

					### MARL \u5efa\u6a21

					\u6bcf\u4e2a\u5206\u533a\u667a\u80fd\u4f53\u89c2\u6d4b\u672c\u5730 DER \u72b6\u6001\u3001\u6bcd\u7ebf\u7535\u538b\u548c\u76f8\u90bb\u533a\u57df\u6458\u8981\u4fe1\u606f\u3002
					\u96c6\u4e2d\u5f0f\u8bc4\u4ef7\u5668 (Critic) \u53ef\u89c1\u5168\u7cfb\u7edf\u72b6\u6001\u3002\u667a\u80fd\u4f53\u901a\u8fc7\u4ee5\u4e0b\u65b9\u5f0f\u534f\u8c03\uff1a

					1. **\u8054\u7edc\u7ebf\u529f\u7387\u4ea4\u6362** -- \u6709\u529f\u548c\u65e0\u529f\u529f\u7387\u4f20\u8f93\u51b3\u7b56
					2. **\u5149\u4f0f\u5f03\u5149** -- \u524a\u51cf\u592a\u9633\u80fd\u51fa\u529b\u4ee5\u9632\u6b62\u8fc7\u7535\u538b
					3. **\u50a8\u80fd\u8c03\u5ea6** -- \u5145\u653e\u7535\u8ba1\u5212\u7528\u4e8e\u524a\u5cf0\u586b\u8c37
					4. **\u5145\u7535\u8d1f\u8377\u8c03\u8282** -- \u7075\u6d3b\u8c03\u6574\u7535\u52a8\u6c7d\u8f66\u5145\u7535\u529f\u7387

					\u591a\u76ee\u6807\u5956\u52b1\u51fd\u6570\u5728\u6240\u6709\u533a\u57df\u95f4\u5e73\u8861\u7ecf\u6d4e\u6548\u7387\u3001\u7535\u538b\u8d28\u91cf\u3001\u7f51\u7edc\u635f\u8017\u3001
					\u78b3\u6392\u653e\u3001\u4ea4\u6362\u5747\u8861\u548c\u7535\u6c60\u5065\u5eb7\u3002
					"""
				)

			# --------------------------------------------------------
			# Tab 2: Zone Power Dispatch
			# --------------------------------------------------------
			with gr.Tab("\u5206\u533a\u529f\u7387\u8c03\u5ea6"):
				gr.Markdown("## \u5206\u533a\u8d1f\u8377\u3001\u53d1\u7535\u4e0e\u529f\u7387\u4ea4\u6362")

				scenario_slider = gr.Radio(
					choices=list(SCENARIO_LABELS.values()),
					value="\u590f\u5b63\u9ad8\u5cf0",
					label="\u573a\u666f",
				)

				dispatch_plot = gr.Plot(value=plot_zone_power_dispatch("\u590f\u5b63\u9ad8\u5cf0"))

				scenario_slider.change(
					fn=plot_zone_power_dispatch,
					inputs=scenario_slider,
					outputs=dispatch_plot,
				)

				gr.Markdown(
					"""
					**\u56fe\u8868\u8bf4\u660e**: \u6bcf\u884c\u5bf9\u5e94\u4e00\u4e2a\u5206\u533a\u3002\u586b\u5145\u533a\u57df\u663e\u793a\u8d1f\u8377\u9700\u6c42\uff0c
					\u7c89\u8272\u865a\u7ebf\u663e\u793a\u5149\u4f0f\u53d1\u7535\u51fa\u529b\uff0c\u67f1\u72b6\u56fe\u663e\u793a\u533a\u95f4\u529f\u7387\u4ea4\u6362
					(\u7eff\u8272 = \u9001\u51fa\uff0c\u7ea2\u8272 = \u53d7\u5165)\u3002\u5348\u95f4 A\u533a/B\u533a \u7684\u5149\u4f0f\u53d1\u7535\u76c8\u4f59\u8865\u7ed9 C\u533a \u7684\u7f3a\u989d\u3002
					"""
				)

			# --------------------------------------------------------
			# Tab 3: Bus Voltage by Zone
			# --------------------------------------------------------
			with gr.Tab("\u5404\u533a\u6bcd\u7ebf\u7535\u538b"):
				gr.Markdown("## \u5404\u6bcd\u7ebf\u7535\u538b\u5e45\u503c (\u6309\u5206\u533a\u5206\u7ec4)")

				step_dropdown = gr.Dropdown(
					choices=[str(i) for i in range(24)],
					value="12",
					label="\u65f6\u95f4\u6b65 (\u5c0f\u65f6)",
				)

				voltage_plot = gr.Plot(value=plot_bus_voltage(12))

				step_dropdown.change(
					fn=lambda s: plot_bus_voltage(int(s)),
					inputs=step_dropdown,
					outputs=voltage_plot,
				)

				gr.Markdown(
					"""
					**\u7535\u538b\u5b89\u5168\u8303\u56f4**: \u7eff\u8272\u9634\u5f71\u533a\u57df\u6807\u8bb0\u7b26\u5408 ANSI C84.1 \u6807\u51c6\u7684\u5b89\u5168\u8303\u56f4
					(0.95-1.05 pu)\u3002\u4e0b\u6e38\u5206\u533a (C\u533a) \u7535\u538b\u964d\u843d\u66f4\u5927\uff0c
					\u800c\u5348\u95f4\u5149\u4f0f\u6ce8\u5165\u4e3a A\u533a \u548c B\u533a \u63d0\u4f9b\u4e86\u8f7b\u5fae\u7684\u7535\u538b\u62ac\u5347\u3002
					"""
				)

			# --------------------------------------------------------
			# Tab 4: Power Flow Sankey
			# --------------------------------------------------------
			with gr.Tab("\u529f\u7387\u6d41\u6851\u57fa\u56fe"):
				gr.Markdown("## \u7cfb\u7edf\u529f\u7387\u6d41\u5411\u56fe")

				sankey_slider = gr.Slider(
					minimum=0, maximum=23, step=1, value=12,
					label="\u65f6\u95f4\u6b65 (\u5c0f\u65f6)",
				)

				sankey_plot = gr.Plot(value=plot_sankey(12))

				sankey_slider.change(
					fn=lambda s: plot_sankey(int(s)),
					inputs=sankey_slider,
					outputs=sankey_plot,
				)

				gr.Markdown(
					"""
					**\u529f\u7387\u6d41\u89e3\u8bfb**: \u8fde\u63a5\u7ebf\u5bbd\u5ea6\u4e0e\u529f\u7387\u6d41 (kW) \u6210\u6b63\u6bd4\u3002
					\u53d8\u7535\u7ad9\u63d0\u4f9b\u7535\u7f51\u529f\u7387\uff1b\u5149\u4f0f\u7535\u6e90\u6ce8\u5165\u5404\u81ea\u5206\u533a\u3002
					\u6bcf\u4e2a\u5206\u533a\u670d\u52a1\u672c\u5730\u8d1f\u8377\uff0c\u5269\u4f59\u90e8\u5206\u4ee5\u7f51\u7edc\u635f\u8017\u5f62\u5f0f\u6d88\u8017\u3002
					\u62d6\u52a8\u6ed1\u5757\u67e5\u770b\u4e0d\u540c\u65f6\u523b\u7684\u53d8\u5316\u2014\u2014\u5348\u95f4\u65f6\u5206\uff0c\u5149\u4f0f\u53d1\u7535\u8986\u76d6\u4e86\u5206\u533a\u8d1f\u8377\u7684\u5927\u90e8\u5206\uff0c
					\u663e\u8457\u51cf\u5c11\u4e86\u5bf9\u53d8\u7535\u7ad9\u7684\u4f9d\u8d56\u3002
					"""
				)

			# --------------------------------------------------------
			# Tab 5: Training Dashboard
			# --------------------------------------------------------
			with gr.Tab("\u8bad\u7ec3\u4eea\u8868\u76d8"):
				gr.Markdown(
					"""
					## HAPPO \u5728 District_34Bus_3Zone \u4e0a\u7684\u8bad\u7ec3
					1500 \u56de\u5408\u7684\u5408\u6210\u8bad\u7ec3\u66f2\u7ebf\uff0c\u5c55\u793a\u591a\u533a\u57df\u534f\u8c03\u80fd\u529b\u7684\u63d0\u5347\u3002
					\u667a\u80fd\u4f53\u9010\u6b65\u5b66\u4f1a\u5e73\u8861\u533a\u95f4\u529f\u7387\u4ea4\u6362\uff0c\u540c\u65f6\u4fdd\u6301\u7535\u538b\u5408\u89c4\u5e76\u6700\u5c0f\u5316\u635f\u8017\u3002
					"""
				)

				reward_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### \u5404\u533a\u5956\u52b1\u5bf9\u6bd4")
				zone_reward_plot = gr.Plot(value=plot_zone_rewards())

				with gr.Row():
					with gr.Column():
						gr.Markdown("### \u7535\u538b\u8d8a\u9650\u7387")
						vv_plot = gr.Plot(value=plot_voltage_violations())
					with gr.Column():
						gr.Markdown("### \u533a\u95f4\u529f\u7387\u5e73\u8861")
						pb_plot = gr.Plot(value=plot_power_balance())

				gr.Markdown(
					"""
					**\u8bad\u7ec3\u52a8\u6001**: A\u533a (\u4e0a\u6e38\uff0c\u9760\u8fd1\u53d8\u7535\u7ad9) \u7531\u4e8e\u7535\u538b\u652f\u6491\u8f83\u5f3a\uff0c\u6536\u655b\u6700\u5feb\u3002
					C\u533a (\u4e0b\u6e38\uff0c\u8fdc\u7aef\u652f\u8def) \u5b66\u4e60\u6709\u6548\u8c03\u5ea6\u7684\u65f6\u95f4\u6700\u957f\u3002\u529f\u7387\u5e73\u8861\u6307\u6807\u663e\u793a\u667a\u80fd\u4f53
					\u9010\u6b65\u5b66\u4f1a\u534f\u4f5c\u4ea4\u6362\u7b56\u7565\uff0c\u6709\u6548\u964d\u4f4e\u4e86\u533a\u95f4\u4e0d\u5e73\u8861\u3002
					"""
				)

		# Footer
		gr.Markdown(
			"""
			---
			**PowerZoo** \u00b7 MIT \u8bb8\u53ef\u8bc1 \u00b7 [XJTU-RL](https://github.com/XJTU-RL) \u00b7 IEEE TSG 2025
			"""
		)

	return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(server_name="0.0.0.0", server_port=7860, share=False)
