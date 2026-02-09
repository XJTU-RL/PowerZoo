"""
PowerZoo SmartGrid: Interactive CMDP Environment Demo
HuggingFace Spaces application with Gradio + Plotly.

5 Tabs: Overview | Voltage Heatmap | Lagrangian Trajectory | Component Status | Training Dashboard

SmartGrid is a modular PV integration environment using CMDP (Constrained MDP)
with Lagrangian relaxation. 360-step annual episodes (1 step = 1 day),
homogeneous agents controlling capacitors, regulators, batteries, and PV.
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
	"primary": "#10B981",      # emerald
	"secondary": "#059669",    # emerald dark
	"accent": "#06B6D4",       # cyan
	"warning": "#F59E0B",      # amber
	"danger": "#EF4444",       # red
	"bg_card": "rgba(16, 185, 129, 0.05)",
	"grid": "rgba(255, 255, 255, 0.08)",
	"text": "#E2E8F0",
	"text_dim": "#94A3B8",
}

PLOTLY_LAYOUT_DEFAULTS = dict(
	template="plotly_dark",
	paper_bgcolor="rgba(0,0,0,0)",
	plot_bgcolor="rgba(0,0,0,0)",
	font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
	margin=dict(l=60, r=30, t=50, b=50),
	xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
	yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)

# Deterministic RNG for reproducible demo data
RNG = np.random.default_rng(seed=42)


# ============================================================
# Demo Data Generators
# ============================================================

BUS_NAMES_13 = [
	"650", "632", "633", "634", "645", "646",
	"671", "680", "684", "611", "652", "692", "675",
]


def generate_voltage_heatmap_data() -> np.ndarray:
	"""Generate 360x13 voltage matrix with seasonal PV patterns.

	Summer months show higher voltage from PV injection;
	winter shows lower voltage from increased load.

	Returns:
		np.ndarray: shape (360, 13), voltage in per-unit.
	"""
	days = np.arange(360)
	n_buses = len(BUS_NAMES_13)

	# Base voltage profile: sinusoidal annual pattern
	# Summer peak (day ~180) pushes voltage up from PV generation
	seasonal = 0.025 * np.sin(2 * np.pi * (days - 90) / 360)

	# Per-bus baseline offset (some buses naturally higher/lower)
	bus_offset = RNG.uniform(-0.015, 0.015, size=n_buses)

	# Build matrix
	voltage = np.ones((360, n_buses))
	for b in range(n_buses):
		voltage[:, b] += seasonal + bus_offset[b]

	# Add daily noise
	noise = RNG.normal(0, 0.008, size=(360, n_buses))
	voltage += noise

	# Inject realistic violations:
	# Summer PV overvoltage on downstream buses (indices 6-12)
	for b in range(6, n_buses):
		summer_mask = (days >= 120) & (days <= 240)
		voltage[summer_mask, b] += RNG.uniform(0.02, 0.045, size=summer_mask.sum())

	# Winter undervoltage on load-heavy buses (indices 3, 10, 12)
	for b in [3, 10, 12]:
		winter_mask = (days <= 60) | (days >= 300)
		voltage[winter_mask, b] -= RNG.uniform(0.01, 0.035, size=winter_mask.sum())

	return np.clip(voltage, 0.90, 1.12)


def generate_lagrangian_data(n_episodes: int = 500) -> dict[str, np.ndarray]:
	"""Generate CMDP convergence curves for Lagrangian multiplier and constraint violation.

	Lambda starts high (~10) and converges to ~2.
	Violation starts at ~15% and drops below 2%.

	Returns:
		dict with keys: episodes, lambda_values, violation_rate
	"""
	episodes = np.arange(n_episodes)

	# Lambda convergence: exponential decay + noise
	lambda_base = 2.0 + 8.0 * np.exp(-episodes / 80)
	lambda_noise = RNG.normal(0, 0.3, size=n_episodes) * np.exp(-episodes / 200)
	lambda_values = np.clip(lambda_base + lambda_noise, 0.5, 12.0)

	# Constraint violation rate: sigmoid-like decrease
	violation_base = 0.15 / (1 + np.exp((episodes - 100) / 40))
	violation_noise = RNG.uniform(-0.005, 0.005, size=n_episodes)
	violation_rate = np.clip(violation_base + violation_noise + 0.012, 0.0, 0.20)
	# Final episodes settle below 2%
	violation_rate[400:] = np.clip(violation_rate[400:] * 0.6, 0.005, 0.02)

	return {
		"episodes": episodes,
		"lambda_values": lambda_values,
		"violation_rate": violation_rate * 100,  # percentage
	}


def generate_component_data(day_of_year: int) -> dict[str, np.ndarray]:
	"""Generate 24-hour component operation data for a given day.

	Seasonal variation affects PV output and battery cycling.

	Args:
		day_of_year: 0-indexed day (0=Jan1, 90=Apr1, 181=Jul1, 272=Oct1).

	Returns:
		dict with keys: hours, cap_kvar, reg_tap, battery_soc, pv_kw, pv_curtail
	"""
	hours = np.arange(24)

	# Season factor (0=winter, 1=summer peak)
	season_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 90) / 360)

	# Capacitor reactive power: switched based on load/voltage
	# Higher in afternoon, lower at night
	load_pattern = np.array([
		0.3, 0.25, 0.2, 0.2, 0.25, 0.35, 0.5, 0.65,
		0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9,
		0.85, 0.9, 0.95, 0.85, 0.7, 0.55, 0.45, 0.35,
	])
	cap_kvar = load_pattern * 300 + RNG.normal(0, 15, size=24)
	cap_kvar = np.clip(cap_kvar, 0, 400)

	# Regulator tap position: -16 to +16, tracks voltage deviation
	reg_base = np.array([
		2, 2, 3, 3, 2, 1, 0, -1,
		-2, -3, -4, -5, -6, -7, -6, -5,
		-4, -3, -2, -1, 0, 1, 2, 2,
	])
	# PV pushes tap down in summer
	reg_tap = reg_base - int(season_factor * 4) + RNG.integers(-1, 2, size=24)
	reg_tap = np.clip(reg_tap, -16, 16)

	# Battery SOC: charges from PV midday, discharges evening peak
	soc = np.zeros(24)
	soc[0] = 50 + season_factor * 10  # initial SOC
	for h in range(1, 24):
		if 9 <= h <= 15:  # charging from PV
			soc[h] = soc[h - 1] + (3.0 + season_factor * 2.0) + RNG.normal(0, 0.5)
		elif 17 <= h <= 21:  # evening discharge
			soc[h] = soc[h - 1] - (4.0 + season_factor * 1.5) + RNG.normal(0, 0.5)
		else:
			soc[h] = soc[h - 1] + RNG.normal(0, 0.3)
	soc = np.clip(soc, 10, 95)

	# PV output: bell curve centered at noon, scaled by season
	pv_peak = 200 + season_factor * 300  # kW
	pv_raw = pv_peak * np.exp(-0.5 * ((hours - 12) / 2.8) ** 2)
	pv_raw[:6] = 0
	pv_raw[19:] = 0
	pv_noise = RNG.normal(0, 10, size=24) * (pv_raw > 0)
	pv_kw = np.clip(pv_raw + pv_noise, 0, 600)

	# PV curtailment: only in summer midday when overvoltage risk
	pv_curtail = np.zeros(24)
	if season_factor > 0.6:
		curtail_hours = (hours >= 10) & (hours <= 15)
		pv_curtail[curtail_hours] = pv_kw[curtail_hours] * RNG.uniform(0.05, 0.20, size=curtail_hours.sum())

	return {
		"hours": hours,
		"cap_kvar": cap_kvar,
		"reg_tap": reg_tap.astype(float),
		"battery_soc": soc,
		"pv_kw": pv_kw,
		"pv_curtail": pv_curtail,
	}


def generate_training_data(n_episodes: int = 500) -> dict[str, np.ndarray]:
	"""Generate CMDP training curves: reward, Lagrangian objective decomposition, constraint rate.

	Returns:
		dict with keys: episodes, rewards, primal_obj, dual_penalty,
		lagrangian_obj, constraint_satisfaction
	"""
	episodes = np.arange(n_episodes)

	# Episode reward: starts bad, improves with noise
	reward_base = -50 + 45 * (1 - np.exp(-episodes / 120))
	reward_noise = RNG.normal(0, 3.0, size=n_episodes) * np.exp(-episodes / 300)
	rewards = reward_base + reward_noise

	# Primal objective J(pi): the reward part of CMDP
	primal_obj = rewards.copy()

	# Lagrangian multiplier (same trajectory as in lagrangian data)
	lambda_vals = 2.0 + 8.0 * np.exp(-episodes / 80)
	lambda_noise = RNG.normal(0, 0.2, size=n_episodes) * np.exp(-episodes / 200)
	lambda_vals = np.clip(lambda_vals + lambda_noise, 0.5, 12.0)

	# Constraint cost g(pi): violation magnitude
	g_pi = 0.12 / (1 + np.exp((episodes - 100) / 40))
	g_noise = RNG.uniform(-0.003, 0.003, size=n_episodes)
	g_pi = np.clip(g_pi + g_noise + 0.008, 0.001, 0.2)
	g_pi[400:] = np.clip(g_pi[400:] * 0.5, 0.002, 0.015)

	# Dual penalty: lambda * g(pi)
	dual_penalty = lambda_vals * g_pi * 100  # scaled for visibility

	# Lagrangian objective: J(pi) - lambda * g(pi)
	lagrangian_obj = primal_obj - dual_penalty

	# Constraint satisfaction rate: 1 - violation_rate
	violation_rate = g_pi / 0.12  # normalized
	constraint_satisfaction = np.clip((1 - violation_rate) * 100, 50, 100)
	constraint_satisfaction[350:] = np.clip(
		constraint_satisfaction[350:] + RNG.uniform(0, 2, size=150), 96, 100
	)

	return {
		"episodes": episodes,
		"rewards": rewards,
		"primal_obj": primal_obj,
		"dual_penalty": dual_penalty,
		"lagrangian_obj": lagrangian_obj,
		"constraint_satisfaction": constraint_satisfaction,
	}


# Pre-generate all demo data at module load
VOLTAGE_DATA = generate_voltage_heatmap_data()
LAGRANGIAN_DATA = generate_lagrangian_data()
TRAINING_DATA = generate_training_data()


# ============================================================
# Plot Factory Functions
# ============================================================

def plot_voltage_heatmap(time_window: str = "Full Year") -> go.Figure:
	"""Create voltage heatmap with bus names on y-axis and days on x-axis.

	Args:
		time_window: one of 'Full Year', 'Q1', 'Q2', 'Q3', 'Q4'.

	Returns:
		Plotly Figure with annotated heatmap.
	"""
	# Slice by quarter
	slices = {
		"Full Year": (0, 360),
		"Q1 (Jan-Mar)": (0, 90),
		"Q2 (Apr-Jun)": (90, 180),
		"Q3 (Jul-Sep)": (180, 270),
		"Q4 (Oct-Dec)": (270, 360),
	}
	start, end = slices.get(time_window, (0, 360))
	data = VOLTAGE_DATA[start:end, :].T  # shape: (n_buses, n_days)
	days = list(range(start, end))

	# Custom colorscale: blue (low) -> green (normal) -> red (high)
	colorscale = [
		[0.0, "#2563EB"],    # blue: severe undervoltage
		[0.25, "#3B82F6"],   # blue: undervoltage
		[0.40, "#10B981"],   # green: entering safe zone
		[0.50, "#059669"],   # dark green: nominal 1.0 pu
		[0.60, "#10B981"],   # green: leaving safe zone
		[0.75, "#F59E0B"],   # amber: overvoltage warning
		[1.0, "#EF4444"],    # red: severe overvoltage
	]

	fig = go.Figure(data=go.Heatmap(
		z=data,
		x=days,
		y=BUS_NAMES_13,
		colorscale=colorscale,
		zmin=0.92,
		zmax=1.10,
		colorbar=dict(
			title=dict(text="Voltage (p.u.)", side="right"),
			tickvals=[0.93, 0.95, 1.00, 1.05, 1.08],
			ticktext=["0.93", "0.95", "1.00", "1.05", "1.08"],
		),
		hovertemplate=(
			"Day: %{x}<br>"
			"Bus: %{y}<br>"
			"Voltage: %{z:.4f} p.u."
			"<extra></extra>"
		),
	))

	# Add violation boundary lines
	fig.add_hline(y=None)  # hlines not applicable for heatmap y
	# Instead, add shapes for voltage limit annotations
	fig.add_annotation(
		text="V_min=0.95 | V_max=1.05",
		xref="paper", yref="paper",
		x=1.0, y=1.05,
		showarrow=False,
		font=dict(size=11, color=COLORS["warning"]),
		xanchor="right",
	)

	layout_overrides = {**PLOTLY_LAYOUT_DEFAULTS}
	layout_overrides["yaxis"] = dict(
		gridcolor=COLORS["grid"],
		zerolinecolor=COLORS["grid"],
		type="category",
		title_text="Bus Name",
	)
	fig.update_layout(
		**layout_overrides,
		height=520,
		title=f"Bus Voltage Profile - {time_window}",
		xaxis_title="Day of Year",
	)
	return fig


def plot_lagrangian_trajectory() -> go.Figure:
	"""Create dual subplot: lambda convergence + constraint violation over episodes.

	Returns:
		Plotly Figure with 1x2 subplots.
	"""
	data = LAGRANGIAN_DATA
	ep = data["episodes"]

	fig = make_subplots(
		rows=1, cols=2,
		subplot_titles=(
			"Lagrangian Multiplier (lambda) Convergence",
			"Voltage Constraint Violation Rate",
		),
		horizontal_spacing=0.12,
	)

	# Lambda convergence
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["lambda_values"],
			mode="lines",
			name="lambda",
			line=dict(color=COLORS["primary"], width=1.8),
			opacity=0.7,
		),
		row=1, col=1,
	)
	# Smoothed lambda (moving average)
	window = 20
	lambda_smooth = np.convolve(data["lambda_values"], np.ones(window) / window, mode="valid")
	fig.add_trace(
		go.Scatter(
			x=ep[window - 1:], y=lambda_smooth,
			mode="lines",
			name="lambda (smoothed)",
			line=dict(color=COLORS["warning"], width=2.5),
		),
		row=1, col=1,
	)
	# Target lambda reference
	fig.add_hline(
		y=2.0, line_dash="dot", line_color=COLORS["text_dim"],
		annotation_text="converged ~2.0",
		annotation_position="bottom right",
		row=1, col=1,
	)

	# Constraint violation rate
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["violation_rate"],
			mode="lines",
			name="violation %",
			line=dict(color=COLORS["danger"], width=1.8),
			opacity=0.7,
		),
		row=1, col=2,
	)
	# Smoothed violation
	viol_smooth = np.convolve(data["violation_rate"], np.ones(window) / window, mode="valid")
	fig.add_trace(
		go.Scatter(
			x=ep[window - 1:], y=viol_smooth,
			mode="lines",
			name="violation % (smoothed)",
			line=dict(color=COLORS["accent"], width=2.5),
		),
		row=1, col=2,
	)
	# Target violation threshold
	fig.add_hline(
		y=2.0, line_dash="dot", line_color=COLORS["text_dim"],
		annotation_text="target < 2%",
		annotation_position="bottom right",
		row=1, col=2,
	)

	fig.update_xaxes(title_text="Training Episode", row=1, col=1)
	fig.update_xaxes(title_text="Training Episode", row=1, col=2)
	fig.update_yaxes(title_text="Lambda Value", row=1, col=1)
	fig.update_yaxes(title_text="Violation Rate (%)", row=1, col=2)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=450,
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
	)
	return fig


def plot_component_status(season: str = "Day 1 (Winter)") -> go.Figure:
	"""Create grouped bar/line chart showing 24-hour component operation.

	Args:
		season: one of the seasonal day selections.

	Returns:
		Plotly Figure with 2x2 subplots for each component type.
	"""
	day_map = {
		"Day 1 (Winter)": 1,
		"Day 91 (Spring)": 91,
		"Day 182 (Summer)": 182,
		"Day 273 (Autumn)": 273,
	}
	day = day_map.get(season, 1)
	data = generate_component_data(day)
	hours = data["hours"]

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"Capacitor Reactive Power",
			"Regulator Tap Position",
			"Battery State of Charge",
			"PV Output & Curtailment",
		),
		vertical_spacing=0.15,
		horizontal_spacing=0.10,
	)

	# Capacitor kVar
	fig.add_trace(
		go.Bar(
			x=hours, y=data["cap_kvar"],
			name="Cap kVar",
			marker_color=COLORS["primary"],
			opacity=0.85,
		),
		row=1, col=1,
	)

	# Regulator tap
	fig.add_trace(
		go.Scatter(
			x=hours, y=data["reg_tap"],
			mode="lines+markers",
			name="Reg Tap",
			line=dict(color=COLORS["accent"], width=2),
			marker=dict(size=6),
		),
		row=1, col=2,
	)
	fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_dim"], row=1, col=2)

	# Battery SOC
	fig.add_trace(
		go.Scatter(
			x=hours, y=data["battery_soc"],
			mode="lines+markers",
			name="SOC %",
			line=dict(color=COLORS["warning"], width=2.5),
			marker=dict(size=5),
			fill="tozeroy",
			fillcolor="rgba(245, 158, 11, 0.1)",
		),
		row=2, col=1,
	)
	# SOC bounds
	fig.add_hline(y=20, line_dash="dot", line_color=COLORS["danger"], row=2, col=1)
	fig.add_hline(y=90, line_dash="dot", line_color=COLORS["danger"], row=2, col=1)

	# PV output + curtailment stacked
	fig.add_trace(
		go.Bar(
			x=hours, y=data["pv_kw"] - data["pv_curtail"],
			name="PV Delivered (kW)",
			marker_color=COLORS["secondary"],
		),
		row=2, col=2,
	)
	fig.add_trace(
		go.Bar(
			x=hours, y=data["pv_curtail"],
			name="PV Curtailed (kW)",
			marker_color=COLORS["danger"],
			opacity=0.7,
		),
		row=2, col=2,
	)

	# Axis labels
	for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
		fig.update_xaxes(title_text="Hour", row=row, col=col)
	fig.update_yaxes(title_text="kVar", row=1, col=1)
	fig.update_yaxes(title_text="Tap Position", row=1, col=2)
	fig.update_yaxes(title_text="SOC (%)", row=2, col=1)
	fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=600,
		barmode="stack",
		title=f"Component Operation - {season} (Day {day})",
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
	)
	return fig


def plot_training_rewards() -> go.Figure:
	"""Plot episode reward curve over CMDP training.

	Returns:
		Plotly Figure with raw + smoothed reward curves.
	"""
	data = TRAINING_DATA
	ep = data["episodes"]
	window = 20

	fig = go.Figure()

	# Raw rewards
	fig.add_trace(go.Scatter(
		x=ep, y=data["rewards"],
		mode="lines",
		name="Episode Reward (raw)",
		line=dict(color=COLORS["primary"], width=1),
		opacity=0.4,
	))

	# Smoothed rewards
	smooth = np.convolve(data["rewards"], np.ones(window) / window, mode="valid")
	fig.add_trace(go.Scatter(
		x=ep[window - 1:], y=smooth,
		mode="lines",
		name="Episode Reward (smoothed)",
		line=dict(color=COLORS["primary"], width=2.5),
	))

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title="CMDP Episode Reward (SmartGrid 34-Bus PV)",
		xaxis_title="Training Episode",
		yaxis_title="Episode Reward",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_lagrangian_decomposition() -> go.Figure:
	"""Plot Lagrangian objective decomposition: J(pi) - lambda*g(pi).

	Shows primal objective, dual penalty, and combined Lagrangian objective.

	Returns:
		Plotly Figure with three overlaid curves.
	"""
	data = TRAINING_DATA
	ep = data["episodes"]
	window = 25

	fig = go.Figure()

	# Helper: smooth curve
	def _smooth(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		s = np.convolve(arr, np.ones(window) / window, mode="valid")
		return ep[window - 1:], s

	# Primal objective J(pi)
	x, y = _smooth(data["primal_obj"])
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode="lines",
		name="J(pi) Primal Objective",
		line=dict(color=COLORS["accent"], width=2.2),
	))

	# Dual penalty lambda*g(pi)
	x, y = _smooth(data["dual_penalty"])
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode="lines",
		name="lambda * g(pi) Dual Penalty",
		line=dict(color=COLORS["danger"], width=2.2, dash="dash"),
	))

	# Lagrangian objective L = J(pi) - lambda*g(pi)
	x, y = _smooth(data["lagrangian_obj"])
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode="lines",
		name="L(pi, lambda) Lagrangian",
		line=dict(color=COLORS["warning"], width=2.8),
	))

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title="Lagrangian Objective Decomposition: L = J(pi) - lambda * g(pi)",
		xaxis_title="Training Episode",
		yaxis_title="Objective Value",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
	)
	return fig


def plot_constraint_satisfaction() -> go.Figure:
	"""Plot constraint satisfaction rate over training.

	Returns:
		Plotly Figure with satisfaction rate and 95% target line.
	"""
	data = TRAINING_DATA
	ep = data["episodes"]
	window = 20

	fig = go.Figure()

	# Raw
	fig.add_trace(go.Scatter(
		x=ep, y=data["constraint_satisfaction"],
		mode="lines",
		name="Constraint Satisfaction (raw)",
		line=dict(color=COLORS["secondary"], width=1),
		opacity=0.35,
	))

	# Smoothed
	smooth = np.convolve(data["constraint_satisfaction"], np.ones(window) / window, mode="valid")
	fig.add_trace(go.Scatter(
		x=ep[window - 1:], y=smooth,
		mode="lines",
		name="Constraint Satisfaction (smoothed)",
		line=dict(color=COLORS["secondary"], width=2.5),
	))

	# 95% target
	fig.add_hline(
		y=95, line_dash="dot", line_color=COLORS["warning"],
		annotation_text="95% target",
		annotation_position="bottom right",
	)
	# 98% excellent
	fig.add_hline(
		y=98, line_dash="dot", line_color=COLORS["primary"],
		annotation_text="98% excellent",
		annotation_position="top right",
	)

	layout_overrides = {**PLOTLY_LAYOUT_DEFAULTS}
	layout_overrides["yaxis"] = dict(
		range=[50, 102],
		gridcolor=COLORS["grid"],
		zerolinecolor=COLORS["grid"],
		title_text="Satisfaction (%)",
	)
	fig.update_layout(
		**layout_overrides,
		height=350,
		title="Voltage Constraint Satisfaction Rate",
		xaxis_title="Training Episode",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


# ============================================================
# Build Gradio App
# ============================================================

def build_app() -> gr.Blocks:
	"""Construct the Gradio Blocks application with 5 tabs."""
	with gr.Blocks(
		title="PowerZoo SmartGrid: CMDP Environment Demo",
		theme=gr.themes.Soft(primary_hue="emerald"),
	) as app:
		# Header
		gr.Markdown(
			"""
			# PowerZoo SmartGrid: Modular PV Integration with CMDP
			**Constrained MDP** | **Lagrangian Relaxation** | **360-Step Annual Episodes** | **Homogeneous Agents**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# Tab 1: Overview
			# --------------------------------------------------------
			with gr.Tab("Overview"):
				gr.Markdown(
					"""
					## SmartGrid Environment

					SmartGrid is a **modular PV integration environment** built on the Constrained Markov
					Decision Process (CMDP) framework with Lagrangian relaxation. Unlike standard MDP
					formulations that fold voltage constraints into the reward, SmartGrid separates the
					**economic objective** (power loss minimization, control smoothness) from the
					**safety constraint** (voltage regulation within 0.95-1.05 p.u.).

					### Key Specifications

					| Property | Value |
					|----------|-------|
					| **Agent Type** | Homogeneous (shared policy) |
					| **Episode Length** | 360 steps (1 step = 1 day, annual cycle) |
					| **Framework** | CMDP with Lagrangian multiplier |
					| **Controlled Devices** | Capacitors, Regulators, Batteries, PV Systems |
					| **Voltage Limits** | 0.95 - 1.05 p.u. (ANSI C84.1) |
					| **Reward** | Power loss + control cost + PV utilization |
					| **Constraint** | Voltage violation rate (squared hinge loss) |

					### What Makes SmartGrid Unique

					- **CMDP Formulation**: The Lagrangian multiplier `lambda` automatically balances
					  economic performance against voltage safety. No manual penalty tuning required.
					- **OOP Circuit Modeling**: Component-based architecture (Capacitor, Regulator,
					  Battery, PV) with SmartGrid's own `Circuit` class wrapping OpenDSS.
					- **Annual Episodes**: 360-step episodes capture seasonal load/PV variation,
					  enabling agents to learn long-horizon strategies.
					- **Curriculum Learning**: Three-phase training (exploration -> optimization ->
					  refinement) with progressive constraint tightening.

					### Supported IEEE Test Systems

					| System | Buses | Agents | Complexity |
					|--------|-------|--------|------------|
					| **13-Bus** | 13 | 2-4 | Rapid prototyping |
					| **34-Bus_PV** | 34 | 6-9 | PV integration studies |
					| **123-Bus** | 123 | 12-20 | Medium-scale validation |
					| **8500-Node** | 8500 | 50+ | Large-scale stress test |

					PV variants available: Conservative, Optimized, Aggressive penetration levels.

					### CMDP Objective

					The agent optimizes the Lagrangian:

					**L(pi, lambda) = J(pi) - lambda * g(pi)**

					where `J(pi)` is the primal reward (economic), `g(pi)` is the constraint cost
					(voltage violations), and `lambda` is the dual variable updated via gradient ascent.
					"""
				)

			# --------------------------------------------------------
			# Tab 2: Voltage Heatmap
			# --------------------------------------------------------
			with gr.Tab("Voltage Heatmap"):
				gr.Markdown(
					"""
					## Bus Voltage Heatmap (IEEE 13-Bus Demo)
					Visualize voltage profiles across all buses over the year.
					**Blue** = undervoltage (<0.95), **Green** = normal (0.95-1.05), **Red** = overvoltage (>1.05).
					Summer PV injection causes overvoltage on downstream buses; winter loads pull voltage down.
					"""
				)
				time_dropdown = gr.Dropdown(
					choices=["Full Year", "Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"],
					value="Full Year",
					label="Time Window",
				)
				heatmap_plot = gr.Plot(value=plot_voltage_heatmap("Full Year"))

				time_dropdown.change(
					fn=plot_voltage_heatmap,
					inputs=time_dropdown,
					outputs=heatmap_plot,
				)

			# --------------------------------------------------------
			# Tab 3: Lagrangian Trajectory
			# --------------------------------------------------------
			with gr.Tab("Lagrangian Trajectory"):
				gr.Markdown(
					"""
					## CMDP Lagrangian Convergence
					The Lagrangian multiplier `lambda` starts high (~10) to enforce strict voltage constraints,
					then converges to ~2 as the policy learns to satisfy constraints naturally.
					The constraint violation rate drops from ~15% to below 2%.

					This dual convergence is the signature behavior of CMDP training with Lagrangian relaxation.
					"""
				)
				lagrangian_plot = gr.Plot(value=plot_lagrangian_trajectory())

			# --------------------------------------------------------
			# Tab 4: Component Status
			# --------------------------------------------------------
			with gr.Tab("Component Status"):
				gr.Markdown(
					"""
					## 24-Hour Component Operation
					View how capacitors, regulators, batteries, and PV systems operate across a full day.
					Select different seasons to see how operation patterns change with load and PV availability.
					"""
				)
				season_radio = gr.Radio(
					choices=["Day 1 (Winter)", "Day 91 (Spring)", "Day 182 (Summer)", "Day 273 (Autumn)"],
					value="Day 182 (Summer)",
					label="Select Day of Year",
				)
				component_plot = gr.Plot(value=plot_component_status("Day 182 (Summer)"))

				season_radio.change(
					fn=plot_component_status,
					inputs=season_radio,
					outputs=component_plot,
				)

			# --------------------------------------------------------
			# Tab 5: Training Dashboard
			# --------------------------------------------------------
			with gr.Tab("Training Dashboard"):
				gr.Markdown(
					"""
					## CMDP Training Dashboard (SmartGrid 34-Bus PV)
					Training curves showing both primal (reward) and dual (constraint) convergence.
					The Lagrangian objective decomposes into J(pi) and the penalty term lambda * g(pi).
					"""
				)

				gr.Markdown("### Episode Reward")
				reward_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### Lagrangian Objective Decomposition")
				decomp_plot = gr.Plot(value=plot_lagrangian_decomposition())

				gr.Markdown("### Constraint Satisfaction Rate")
				constraint_plot = gr.Plot(value=plot_constraint_satisfaction())

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
