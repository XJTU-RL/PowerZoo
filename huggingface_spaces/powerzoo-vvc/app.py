"""
PowerZoo VVC (Volt-VAR Control) Environment Demo
HuggingFace Space - Self-contained Gradio + Plotly application.

5 Tabs: Overview | Voltage Profile | Device Schedule | Reward Analysis | Training Dashboard
"""
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Monkey-patch: fix Gradio 6.x + Plotly additionalProperties schema error ===
_original_plot_init = gr.Plot.__init__


def _patched_plot_init(self, *args, **kwargs):
	_original_plot_init(self, *args, **kwargs)
	if hasattr(self, "schema") and isinstance(self.schema, dict):
		self.schema.pop("additionalProperties", None)


gr.Plot.__init__ = _patched_plot_init


# ============================================================
# Color Palette & Theme
# ============================================================
COLORS = {
	"primary": "#6366F1",
	"secondary": "#8B5CF6",
	"accent": "#22D3EE",
	"warning": "#F59E0B",
	"danger": "#EF4444",
	"success": "#10B981",
	"bg": "#0F172A",
	"surface": "#1E293B",
	"text": "#E2E8F0",
	"muted": "#94A3B8",
	"agents": ["#6366F1", "#8B5CF6", "#22D3EE", "#F59E0B", "#EF4444", "#10B981"],
}

PLOTLY_LAYOUT = dict(
	template="plotly_dark",
	paper_bgcolor=COLORS["bg"],
	plot_bgcolor=COLORS["surface"],
	font=dict(color=COLORS["text"], family="Inter, sans-serif"),
	margin=dict(l=50, r=30, t=50, b=50),
	hoverlabel=dict(bgcolor=COLORS["surface"], font_color=COLORS["text"]),
)


# ============================================================
# Demo Data Generators
# ============================================================
# All data is deterministic (seeded) so the demo is reproducible.


def _seed() -> np.random.Generator:
	"""Return a seeded random generator for reproducible demo data."""
	return np.random.default_rng(42)


# --- IEEE 13-Bus names ---
BUS_NAMES: list[str] = [
	"650", "632", "633", "634", "645", "646", "671",
	"680", "684", "611", "652", "692", "675",
]

# --- Base voltage profile (pu) for 13 buses at noon ---
_BASE_VOLTAGES = np.array([
	1.040, 1.025, 1.018, 1.012, 1.008, 1.005, 0.990,
	0.985, 0.978, 0.965, 0.958, 0.992, 0.988,
])


def generate_voltage_profile(step: int) -> np.ndarray:
	"""Generate realistic 13-bus voltage magnitudes for a given hour (0-23).

	Night hours (0-6, 20-23): slightly lower voltages due to light load.
	Midday (10-14): PV injection pushes upstream buses high, downstream stays moderate.
	Evening peak (17-19): heavy load sags voltage.
	"""
	rng = np.random.default_rng(step * 137 + 7)
	hour_offset = np.zeros(13)

	if 0 <= step <= 5:
		# Night: low load, voltages drift slightly below nominal
		hour_offset = np.array([
			-0.005, -0.008, -0.010, -0.012, -0.015, -0.016, -0.020,
			-0.022, -0.025, -0.030, -0.032, -0.018, -0.020,
		])
	elif 6 <= step <= 9:
		# Morning ramp: load increases, PV starts
		t = (step - 6) / 3.0
		hour_offset = np.array([
			0.002, 0.000, -0.002, -0.005, -0.008, -0.010, -0.015,
			-0.018, -0.020, -0.025, -0.028, -0.012, -0.014,
		]) * (1.0 - 0.5 * t)
	elif 10 <= step <= 14:
		# Midday peak PV: upstream voltages rise, downstream moderate
		hour_offset = np.array([
			0.010, 0.008, 0.005, 0.003, 0.000, -0.002, -0.008,
			-0.010, -0.015, -0.020, -0.022, -0.005, -0.008,
		])
	elif 15 <= step <= 16:
		# Afternoon transition
		hour_offset = np.array([
			0.005, 0.002, -0.002, -0.006, -0.010, -0.012, -0.018,
			-0.020, -0.024, -0.028, -0.030, -0.015, -0.018,
		])
	elif 17 <= step <= 19:
		# Evening peak: heavy load, voltage sags
		hour_offset = np.array([
			-0.008, -0.012, -0.018, -0.022, -0.028, -0.030, -0.038,
			-0.042, -0.048, -0.055, -0.058, -0.035, -0.040,
		])
	else:
		# Late evening (20-23): load decreasing
		hour_offset = np.array([
			-0.003, -0.006, -0.009, -0.012, -0.016, -0.018, -0.024,
			-0.028, -0.032, -0.038, -0.040, -0.022, -0.025,
		])

	noise = rng.normal(0, 0.003, size=13)
	return _BASE_VOLTAGES + hour_offset + noise


def generate_device_schedules() -> dict[str, np.ndarray]:
	"""Generate 24-step device operation profiles.

	Returns dict with keys:
	  cap1, cap2: (24,) int {0, 1}
	  reg1, reg2: (24,) int [0, 16]
	  battery_kw: (24,) float (negative=charge, positive=discharge)
	  pv_output_kw: (24,) float
	  pv_curtail_kw: (24,) float
	"""
	rng = _seed()
	hours = np.arange(24)

	# Capacitors: on during high-load periods
	cap1 = np.zeros(24, dtype=int)
	cap1[7:21] = 1
	cap1[12:14] = 0  # Brief switch during midday PV peak
	cap2 = np.zeros(24, dtype=int)
	cap2[9:20] = 1

	# Regulators: tap varies with voltage needs
	reg1_base = 8 * np.ones(24, dtype=int)
	reg1_base[0:6] = 10
	reg1_base[6:10] = 9
	reg1_base[10:15] = 6
	reg1_base[15:17] = 8
	reg1_base[17:20] = 12
	reg1_base[20:24] = 10
	reg1 = np.clip(reg1_base + rng.integers(-1, 2, size=24), 0, 16)

	reg2_base = 7 * np.ones(24, dtype=int)
	reg2_base[0:6] = 9
	reg2_base[10:15] = 5
	reg2_base[17:20] = 11
	reg2 = np.clip(reg2_base + rng.integers(-1, 2, size=24), 0, 16)

	# Battery: charge from PV midday, discharge evening peak
	battery_kw = np.zeros(24)
	battery_kw[10:14] = -np.array([80, 120, 130, 100])  # Charge
	battery_kw[17:21] = np.array([100, 140, 120, 60])  # Discharge
	battery_kw += rng.normal(0, 5, size=24)
	battery_kw[:6] = rng.normal(0, 3, size=6)

	# PV output: bell curve peaking at noon
	pv_max = 350.0
	solar_envelope = pv_max * np.exp(-0.5 * ((hours - 12.5) / 3.0) ** 2)
	solar_envelope[:6] = 0
	solar_envelope[20:] = 0
	cloud_factor = np.ones(24)
	cloud_factor[9] = 0.6
	cloud_factor[13] = 0.75
	pv_output_kw = solar_envelope * cloud_factor + rng.normal(0, 5, size=24)
	pv_output_kw = np.clip(pv_output_kw, 0, pv_max)

	# PV curtailment: agent reduces output during overvoltage
	pv_curtail_kw = np.zeros(24)
	pv_curtail_kw[11:14] = np.array([20, 45, 30])
	pv_curtail_kw += rng.uniform(0, 5, size=24)
	pv_curtail_kw = np.clip(pv_curtail_kw, 0, pv_output_kw * 0.3)

	return {
		"cap1": cap1,
		"cap2": cap2,
		"reg1": reg1,
		"reg2": reg2,
		"battery_kw": battery_kw,
		"pv_output_kw": pv_output_kw,
		"pv_curtail_kw": pv_curtail_kw,
	}


def generate_reward_data() -> dict[str, np.ndarray]:
	"""Generate 24-step reward component data.

	Reward components (all negative, closer to 0 is better):
	  power_loss: proportional to line losses
	  voltage_violation: penalty for out-of-band voltages
	  control_penalty: penalty for device switching
	"""
	rng = _seed()
	hours = np.arange(24)

	# Power loss: moderate baseline, higher during peak
	loss_base = -0.3 * np.ones(24)
	loss_base[17:20] = -0.6  # Evening peak
	loss_base[10:14] = -0.2  # PV reduces loss
	power_loss = loss_base + rng.normal(0, 0.03, size=24)

	# Voltage violation: high early morning & evening, low midday
	vv_base = np.zeros(24)
	vv_base[0:6] = -0.15
	vv_base[17:20] = -0.35
	vv_base[20:24] = -0.12
	vv_base[10:14] = -0.05
	voltage_violation = vv_base + rng.normal(0, 0.02, size=24)
	voltage_violation = np.clip(voltage_violation, -1.0, 0.0)

	# Control penalty: spike when devices switch
	control_penalty = rng.uniform(-0.05, 0.0, size=24)
	control_penalty[7] = -0.20  # Cap switch-on
	control_penalty[12] = -0.15  # Cap toggle
	control_penalty[17] = -0.18  # Reg big tap change
	control_penalty[21] = -0.12  # Cap switch-off

	return {
		"power_loss": power_loss,
		"voltage_violation": voltage_violation,
		"control_penalty": control_penalty,
	}


def generate_training_data() -> dict[str, np.ndarray]:
	"""Generate synthetic HAPPO training curves (2000 episodes).

	Returns dict with:
	  episodes: (2000,) int
	  episode_rewards: (2000,) float - total reward per episode
	  agent_policy_loss: (2000, 6) float - per-agent policy loss
	  power_loss_kw: (2000,) float - episode-mean power loss
	"""
	rng = _seed()
	n_ep = 2000
	episodes = np.arange(n_ep)

	# Episode reward: starts around -15, converges to ~ -4
	# Exponential decay + noise
	converged = -4.0
	initial = -15.0
	tau = 400.0  # Decay constant
	base_curve = converged + (initial - converged) * np.exp(-episodes / tau)
	noise = rng.normal(0, 0.8, size=n_ep)
	# Smoothed noise for realistic jitter
	kernel = np.ones(20) / 20.0
	smooth_noise = np.convolve(noise, kernel, mode="same")
	episode_rewards = base_curve + smooth_noise

	# Per-agent policy loss: 6 agents, each converges differently
	agent_policy_loss = np.zeros((n_ep, 6))
	for i in range(6):
		agent_tau = 300 + i * 60
		agent_init = 2.5 + rng.uniform(-0.3, 0.3)
		agent_final = 0.3 + rng.uniform(-0.05, 0.05)
		agent_curve = agent_final + (agent_init - agent_final) * np.exp(-episodes / agent_tau)
		agent_noise = rng.normal(0, 0.15, size=n_ep)
		agent_smooth = np.convolve(agent_noise, kernel, mode="same")
		agent_policy_loss[:, i] = agent_curve + agent_smooth

	# Power loss reduction: starts ~180 kW, drops to ~90 kW
	pl_init = 180.0
	pl_final = 90.0
	pl_tau = 500.0
	power_loss_kw = pl_final + (pl_init - pl_final) * np.exp(-episodes / pl_tau)
	power_loss_kw += rng.normal(0, 5.0, size=n_ep)

	return {
		"episodes": episodes,
		"episode_rewards": episode_rewards,
		"agent_policy_loss": agent_policy_loss,
		"power_loss_kw": power_loss_kw,
	}


# Pre-generate all demo data
DEVICE_DATA = generate_device_schedules()
REWARD_DATA = generate_reward_data()
TRAINING_DATA = generate_training_data()


# ============================================================
# Plot Factory Functions
# ============================================================


def plot_voltage_profile(step: int = 12) -> go.Figure:
	"""Create interactive bar chart of 13-bus voltage magnitudes.

	Args:
		step: Hour of day (0-23).

	Returns:
		Plotly Figure with colored bars and reference lines.
	"""
	voltages = generate_voltage_profile(step)

	# Color coding by voltage status
	bar_colors = []
	for v in voltages:
		if 0.95 <= v <= 1.05:
			bar_colors.append(COLORS["success"])
		elif (0.93 <= v < 0.95) or (1.05 < v <= 1.07):
			bar_colors.append(COLORS["warning"])
		else:
			bar_colors.append(COLORS["danger"])

	fig = go.Figure()

	fig.add_trace(go.Bar(
		x=BUS_NAMES,
		y=voltages,
		marker=dict(color=bar_colors, line=dict(width=1, color=COLORS["muted"])),
		text=[f"{v:.4f}" for v in voltages],
		textposition="outside",
		textfont=dict(size=10, color=COLORS["text"]),
		hovertemplate="Bus %{x}<br>Voltage: %{y:.4f} pu<extra></extra>",
	))

	# Reference lines
	fig.add_hline(y=1.05, line_dash="dash", line_color=COLORS["warning"],
				  annotation_text="Upper limit (1.05)", annotation_position="top right",
				  annotation_font_color=COLORS["warning"])
	fig.add_hline(y=0.95, line_dash="dash", line_color=COLORS["warning"],
				  annotation_text="Lower limit (0.95)", annotation_position="bottom right",
				  annotation_font_color=COLORS["warning"])

	# Color legend via invisible traces
	for label, color in [("Normal (0.95-1.05)", COLORS["success"]),
						 ("Warning", COLORS["warning"]),
						 ("Violation", COLORS["danger"])]:
		fig.add_trace(go.Bar(
			x=[None], y=[None],
			marker=dict(color=color),
			name=label,
			showlegend=True,
		))

	fig.update_layout(
		**PLOTLY_LAYOUT,
		title=f"IEEE 13-Bus Voltage Profile - Hour {step}:00",
		xaxis_title="Bus ID",
		yaxis_title="Voltage Magnitude (pu)",
		yaxis=dict(range=[0.92, 1.08]),
		height=520,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		bargap=0.15,
	)
	return fig


def plot_device_schedule() -> go.Figure:
	"""Create 2x2 subplot showing 24-step device operations.

	Subplots:
	  1. Capacitor Status (step plot, on/off)
	  2. Regulator Tap Position (line plot, 0-16)
	  3. Battery Power (bar chart, charge/discharge)
	  4. PV Output with curtailment shading (area chart)
	"""
	hours = list(range(24))
	d = DEVICE_DATA

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"Capacitor Status", "Regulator Tap Position",
			"Battery Power (kW)", "PV Output & Curtailment (kW)",
		),
		vertical_spacing=0.14,
		horizontal_spacing=0.10,
	)

	# --- Subplot 1: Capacitors (step plot) ---
	for name, data, color, offset in [
		("Cap 1", d["cap1"], COLORS["primary"], 0),
		("Cap 2", d["cap2"], COLORS["accent"], 0),
	]:
		fig.add_trace(go.Scatter(
			x=hours, y=data,
			mode="lines",
			name=name,
			line=dict(shape="hv", color=color, width=2.5),
			legendgroup="cap",
		), row=1, col=1)

	fig.update_yaxes(tickvals=[0, 1], ticktext=["OFF", "ON"], range=[-0.1, 1.3], row=1, col=1)

	# --- Subplot 2: Regulators (line plot) ---
	for name, data, color in [
		("Reg 1", d["reg1"], COLORS["secondary"]),
		("Reg 2", d["reg2"], COLORS["warning"]),
	]:
		fig.add_trace(go.Scatter(
			x=hours, y=data,
			mode="lines+markers",
			name=name,
			line=dict(color=color, width=2),
			marker=dict(size=5),
			legendgroup="reg",
		), row=1, col=2)

	fig.update_yaxes(range=[-0.5, 16.5], dtick=4, row=1, col=2)

	# --- Subplot 3: Battery (bar chart) ---
	bat_colors = [COLORS["accent"] if v >= 0 else COLORS["secondary"] for v in d["battery_kw"]]
	fig.add_trace(go.Bar(
		x=hours, y=d["battery_kw"],
		name="Battery",
		marker=dict(color=bat_colors, line=dict(width=0.5, color=COLORS["muted"])),
		showlegend=True,
		legendgroup="bat",
		hovertemplate="Hour %{x}<br>Power: %{y:.1f} kW<extra></extra>",
	), row=2, col=1)

	fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"], row=2, col=1)

	# --- Subplot 4: PV Output (area) + Curtailment shading ---
	net_pv = d["pv_output_kw"] - d["pv_curtail_kw"]

	# Available (total) as upper envelope
	fig.add_trace(go.Scatter(
		x=hours, y=d["pv_output_kw"],
		mode="lines",
		name="PV Available",
		line=dict(color=COLORS["warning"], width=1, dash="dot"),
		fill="tozeroy",
		fillcolor="rgba(245, 158, 11, 0.15)",
		legendgroup="pv",
	), row=2, col=2)

	# Actual output (after curtailment) as solid area
	fig.add_trace(go.Scatter(
		x=hours, y=net_pv,
		mode="lines",
		name="PV Delivered",
		line=dict(color=COLORS["warning"], width=2.5),
		fill="tozeroy",
		fillcolor="rgba(245, 158, 11, 0.35)",
		legendgroup="pv",
	), row=2, col=2)

	# Global layout
	fig.update_layout(
		**PLOTLY_LAYOUT,
		height=680,
		title_text="24-Hour Device Operation Schedule",
		legend=dict(
			orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
			font=dict(size=11),
		),
	)

	# Common x-axis styling
	for row in [1, 2]:
		for col in [1, 2]:
			fig.update_xaxes(title_text="Hour", dtick=4, row=row, col=col)

	return fig


def plot_reward_analysis() -> go.Figure:
	"""Create stacked bar chart of reward components with cumulative line.

	Left y-axis: stacked bars (power_loss + voltage_violation + control_penalty).
	Right y-axis: cumulative total reward line.
	"""
	hours = list(range(24))
	r = REWARD_DATA

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# Stacked bars (all negative values)
	for name, data, color in [
		("Power Loss", r["power_loss"], COLORS["primary"]),
		("Voltage Violation", r["voltage_violation"], COLORS["danger"]),
		("Control Penalty", r["control_penalty"], COLORS["warning"]),
	]:
		fig.add_trace(go.Bar(
			x=hours, y=data,
			name=name,
			marker=dict(color=color, opacity=0.85),
			hovertemplate=f"{name}<br>Hour %{{x}}: %{{y:.3f}}<extra></extra>",
		), secondary_y=False)

	# Cumulative total reward line
	total_per_step = r["power_loss"] + r["voltage_violation"] + r["control_penalty"]
	cumulative = np.cumsum(total_per_step)

	fig.add_trace(go.Scatter(
		x=hours, y=cumulative,
		mode="lines+markers",
		name="Cumulative Reward",
		line=dict(color=COLORS["accent"], width=3),
		marker=dict(size=6, symbol="diamond"),
		hovertemplate="Hour %{x}<br>Cumulative: %{y:.2f}<extra></extra>",
	), secondary_y=True)

	fig.update_layout(
		**PLOTLY_LAYOUT,
		barmode="relative",
		height=520,
		title="Episode Reward Decomposition (24 Steps)",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
		bargap=0.12,
	)

	fig.update_xaxes(title_text="Hour", dtick=2)
	fig.update_yaxes(title_text="Step Reward", secondary_y=False)
	fig.update_yaxes(title_text="Cumulative Reward", secondary_y=True,
					 gridcolor="rgba(148, 163, 184, 0.1)")

	return fig


def plot_training_rewards() -> go.Figure:
	"""Plot episode reward curve over 2000 episodes with rolling mean."""
	t = TRAINING_DATA
	ep = t["episodes"]
	rw = t["episode_rewards"]

	# Rolling mean (window=50)
	window = 50
	rolling = np.convolve(rw, np.ones(window) / window, mode="valid")
	rolling_x = ep[window - 1:]

	fig = go.Figure()

	# Raw rewards (faded)
	fig.add_trace(go.Scatter(
		x=ep, y=rw,
		mode="lines",
		name="Raw Reward",
		line=dict(color=COLORS["primary"], width=0.8),
		opacity=0.3,
	))

	# Rolling mean
	fig.add_trace(go.Scatter(
		x=rolling_x, y=rolling,
		mode="lines",
		name=f"Rolling Mean ({window} ep)",
		line=dict(color=COLORS["accent"], width=2.5),
	))

	fig.update_layout(
		**PLOTLY_LAYOUT,
		height=450,
		title="HAPPO Training - Episode Rewards (IEEE 13-Bus VVC)",
		xaxis_title="Episode",
		yaxis_title="Total Episode Reward",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_agent_policy_loss() -> go.Figure:
	"""Plot per-agent policy loss comparison (6 agents)."""
	t = TRAINING_DATA
	ep = t["episodes"]
	losses = t["agent_policy_loss"]
	window = 30

	fig = go.Figure()

	for i in range(6):
		raw = losses[:, i]
		smooth = np.convolve(raw, np.ones(window) / window, mode="valid")
		fig.add_trace(go.Scatter(
			x=ep[window - 1:], y=smooth,
			mode="lines",
			name=f"Agent {i}",
			line=dict(color=COLORS["agents"][i], width=2),
		))

	fig.update_layout(
		**PLOTLY_LAYOUT,
		height=450,
		title="Per-Agent Policy Loss (Smoothed)",
		xaxis_title="Episode",
		yaxis_title="Policy Loss",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
	)
	return fig


def plot_power_loss_reduction() -> go.Figure:
	"""Plot power loss (kW) reduction over training."""
	t = TRAINING_DATA
	ep = t["episodes"]
	pl = t["power_loss_kw"]
	window = 50

	rolling = np.convolve(pl, np.ones(window) / window, mode="valid")
	rolling_x = ep[window - 1:]

	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=ep, y=pl,
		mode="lines",
		name="Raw",
		line=dict(color=COLORS["danger"], width=0.8),
		opacity=0.25,
	))

	fig.add_trace(go.Scatter(
		x=rolling_x, y=rolling,
		mode="lines",
		name=f"Rolling Mean ({window} ep)",
		line=dict(color=COLORS["success"], width=2.5),
	))

	# Initial and final annotations
	fig.add_annotation(
		x=0, y=pl[0],
		text=f"Initial: {pl[0]:.0f} kW",
		showarrow=True, arrowhead=2,
		font=dict(color=COLORS["danger"]),
		arrowcolor=COLORS["danger"],
	)
	fig.add_annotation(
		x=1950, y=rolling[-50],
		text=f"Converged: {rolling[-50]:.0f} kW",
		showarrow=True, arrowhead=2,
		font=dict(color=COLORS["success"]),
		arrowcolor=COLORS["success"],
	)

	fig.update_layout(
		**PLOTLY_LAYOUT,
		height=450,
		title="Distribution Power Loss Reduction During Training",
		xaxis_title="Episode",
		yaxis_title="Power Loss (kW)",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


# ============================================================
# Gradio Application
# ============================================================


def build_app() -> gr.Blocks:
	"""Construct the Gradio Blocks application with 5 tabs."""
	with gr.Blocks(
		title="PowerZoo VVC - Volt-VAR Control Demo",
		theme=gr.themes.Soft(primary_hue="indigo"),
		css="""
		.footer-text {
			text-align: center;
			color: #94A3B8;
			font-size: 0.85em;
			padding: 16px 0;
		}
		""",
	) as app:
		# Header
		gr.Markdown(
			"""
			# PowerZoo VVC: Volt-VAR Control Environment
			**6 Agents** · **24 Steps/Episode** · **Mixed Action Space** · **IEEE Distribution Systems**
			"""
		)

		with gr.Tabs():
			# ================================================================
			# Tab 1: Overview
			# ================================================================
			with gr.Tab("Overview"):
				gr.Markdown(
					"""
					## Environment Description

					The **VVC (Volt-VAR Control)** environment simulates real-time voltage and
					reactive power management on IEEE distribution networks using OpenDSS as the
					power flow backend. Agents cooperatively control capacitor banks, voltage
					regulators, battery energy storage systems, and PV inverters to minimize
					power losses while maintaining voltage within ANSI limits (0.95-1.05 pu).

					Each episode spans **24 hourly time steps** (one day). The environment supports
					**6 homogeneous agents**, each responsible for a subset of controllable devices.
					The multi-agent formulation enables scalable control on large distribution networks
					where centralized optimization becomes intractable.

					### Key Specifications
					"""
				)

				# Specs table
				specs_df = pd.DataFrame([
					{"Parameter": "Agents", "Value": "6 (homogeneous)"},
					{"Parameter": "Episode Length", "Value": "24 steps (hourly)"},
					{"Parameter": "Action Space", "Value": "Mixed: discrete (cap/reg) + continuous (bat/PV)"},
					{"Parameter": "Observation", "Value": "Bus voltages, power flows, device states, load/PV profiles"},
					{"Parameter": "Reward", "Value": "power_loss + voltage_violation + control_penalty"},
					{"Parameter": "Backend", "Value": "OpenDSS via dss-python"},
					{"Parameter": "Algorithms", "Value": "HAPPO, MAPPO, HATRPO, HADDPG, HASAC, QMix, ..."},
				])
				gr.Dataframe(
					value=specs_df,
					label="Environment Specifications",
					interactive=False,
				)

				gr.Markdown(
					"""
					### Supported IEEE Systems

					| System | Buses | Branches | Loads | Generators | Use Case |
					|--------|-------|----------|-------|------------|----------|
					| **13-Bus** | 13 | 12 | 9 | 1 | Rapid prototyping, algorithm development |
					| **34-Bus** | 34 | 33 | 20 | 1 | Medium-scale validation with PV variants |
					| **123-Bus** | 123 | 122 | 85 | 1 | Large-scale scalability testing |

					### Action Space Detail

					| Device | Type | Range | Description |
					|--------|------|-------|-------------|
					| Capacitor | Discrete | {0, 1} | Switch on/off |
					| Regulator | Discrete | {0, ..., 16} | Tap position |
					| Battery | Continuous | [-1, 1] | Charge/discharge rate |
					| PV Inverter | Continuous | [0, 1] | Curtailment ratio |

					### Links

					[GitHub Repository](https://github.com/XJTU-RL/PowerZoo) ·
					[Documentation](https://xjtu-rl.github.io/PowerZoo/) ·
					IEEE Transactions on Smart Grid, 2025
					"""
				)

			# ================================================================
			# Tab 2: Voltage Profile
			# ================================================================
			with gr.Tab("Voltage Profile"):
				gr.Markdown(
					"""
					## IEEE 13-Bus Voltage Profile

					Explore bus voltage magnitudes across 24 hourly steps. Bars are colored by
					voltage status: **green** (normal, 0.95-1.05 pu), **yellow** (warning,
					0.93-0.95 or 1.05-1.07 pu), **red** (violation). During midday, PV injection
					raises upstream voltages; during evening peak, heavy load causes voltage sag on
					downstream buses.
					"""
				)

				step_dropdown = gr.Dropdown(
					choices=list(range(24)),
					value=12,
					label="Select Hour (0-23)",
				)
				voltage_plot = gr.Plot(value=plot_voltage_profile(12))

				step_dropdown.change(
					fn=plot_voltage_profile,
					inputs=step_dropdown,
					outputs=voltage_plot,
				)

			# ================================================================
			# Tab 3: Device Schedule
			# ================================================================
			with gr.Tab("Device Schedule"):
				gr.Markdown(
					"""
					## 24-Hour Device Operation Schedule

					Visualize how 6 agents coordinate device operations across a full day.
					- **Capacitors**: Discrete on/off switching to inject reactive power
					- **Regulators**: Tap adjustments (0-16) to regulate bus voltage
					- **Battery**: Charges from PV midday, discharges during evening peak
					- **PV Inverter**: Curtailment during overvoltage conditions (shaded area = curtailed)
					"""
				)

				device_plot = gr.Plot(value=plot_device_schedule())

			# ================================================================
			# Tab 4: Reward Analysis
			# ================================================================
			with gr.Tab("Reward Analysis"):
				gr.Markdown(
					"""
					## Episode Reward Decomposition

					The VVC reward function has three components, all negative (closer to zero is better):
					- **Power Loss** (blue): Penalizes distribution line losses
					- **Voltage Violation** (red): Penalizes buses outside ANSI voltage limits
					- **Control Penalty** (orange): Penalizes excessive device switching

					The stacked bars show per-step decomposition. The cyan line tracks
					cumulative reward across the episode.
					"""
				)

				reward_plot = gr.Plot(value=plot_reward_analysis())

			# ================================================================
			# Tab 5: Training Dashboard
			# ================================================================
			with gr.Tab("Training Dashboard"):
				gr.Markdown(
					"""
					## HAPPO Training on IEEE 13-Bus VVC

					Synthetic training curves demonstrating HAPPO algorithm convergence on the
					VVC environment (2000 episodes, 6 agents, MLP policy).
					"""
				)

				gr.Markdown("### Episode Reward Curve")
				training_reward_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### Per-Agent Policy Loss")
				agent_loss_plot = gr.Plot(value=plot_agent_policy_loss())

				gr.Markdown("### Power Loss Reduction")
				power_loss_plot = gr.Plot(value=plot_power_loss_reduction())

		# Footer
		gr.Markdown(
			"""
			---
			<p class="footer-text">
				PowerZoo · MIT License · XJTU-RL · IEEE TSG 2025
			</p>
			""",
		)

	return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(server_name="0.0.0.0", server_port=7860, share=False)
