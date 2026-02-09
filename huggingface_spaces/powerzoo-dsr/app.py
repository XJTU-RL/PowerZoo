"""
PowerZoo DSR: Distribution Service Restoration Demo
HuggingFace Spaces application with Gradio + Plotly.

5 Tabs: Overview | Restoration Progress | Network State | Action Mask & Agent Decisions | Training Dashboard

Self-contained demo -- all data generated inline, no external imports
beyond gradio, plotly, numpy, pandas.
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
	"primary": "#EF4444",
	"secondary": "#F97316",
	"accent": "#06B6D4",
	"success": "#10B981",
	"fault": "#DC2626",
	"energized": "#22C55E",
	"deenergized": "#6B7280",
	"priority_critical": "#EF4444",
	"priority_important": "#F97316",
	"priority_normal": "#3B82F6",
	"bg_dark": "#1a1a2e",
	"grid_dark": "#2d2d44",
	"text_light": "#e0e0e0",
}

N_STEPS = 15  # Episode length for DSR


# ============================================================
# Demo Data Generators
# ============================================================

def _generate_restoration_data(severity: str) -> dict:
	"""Generate restoration progress data for a given fault severity.

	Args:
		severity: One of 'mild', 'moderate', 'severe'.

	Returns:
		Dict with keys: steps, critical, important, normal, total_pct.
	"""
	np.random.seed({"mild": 10, "moderate": 20, "severe": 30}[severity])
	steps = list(range(N_STEPS))

	# Total loads per priority: critical=4, important=5, normal=6 -> 15 total
	n_critical, n_important, n_normal = 4, 5, 6
	total_loads = n_critical + n_important + n_normal

	# Initial damage depends on severity
	init_frac = {"mild": 0.55, "moderate": 0.30, "severe": 0.15}[severity]
	# Recovery speed
	speed = {"mild": 0.08, "moderate": 0.06, "severe": 0.04}[severity]

	critical = []
	important = []
	normal = []
	total_pct = []

	# Critical loads restored first, then important, then normal
	for t in steps:
		progress = min(1.0, init_frac + speed * t + 0.01 * np.random.randn())
		progress = np.clip(progress, 0.0, 1.0)

		# Critical restored faster (priority)
		c_frac = min(1.0, progress * 1.3 + 0.05 * np.random.randn())
		c_frac = np.clip(c_frac, 0.0, 1.0)
		c_count = round(c_frac * n_critical)

		# Important follows
		i_frac = min(1.0, progress * 1.1 + 0.04 * np.random.randn())
		i_frac = np.clip(i_frac, 0.0, 1.0)
		i_count = round(i_frac * n_important)

		# Normal last
		n_frac = min(1.0, progress * 0.9 + 0.03 * np.random.randn())
		n_frac = np.clip(n_frac, 0.0, 1.0)
		n_count = round(n_frac * n_normal)

		critical.append(c_count)
		important.append(i_count)
		normal.append(n_count)
		total_pct.append((c_count + i_count + n_count) / total_loads * 100)

	return {
		"steps": steps,
		"critical": critical,
		"important": important,
		"normal": normal,
		"total_pct": total_pct,
		"n_critical": n_critical,
		"n_important": n_important,
		"n_normal": n_normal,
	}


def _generate_13bus_topology() -> dict:
	"""Generate IEEE 13-bus system topology with coordinates and connections.

	Returns:
		Dict with bus_names, x, y, connections (list of (from_idx, to_idx)).
	"""
	bus_names = [
		"650", "632", "633", "634", "645", "646",
		"671", "680", "684", "611", "652", "692", "675",
	]
	# Layout coordinates (manually placed for readability)
	x = [0.0, 2.0, 3.5, 5.0, 3.5, 5.0,
		 4.0, 6.0, 3.0, 3.0, 1.5, 5.5, 7.0]
	y = [5.0, 5.0, 6.0, 6.0, 7.5, 7.5,
		 3.5, 3.5, 2.0, 0.5, 2.0, 2.0, 2.0]

	# Line connections (index pairs)
	connections = [
		(0, 1),   # 650-632
		(1, 2),   # 632-633
		(2, 3),   # 633-634
		(1, 4),   # 632-645
		(4, 5),   # 645-646
		(1, 6),   # 632-671
		(6, 7),   # 671-680
		(6, 8),   # 671-684
		(8, 9),   # 684-611
		(8, 10),  # 684-652
		(6, 11),  # 671-692
		(11, 12), # 692-675
	]

	# Tie switches (normally open, can be closed for restoration)
	tie_switches = [
		(7, 12),  # 680-675 (tie)
		(9, 10),  # 611-652 (tie)
	]

	return {
		"bus_names": bus_names,
		"x": x,
		"y": y,
		"connections": connections,
		"tie_switches": tie_switches,
	}


def _generate_network_states() -> list[dict]:
	"""Generate per-step bus/line status for the 13-bus restoration scenario.

	Returns:
		List of 15 dicts, each with bus_status and line_status arrays.
	"""
	topo = _generate_13bus_topology()
	n_bus = len(topo["bus_names"])
	n_lines = len(topo["connections"])
	n_ties = len(topo["tie_switches"])
	states = []

	# Fault on line 632-671 (index 5) and bus 671 area initially de-energized
	faulted_line = 5
	# Buses downstream of fault: 671(6), 680(7), 684(8), 611(9), 652(10), 692(11), 675(12)
	downstream_buses = {6, 7, 8, 9, 10, 11, 12}

	for t in range(N_STEPS):
		bus_status = ["energized"] * n_bus
		line_status = ["closed"] * n_lines
		tie_status = ["open"] * n_ties

		if t < 2:
			# Step 0-1: fault detected, downstream de-energized
			for b in downstream_buses:
				bus_status[b] = "deenergized"
			line_status[faulted_line] = "faulted"
		elif t < 4:
			# Step 2-3: fault isolated, tie switch 680-675 closed
			for b in downstream_buses:
				bus_status[b] = "deenergized"
			line_status[faulted_line] = "faulted"
			# Isolate: open lines adjacent to fault
			tie_status[0] = "closed"  # 680-675 tie closed
			# 675(12) and 692(11) get power from tie
			bus_status[12] = "energized"
			bus_status[11] = "energized"
		elif t < 7:
			# Step 4-6: progressive restoration via PV + load shedding
			line_status[faulted_line] = "faulted"
			tie_status[0] = "closed"
			restored = {12, 11, 7}
			for b in downstream_buses:
				if b in restored:
					bus_status[b] = "energized"
				else:
					bus_status[b] = "deenergized"
		elif t < 10:
			# Step 7-9: more buses restored
			line_status[faulted_line] = "faulted"
			tie_status[0] = "closed"
			tie_status[1] = "closed"  # 611-652 tie closed
			restored = {12, 11, 7, 8, 10}
			for b in downstream_buses:
				if b in restored:
					bus_status[b] = "energized"
				else:
					bus_status[b] = "deenergized"
		else:
			# Step 10-14: nearly full restoration
			line_status[faulted_line] = "faulted"
			tie_status[0] = "closed"
			tie_status[1] = "closed"
			restored = {12, 11, 7, 8, 10, 9}
			for b in downstream_buses:
				if b in restored:
					bus_status[b] = "energized"
				else:
					bus_status[b] = "deenergized"
			# Bus 6 (671) stays faulted area
			bus_status[6] = "energized" if t >= 12 else "deenergized"

		states.append({
			"bus_status": bus_status,
			"line_status": line_status,
			"tie_status": tie_status,
		})

	return states


def _generate_action_mask_data() -> dict:
	"""Generate action mask and agent decision data for 15 steps.

	Agents: 1 Switch (actions 0-4), 3 PV (actions 5-7), 4 Load (actions 8-11)
	Total 12 action indices.

	Returns:
		Dict with mask (15x12), selected (15x12), agent_labels, action_labels.
	"""
	np.random.seed(42)

	agent_labels = [
		"Switch-0", "Switch-1", "Switch-2", "Switch-3", "Switch-4",
		"PV-0", "PV-1", "PV-2",
		"Load-0", "Load-1", "Load-2", "Load-3",
	]
	n_actions = len(agent_labels)

	# mask: 1=available, 0=masked
	mask = np.ones((N_STEPS, n_actions), dtype=int)
	selected = np.zeros((N_STEPS, n_actions), dtype=int)

	for t in range(N_STEPS):
		# Switch actions: masked until isolation (step 2+)
		if t < 2:
			mask[t, 0:5] = 0  # all switch actions masked
		elif t < 4:
			mask[t, [0, 1]] = 1  # only first two switches available
			mask[t, [2, 3, 4]] = 0
		else:
			mask[t, 0:5] = 1  # all switches available

		# PV actions: always available after step 1
		if t < 1:
			mask[t, 5:8] = 0
		else:
			mask[t, 5:8] = 1

		# Load actions: masked until power is available at their bus
		if t < 3:
			mask[t, 8:12] = 0
		elif t < 6:
			mask[t, [8, 9]] = 1
			mask[t, [10, 11]] = 0
		else:
			mask[t, 8:12] = 1

		# Select actions from available ones
		available_idx = np.where(mask[t] == 1)[0]
		if len(available_idx) > 0:
			# Select ~40% of available actions
			n_select = max(1, len(available_idx) // 3)
			chosen = np.random.choice(available_idx, size=n_select, replace=False)
			selected[t, chosen] = 1

	return {
		"mask": mask,
		"selected": selected,
		"agent_labels": agent_labels,
	}


def _generate_agent_decisions() -> dict:
	"""Generate per-agent action timeline data.

	Returns:
		Dict with agent_names, step actions per agent.
	"""
	np.random.seed(55)
	agents = {
		"Switch Agent": {
			"actions": ["idle", "idle", "close_tie_1", "close_tie_1", "close_tie_2",
						 "reroute", "reroute", "close_tie_2", "monitor", "monitor",
						 "monitor", "monitor", "open_fault", "open_fault", "verify"],
			"color": COLORS["fault"],
		},
		"PV Agent 0": {
			"actions": ["off", "ramp_up", "ramp_up", "100%", "100%",
						"100%", "100%", "100%", "100%", "100%",
						"80%", "80%", "60%", "60%", "60%"],
			"color": COLORS["secondary"],
		},
		"PV Agent 1": {
			"actions": ["off", "off", "ramp_up", "ramp_up", "100%",
						"100%", "100%", "100%", "100%", "80%",
						"80%", "60%", "60%", "40%", "40%"],
			"color": "#FBBF24",
		},
		"PV Agent 2": {
			"actions": ["off", "off", "off", "ramp_up", "ramp_up",
						"100%", "100%", "80%", "80%", "80%",
						"60%", "60%", "40%", "40%", "40%"],
			"color": "#F59E0B",
		},
		"Load Agent 0 (Critical)": {
			"actions": ["shed", "shed", "shed", "restore", "restore",
						"restore", "full", "full", "full", "full",
						"full", "full", "full", "full", "full"],
			"color": COLORS["primary"],
		},
		"Load Agent 1 (Critical)": {
			"actions": ["shed", "shed", "shed", "shed", "restore",
						"restore", "full", "full", "full", "full",
						"full", "full", "full", "full", "full"],
			"color": "#F87171",
		},
		"Load Agent 2 (Important)": {
			"actions": ["shed", "shed", "shed", "shed", "shed",
						"shed", "restore", "restore", "full", "full",
						"full", "full", "full", "full", "full"],
			"color": COLORS["secondary"],
		},
		"Load Agent 3 (Normal)": {
			"actions": ["shed", "shed", "shed", "shed", "shed",
						"shed", "shed", "shed", "restore", "restore",
						"restore", "full", "full", "full", "full"],
			"color": COLORS["accent"],
		},
	}
	return agents


def _generate_training_data() -> dict:
	"""Generate DSR training curves over 1000 episodes.

	Returns:
		Dict with episodes, rewards, success_rate, avg_restoration_time.
	"""
	np.random.seed(77)
	n_episodes = 1000
	episodes = np.arange(n_episodes)

	# Episode rewards: starts around -25, improves to ~+15
	base_reward = -25 + 40 * (1 - np.exp(-episodes / 300))
	noise = np.random.randn(n_episodes) * 3
	rewards = base_reward + noise

	# Smoothed rewards (rolling window 50)
	kernel = np.ones(50) / 50
	rewards_smooth = np.convolve(rewards, kernel, mode="same")

	# Success rate: starts ~10%, climbs to ~92%
	base_success = 0.10 + 0.82 * (1 - np.exp(-episodes / 250))
	success_noise = np.random.randn(n_episodes) * 0.05
	success_rate = np.clip(base_success + success_noise, 0.0, 1.0)
	success_smooth = np.convolve(success_rate, kernel, mode="same") * 100

	# Average restoration time: starts ~14 steps, improves to ~6 steps
	base_time = 14 - 8 * (1 - np.exp(-episodes / 350))
	time_noise = np.random.randn(n_episodes) * 0.8
	avg_time = np.clip(base_time + time_noise, 2.0, 15.0)
	time_smooth = np.convolve(avg_time, kernel, mode="same")

	return {
		"episodes": episodes.tolist(),
		"rewards": rewards.tolist(),
		"rewards_smooth": rewards_smooth.tolist(),
		"success_rate": success_smooth.tolist(),
		"avg_time": avg_time.tolist(),
		"avg_time_smooth": time_smooth.tolist(),
	}


# Pre-generate all data at module load
TOPO = _generate_13bus_topology()
NETWORK_STATES = _generate_network_states()
ACTION_DATA = _generate_action_mask_data()
AGENT_DECISIONS = _generate_agent_decisions()
TRAINING_DATA = _generate_training_data()


# ============================================================
# Plot Factory Functions
# ============================================================

def _dark_layout(**kwargs) -> dict:
	"""Return common dark-theme layout kwargs."""
	base = dict(
		template="plotly_dark",
		paper_bgcolor="#1a1a2e",
		plot_bgcolor="#16213e",
		font=dict(color="#e0e0e0"),
		margin=dict(l=60, r=40, t=60, b=50),
	)
	base.update(kwargs)
	return base


def plot_restoration_progress(severity: str) -> go.Figure:
	"""Plot restoration progress with stacked area + total percentage line.

	Args:
		severity: 'Mild', 'Moderate', or 'Severe'.
	"""
	sev = severity.lower()
	data = _generate_restoration_data(sev)
	steps = data["steps"]

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# Stacked area: normal (bottom), important (middle), critical (top)
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["normal"],
			name=f"Priority 3 - Normal (max {data['n_normal']})",
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(59, 130, 246, 0.5)",
			fill="tozeroy",
			stackgroup="loads",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["important"],
			name=f"Priority 2 - Important (max {data['n_important']})",
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(249, 115, 22, 0.5)",
			fill="tonexty",
			stackgroup="loads",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["critical"],
			name=f"Priority 1 - Critical (max {data['n_critical']})",
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(239, 68, 68, 0.5)",
			fill="tonexty",
			stackgroup="loads",
		),
		secondary_y=False,
	)

	# Total restoration percentage on secondary y-axis
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["total_pct"],
			name="Total Restoration %",
			mode="lines+markers",
			line=dict(color=COLORS["success"], width=3),
			marker=dict(size=7, symbol="diamond"),
		),
		secondary_y=True,
	)

	# 100% target line
	fig.add_hline(
		y=100, line_dash="dash", line_color="rgba(255,255,255,0.4)",
		annotation_text="100% Target",
		annotation_font_color="rgba(255,255,255,0.6)",
		secondary_y=True,
	)

	fig.update_layout(
		**_dark_layout(
			height=520,
			title=f"Load Restoration Progress ({severity} Fault)",
			legend=dict(
				orientation="h", yanchor="bottom", y=1.02,
				xanchor="center", x=0.5, font=dict(size=11),
			),
			hovermode="x unified",
		),
	)
	fig.update_xaxes(title_text="Restoration Step", dtick=1)
	fig.update_yaxes(title_text="# Restored Loads", secondary_y=False, rangemode="tozero")
	fig.update_yaxes(title_text="Restoration %", secondary_y=True, range=[0, 110])

	return fig


def plot_network_state(step: int) -> go.Figure:
	"""Plot 13-bus network state at a given step.

	Args:
		step: Restoration step index (0-14).
	"""
	step = int(np.clip(step, 0, N_STEPS - 1))
	state = NETWORK_STATES[step]
	bus_status = state["bus_status"]
	line_status = state["line_status"]
	tie_status = state["tie_status"]

	fig = go.Figure()

	# Draw regular lines
	for idx, (i, j) in enumerate(TOPO["connections"]):
		ls = line_status[idx]
		if ls == "faulted":
			color = COLORS["fault"]
			width = 4
			dash = "solid"
		elif ls == "closed":
			color = COLORS["energized"] if bus_status[i] == "energized" and bus_status[j] == "energized" else COLORS["deenergized"]
			width = 2.5
			dash = "solid"
		else:
			color = COLORS["deenergized"]
			width = 1.5
			dash = "dash"

		fig.add_trace(go.Scatter(
			x=[TOPO["x"][i], TOPO["x"][j]],
			y=[TOPO["y"][i], TOPO["y"][j]],
			mode="lines",
			line=dict(color=color, width=width, dash=dash),
			showlegend=False,
			hoverinfo="skip",
		))

		# Fault marker on faulted line
		if ls == "faulted":
			mx = (TOPO["x"][i] + TOPO["x"][j]) / 2
			my = (TOPO["y"][i] + TOPO["y"][j]) / 2
			fig.add_trace(go.Scatter(
				x=[mx], y=[my],
				mode="markers+text",
				marker=dict(size=18, color=COLORS["fault"], symbol="x"),
				text=["FAULT"],
				textposition="top center",
				textfont=dict(color=COLORS["fault"], size=10, family="monospace"),
				showlegend=False,
				hoverinfo="text",
				hovertext=f"Line Fault: {TOPO['bus_names'][i]}-{TOPO['bus_names'][j]}",
			))

	# Draw tie switches
	for idx, (i, j) in enumerate(TOPO["tie_switches"]):
		ts = tie_status[idx]
		if ts == "closed":
			color = COLORS["accent"]
			width = 2.5
			dash = "dot"
		else:
			color = COLORS["deenergized"]
			width = 1.5
			dash = "dash"

		fig.add_trace(go.Scatter(
			x=[TOPO["x"][i], TOPO["x"][j]],
			y=[TOPO["y"][i], TOPO["y"][j]],
			mode="lines",
			line=dict(color=color, width=width, dash=dash),
			showlegend=False,
			hoverinfo="text",
			hovertext=f"Tie Switch: {TOPO['bus_names'][i]}-{TOPO['bus_names'][j]} ({ts})",
		))

	# Draw buses
	colors_map = {
		"energized": COLORS["energized"],
		"deenergized": COLORS["deenergized"],
		"faulted": COLORS["fault"],
	}
	for status_type, legend_name, symbol in [
		("energized", "Energized", "circle"),
		("deenergized", "De-energized", "circle"),
		("faulted", "Faulted", "circle"),
	]:
		indices = [i for i, s in enumerate(bus_status) if s == status_type]
		if not indices:
			continue
		fig.add_trace(go.Scatter(
			x=[TOPO["x"][i] for i in indices],
			y=[TOPO["y"][i] for i in indices],
			mode="markers+text",
			marker=dict(
				size=22,
				color=colors_map[status_type],
				line=dict(width=2, color="white"),
				symbol=symbol,
			),
			text=[TOPO["bus_names"][i] for i in indices],
			textposition="bottom center",
			textfont=dict(size=10, color="#e0e0e0"),
			name=legend_name,
			hovertext=[
				f"Bus {TOPO['bus_names'][i]}: {status_type}"
				for i in indices
			],
			hoverinfo="text",
		))

	# Count stats
	n_energized = sum(1 for s in bus_status if s == "energized")
	n_total = len(bus_status)
	n_ties_closed = sum(1 for s in tie_status if s == "closed")

	fig.update_layout(
		**_dark_layout(
			height=600,
			title=f"IEEE 13-Bus Network State (Step {step}) | "
				  f"Energized: {n_energized}/{n_total} | "
				  f"Tie Switches Closed: {n_ties_closed}/{len(tie_status)}",
			showlegend=True,
			legend=dict(
				orientation="h", yanchor="bottom", y=-0.12,
				xanchor="center", x=0.5,
			),
		),
	)
	fig.update_xaxes(
		showgrid=False, zeroline=False, showticklabels=False,
		range=[-0.8, 8.0],
	)
	fig.update_yaxes(
		showgrid=False, zeroline=False, showticklabels=False,
		scaleanchor="x", scaleratio=1,
		range=[-0.5, 8.5],
	)

	return fig


def plot_action_mask_heatmap() -> go.Figure:
	"""Plot action mask heatmap showing availability and selections over steps."""
	mask = ACTION_DATA["mask"]
	selected = ACTION_DATA["selected"]
	labels = ACTION_DATA["agent_labels"]

	# Build color matrix: 0=masked(gray), 1=available(green), 2=selected(blue)
	color_matrix = np.zeros_like(mask, dtype=float)
	color_matrix[mask == 0] = 0.0   # masked
	color_matrix[mask == 1] = 0.5   # available
	color_matrix[selected == 1] = 1.0  # selected

	# Custom colorscale: gray -> green -> blue
	colorscale = [
		[0.0, "#374151"],   # masked (dark gray)
		[0.25, "#374151"],
		[0.25, "#059669"],  # available (green)
		[0.75, "#059669"],
		[0.75, "#2563EB"],  # selected (blue)
		[1.0, "#2563EB"],
	]

	# Hover text
	hover_text = []
	for t in range(N_STEPS):
		row = []
		for a in range(len(labels)):
			if selected[t, a] == 1:
				status = "SELECTED"
			elif mask[t, a] == 1:
				status = "Available"
			else:
				status = "Masked"
			row.append(f"Step {t} | {labels[a]}<br>Status: {status}")
		hover_text.append(row)

	fig = go.Figure(data=go.Heatmap(
		z=color_matrix.T,
		x=list(range(N_STEPS)),
		y=labels,
		hovertext=np.array(hover_text).T.tolist(),
		hoverinfo="text",
		colorscale=colorscale,
		showscale=False,
		xgap=2,
		ygap=2,
	))

	# Add separators between agent groups
	for y_pos in [4.5, 7.5]:
		fig.add_hline(y=y_pos, line_color="rgba(255,255,255,0.3)", line_width=2)

	# Agent group annotations
	fig.add_annotation(
		x=-1.5, y=2, text="Switch", textangle=-90,
		showarrow=False, font=dict(color=COLORS["fault"], size=12),
		xref="x", yref="y",
	)
	fig.add_annotation(
		x=-1.5, y=6, text="PV", textangle=-90,
		showarrow=False, font=dict(color=COLORS["secondary"], size=12),
		xref="x", yref="y",
	)
	fig.add_annotation(
		x=-1.5, y=9.5, text="Load", textangle=-90,
		showarrow=False, font=dict(color=COLORS["accent"], size=12),
		xref="x", yref="y",
	)

	fig.update_layout(
		**_dark_layout(
			height=500,
			title="Action Mask & Selection Heatmap",
			xaxis_title="Restoration Step",
		),
	)
	fig.update_xaxes(dtick=1)

	return fig


def plot_agent_timeline() -> go.Figure:
	"""Plot agent decision timeline as a horizontal action chart."""
	agents = AGENT_DECISIONS

	fig = go.Figure()
	agent_names = list(agents.keys())

	for idx, (name, info) in enumerate(agents.items()):
		actions = info["actions"]
		color = info["color"]

		# Each action as a colored marker at (step, agent_idx)
		for t, action in enumerate(actions):
			fig.add_trace(go.Scatter(
				x=[t],
				y=[idx],
				mode="markers+text",
				marker=dict(
					size=16,
					color=color,
					opacity=0.85,
					line=dict(width=1, color="white"),
				),
				text=[action],
				textposition="top center",
				textfont=dict(size=7, color="#e0e0e0"),
				showlegend=False,
				hovertext=f"Step {t} | {name}: {action}",
				hoverinfo="text",
			))

	fig.update_layout(
		**_dark_layout(
			height=550,
			title="Agent Decision Timeline",
			xaxis_title="Restoration Step",
		),
	)
	fig.update_xaxes(dtick=1, range=[-0.5, N_STEPS - 0.5])
	fig.update_yaxes(
		tickvals=list(range(len(agent_names))),
		ticktext=agent_names,
		range=[-0.8, len(agent_names) - 0.2],
	)

	return fig


def plot_training_rewards() -> go.Figure:
	"""Plot episode reward curve with raw + smoothed."""
	episodes = TRAINING_DATA["episodes"]
	rewards = TRAINING_DATA["rewards"]
	smooth = TRAINING_DATA["rewards_smooth"]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=episodes, y=rewards,
		mode="lines",
		name="Raw Rewards",
		line=dict(color="rgba(239, 68, 68, 0.2)", width=1),
	))
	fig.add_trace(go.Scatter(
		x=episodes, y=smooth,
		mode="lines",
		name="Smoothed (window=50)",
		line=dict(color=COLORS["primary"], width=3),
	))
	fig.add_hline(
		y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)",
	)
	fig.update_layout(
		**_dark_layout(
			height=420,
			title="Episode Reward Curve (HAPPO on DSR 123-Bus)",
			xaxis_title="Episode",
			yaxis_title="Total Episode Reward",
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		),
	)
	return fig


def plot_success_rate() -> go.Figure:
	"""Plot restoration success rate over training."""
	episodes = TRAINING_DATA["episodes"]
	success = TRAINING_DATA["success_rate"]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=episodes, y=success,
		mode="lines",
		name="Success Rate",
		line=dict(color=COLORS["success"], width=3),
		fill="tozeroy",
		fillcolor="rgba(16, 185, 129, 0.15)",
	))
	fig.add_hline(
		y=90, line_dash="dash", line_color=COLORS["secondary"],
		annotation_text="90% threshold",
		annotation_font_color=COLORS["secondary"],
	)
	fig.update_layout(
		**_dark_layout(
			height=420,
			title="Restoration Success Rate (>90% Load Restored)",
			xaxis_title="Episode",
			yaxis_title="Success Rate (%)",
		),
	)
	fig.update_yaxes(range=[0, 105])
	return fig


def plot_restoration_time() -> go.Figure:
	"""Plot average restoration time over training."""
	episodes = TRAINING_DATA["episodes"]
	avg_time = TRAINING_DATA["avg_time"]
	smooth = TRAINING_DATA["avg_time_smooth"]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=episodes, y=avg_time,
		mode="lines",
		name="Raw",
		line=dict(color="rgba(6, 182, 212, 0.2)", width=1),
	))
	fig.add_trace(go.Scatter(
		x=episodes, y=smooth,
		mode="lines",
		name="Smoothed (window=50)",
		line=dict(color=COLORS["accent"], width=3),
	))
	fig.add_hline(
		y=N_STEPS, line_dash="dash", line_color="rgba(255,255,255,0.2)",
		annotation_text=f"Max ({N_STEPS} steps)",
		annotation_font_color="rgba(255,255,255,0.4)",
	)
	fig.update_layout(
		**_dark_layout(
			height=420,
			title="Average Steps to 90% Restoration",
			xaxis_title="Episode",
			yaxis_title="Steps",
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		),
	)
	fig.update_yaxes(range=[0, N_STEPS + 2])
	return fig


# ============================================================
# Build Gradio App
# ============================================================

def build_app() -> gr.Blocks:
	"""Construct the DSR demo Gradio app with 5 tabs."""
	with gr.Blocks(
		title="PowerZoo DSR: Distribution Service Restoration",
		theme=gr.themes.Soft(primary_hue="red"),
	) as app:
		# Header
		gr.Markdown(
			"""
			# PowerZoo DSR: Distribution Service Restoration
			**Heterogeneous MARL for Fault Recovery** | **Action Masks** | **Priority-Based Restoration**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# Tab 1: Overview
			# --------------------------------------------------------
			with gr.Tab("Overview"):
				gr.Markdown(
					"""
					## Distribution Service Restoration (DSR)

					DSR simulates **fault recovery in distribution networks**. When a line fault
					occurs, the system must isolate the faulted section and progressively restore
					power to de-energized loads using tie switches, distributed PV generation,
					and intelligent load shedding.

					### Scenario
					1. **Fault Detection** -- A line fault is detected and downstream buses lose power
					2. **Fault Isolation** -- The faulted line is isolated by opening adjacent switches
					3. **Service Restoration** -- Tie switches are closed, PV generators ramp up, and loads
					   are restored in priority order (critical > important > normal)

					---

					### Key Specifications

					| Property | Value |
					|---|---|
					| **Agent Types** | Heterogeneous: 1 Switch + 3 PV + 4 Load |
					| **Episode Length** | 15 steps (fault recovery event) |
					| **Action Space** | Discrete with action masks |
					| **Observation** | Bus voltages, line states, load status, PV output |
					| **Reward** | Restoration rate + voltage penalties + overload penalties |
					| **Action Masks** | Prevent invalid switching (topology constraints) |

					---

					### Supported IEEE Test Systems

					| System | Buses | Lines | Switches | Complexity |
					|---|---|---|---|---|
					| **13-Bus** | 13 | 12 + 2 ties | 5 | Prototyping |
					| **123-Bus** | 123 | 120 + 8 ties | 20 | Standard |
					| **8500-Node** | 8500 | ~8400 + ties | 50+ | Scalability |

					---

					### Heterogeneous Agent Design

					- **Switch Agent**: Controls tie switches to reroute power around faulted sections.
					  Actions are heavily masked based on network topology constraints.
					- **PV Agents**: Control distributed PV generation output (0-100%). Ramp up during
					  restoration to provide local power to de-energized zones.
					- **Load Agents**: Manage load shedding priorities. Critical loads (hospitals, etc.)
					  are restored first; normal loads last. Actions masked until bus is energized.

					### Algorithm Support
					DSR works with all 15 MARL algorithms in PowerZoo (HAPPO, MAPPO, HATRPO, etc.),
					with HAPPO recommended for its sequential update mechanism that handles
					heterogeneous agents naturally.
					"""
				)

			# --------------------------------------------------------
			# Tab 2: Restoration Progress
			# --------------------------------------------------------
			with gr.Tab("Restoration Progress"):
				gr.Markdown(
					"""
					## Load Restoration Progress

					Stacked area chart shows the number of restored loads by priority level.
					The green line tracks total restoration percentage.
					Adjust fault severity to see how it affects recovery speed.
					"""
				)
				severity_slider = gr.Radio(
					choices=["Mild", "Moderate", "Severe"],
					value="Moderate",
					label="Fault Severity",
				)
				restoration_plot = gr.Plot(
					value=plot_restoration_progress("Moderate"),
				)
				severity_slider.change(
					fn=plot_restoration_progress,
					inputs=severity_slider,
					outputs=restoration_plot,
				)

			# --------------------------------------------------------
			# Tab 3: Network State
			# --------------------------------------------------------
			with gr.Tab("Network State"):
				gr.Markdown(
					"""
					## IEEE 13-Bus Network Topology

					Visualize the network state at each restoration step.
					- **Green nodes**: Energized buses
					- **Gray nodes**: De-energized buses
					- **Red X**: Faulted line
					- **Dashed gray lines**: Open switches
					- **Dotted cyan lines**: Closed tie switches (for restoration)
					"""
				)
				step_slider = gr.Slider(
					minimum=0, maximum=N_STEPS - 1, step=1, value=0,
					label="Restoration Step",
				)
				network_plot = gr.Plot(
					value=plot_network_state(0),
				)
				step_slider.change(
					fn=plot_network_state,
					inputs=step_slider,
					outputs=network_plot,
				)

			# --------------------------------------------------------
			# Tab 4: Action Mask & Agent Decisions
			# --------------------------------------------------------
			with gr.Tab("Action Mask & Agent Decisions"):
				gr.Markdown(
					"""
					## Action Availability & Agent Decisions

					**Heatmap** shows action availability per step:
					- **Dark gray**: Masked (unavailable due to physical constraints)
					- **Green**: Available but not selected
					- **Blue**: Selected by the agent

					**Timeline** below shows the specific action each agent took at each step.
					"""
				)
				mask_plot = gr.Plot(value=plot_action_mask_heatmap())
				gr.Markdown("---")
				timeline_plot = gr.Plot(value=plot_agent_timeline())

			# --------------------------------------------------------
			# Tab 5: Training Dashboard
			# --------------------------------------------------------
			with gr.Tab("Training Dashboard"):
				gr.Markdown(
					"""
					## DSR Training Performance (HAPPO on 123-Bus)

					Training curves from a HAPPO agent trained on the DSR environment
					with the IEEE 123-Bus system over 1000 episodes.
					"""
				)
				reward_plot = gr.Plot(value=plot_training_rewards())

				with gr.Row():
					with gr.Column():
						success_plot = gr.Plot(value=plot_success_rate())
					with gr.Column():
						time_plot = gr.Plot(value=plot_restoration_time())

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
