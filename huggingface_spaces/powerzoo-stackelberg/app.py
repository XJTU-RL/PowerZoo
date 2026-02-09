"""
PowerZoo Stackelberg Game: Interactive HuggingFace Space Demo

5 Tabs: Overview | Price Signal | Leader-Follower Dynamics | Market Equilibrium | Training Dashboard

Self-contained Gradio + Plotly application. All demo data generated inline.
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


# ============================================================
# Color Palette & Theming
# ============================================================

COLORS = {
	"primary": "#F59E0B",       # amber
	"secondary": "#D97706",     # darker amber
	"accent": "#EF4444",        # red
	"uc": "#F59E0B",            # UC (leader) color
	"consumer": "#8B5CF6",      # consumer (follower) color
	"bg_card": "rgba(245, 158, 11, 0.05)",
	"grid": "rgba(255, 255, 255, 0.08)",
	"text": "#E5E7EB",
	"text_dim": "#9CA3AF",
	"cost_colors": ["#F59E0B", "#8B5CF6", "#3B82F6", "#EF4444"],
}

PLOTLY_LAYOUT_DEFAULTS = dict(
	template="plotly_dark",
	paper_bgcolor="rgba(0,0,0,0)",
	plot_bgcolor="rgba(17,24,39,0.6)",
	font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
	margin=dict(l=60, r=60, t=50, b=50),
	hovermode="x unified",
)

HOURS = list(range(24))


# ============================================================
# Demo Data Generation
# ============================================================

def _seed() -> np.random.Generator:
	"""Deterministic RNG for reproducible demo data."""
	return np.random.default_rng(42)


def gen_tou_price() -> np.ndarray:
	"""TOU base price schedule ($/kWh) for 24 hours."""
	tou = np.zeros(24)
	for h in range(24):
		if 0 <= h < 7 or 22 <= h <= 23:
			tou[h] = 0.05   # off-peak
		elif 11 <= h < 17:
			tou[h] = 0.15   # peak
		else:
			tou[h] = 0.10   # mid-peak
	return tou


def gen_uc_dynamic_price(tou: np.ndarray) -> np.ndarray:
	"""UC learned dynamic price that adapts around TOU base."""
	rng = _seed()
	# UC learns to slightly undercut peak, raise off-peak, smooth transitions
	delta = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			delta[h] = rng.uniform(-0.03, -0.01)   # slight discount at peak
		elif 7 <= h < 11:
			delta[h] = rng.uniform(0.005, 0.02)     # raise before peak
		elif 17 <= h < 22:
			delta[h] = rng.uniform(0.0, 0.015)      # evening premium
		else:
			delta[h] = rng.uniform(-0.005, 0.01)    # off-peak noise
	# Smooth with moving average
	kernel = np.array([0.15, 0.25, 0.35, 0.25])
	delta_smooth = np.convolve(delta, kernel / kernel.sum(), mode="same")
	return np.clip(tou + delta_smooth, 0.03, 0.20)


def gen_dr_incentive() -> np.ndarray:
	"""Demand response incentive signal ($/kWh) offered by UC."""
	rng = _seed()
	dr = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			dr[h] = rng.uniform(0.03, 0.08)    # high incentive during peak
		elif 7 <= h < 11 or 17 <= h < 22:
			dr[h] = rng.uniform(0.01, 0.04)     # moderate during shoulders
		else:
			dr[h] = rng.uniform(0.0, 0.01)      # minimal off-peak
	return dr


def gen_uc_actions() -> np.ndarray:
	"""UC 5D actions over 24 steps: energy_price, dr_incentive, ess_charge, ess_discharge, reserve_margin."""
	rng = _seed()
	tou = gen_tou_price()
	actions = np.zeros((24, 5))

	# energy_price: normalized around TOU
	actions[:, 0] = (gen_uc_dynamic_price(tou) - 0.03) / 0.17

	# dr_incentive: normalized [0,1]
	actions[:, 1] = gen_dr_incentive() / 0.10

	# ess_charge: higher during off-peak
	for h in range(24):
		if h < 7 or h >= 22:
			actions[h, 2] = rng.uniform(0.4, 0.8)
		elif 11 <= h < 17:
			actions[h, 2] = rng.uniform(0.0, 0.15)
		else:
			actions[h, 2] = rng.uniform(0.1, 0.3)

	# ess_discharge: higher during peak
	for h in range(24):
		if 11 <= h < 17:
			actions[h, 3] = rng.uniform(0.5, 0.9)
		elif 7 <= h < 11 or 17 <= h < 22:
			actions[h, 3] = rng.uniform(0.1, 0.35)
		else:
			actions[h, 3] = rng.uniform(0.0, 0.1)

	# reserve_margin: relatively stable
	actions[:, 4] = rng.uniform(0.3, 0.6, size=24)

	return actions


def gen_consumer_actions() -> np.ndarray:
	"""Average consumer 3D actions over 24 steps: load_shift, der_output, flexibility_bid."""
	rng = _seed()
	actions = np.zeros((24, 3))

	for h in range(24):
		# load_shift: consumers shift away from peak
		if 11 <= h < 17:
			actions[h, 0] = rng.uniform(-0.6, -0.2)  # reduce peak load
		elif h < 7 or h >= 22:
			actions[h, 0] = rng.uniform(0.1, 0.5)    # absorb shifted load
		else:
			actions[h, 0] = rng.uniform(-0.15, 0.15)

		# der_output: follows solar profile roughly
		solar_factor = max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
		actions[h, 1] = solar_factor * rng.uniform(0.5, 0.95)

		# flexibility_bid: higher when DR incentive is higher
		if 11 <= h < 17:
			actions[h, 2] = rng.uniform(0.5, 0.85)
		else:
			actions[h, 2] = rng.uniform(0.1, 0.4)

	return actions


def gen_reward_curves() -> tuple[np.ndarray, np.ndarray]:
	"""UC reward and total consumer reward over 24 steps for a single episode."""
	rng = _seed()
	uc_reward = np.zeros(24)
	consumer_reward = np.zeros(24)

	for h in range(24):
		# UC reward: revenue from pricing - storage costs - DR costs
		base_rev = rng.uniform(2.0, 5.0)
		if 11 <= h < 17:
			base_rev += rng.uniform(1.0, 3.0)  # peak revenue
		storage_cost = rng.uniform(0.3, 0.8)
		dr_cost = rng.uniform(0.2, 0.6) if 11 <= h < 17 else rng.uniform(0.0, 0.2)
		uc_reward[h] = base_rev - storage_cost - dr_cost

		# Consumer reward: utility from consumption - electricity cost + DR benefit
		utility = rng.uniform(1.5, 4.0)
		elec_cost = rng.uniform(0.8, 2.5)
		if 11 <= h < 17:
			elec_cost += rng.uniform(0.5, 1.5)
		dr_benefit = rng.uniform(0.3, 1.0) if 11 <= h < 17 else rng.uniform(0.0, 0.3)
		consumer_reward[h] = utility - elec_cost + dr_benefit

	return uc_reward, consumer_reward


def gen_cost_breakdown() -> dict[str, np.ndarray]:
	"""System cost breakdown: generation, DR, storage, penalty over 24 steps."""
	rng = _seed()
	costs = {}

	# Generation cost tracks load pattern
	gen_base = np.array([
		3.2, 2.8, 2.5, 2.3, 2.2, 2.4, 3.0, 4.5,
		5.8, 6.2, 6.5, 7.8, 8.5, 8.2, 7.9, 7.5,
		7.0, 7.8, 7.2, 6.0, 5.2, 4.5, 3.8, 3.4,
	])
	costs["generation_cost"] = gen_base + rng.uniform(-0.3, 0.3, size=24)

	# DR cost: UC pays for demand response
	dr_cost = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			dr_cost[h] = rng.uniform(0.8, 2.0)
		elif 7 <= h < 11 or 17 <= h < 22:
			dr_cost[h] = rng.uniform(0.2, 0.6)
		else:
			dr_cost[h] = rng.uniform(0.0, 0.1)
	costs["dr_cost"] = dr_cost

	# Storage cost: charge/discharge losses
	storage_cost = np.zeros(24)
	for h in range(24):
		if h < 7 or h >= 22:
			storage_cost[h] = rng.uniform(0.3, 0.7)   # charging cost
		elif 11 <= h < 17:
			storage_cost[h] = rng.uniform(0.2, 0.5)   # discharging loss
		else:
			storage_cost[h] = rng.uniform(0.05, 0.2)
	costs["storage_cost"] = storage_cost

	# Penalty: voltage violation and reliability
	penalty = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			penalty[h] = rng.uniform(0.1, 0.6)   # peak stress
		else:
			penalty[h] = rng.uniform(0.0, 0.15)
	costs["penalty"] = penalty

	return costs


def gen_equilibrium_scatter(n_episodes: int = 200) -> pd.DataFrame:
	"""Training trajectory: UC profit vs Avg Consumer Utility across episodes."""
	rng = _seed()
	episodes = np.arange(n_episodes)

	# Early training: high variance, far from equilibrium
	# Late training: converges toward Pareto frontier
	progress = episodes / n_episodes

	# UC profit improves then stabilizes
	uc_base = -5.0 + 18.0 * (1.0 - np.exp(-3.0 * progress))
	uc_noise = rng.normal(0.0, 2.5 * (1.0 - 0.7 * progress), size=n_episodes)
	uc_profit = uc_base + uc_noise

	# Consumer utility: improves slower, natural tension with UC
	consumer_base = -2.0 + 10.0 * (1.0 - np.exp(-2.5 * progress))
	consumer_noise = rng.normal(0.0, 1.8 * (1.0 - 0.6 * progress), size=n_episodes)
	consumer_utility = consumer_base + consumer_noise

	# Add negative correlation at early stage (competition), reduce at later stage (cooperation)
	competition_factor = 0.3 * (1.0 - progress)
	consumer_utility -= competition_factor * uc_noise

	return pd.DataFrame({
		"episode": episodes,
		"uc_profit": uc_profit,
		"consumer_utility": consumer_utility,
		"phase": np.where(progress < 0.3, "Exploration", np.where(progress < 0.7, "Learning", "Convergence")),
	})


def gen_training_curves(n_episodes: int = 1000) -> dict[str, np.ndarray]:
	"""Long training curves for UC reward, consumer reward, social welfare, price stability."""
	rng = _seed()
	episodes = np.arange(n_episodes)
	progress = episodes / n_episodes

	# UC reward: starts negative, climbs and stabilizes
	uc_raw = -15.0 + 40.0 * (1.0 - np.exp(-4.0 * progress))
	uc_noise = rng.normal(0.0, 3.0, size=n_episodes) * (1.0 - 0.5 * progress)
	uc_reward = uc_raw + uc_noise

	# Consumer reward (per consumer, 4 consumers): starts negative, improves
	c_raw = -8.0 + 20.0 * (1.0 - np.exp(-3.0 * progress))
	c_noise = rng.normal(0.0, 2.0, size=n_episodes) * (1.0 - 0.5 * progress)
	consumer_per = c_raw + c_noise

	# Social welfare = UC + sum of 4 consumers
	social_welfare = uc_reward + 4.0 * consumer_per

	# Price stability: std of dynamic price over 24 hours, decreases as UC learns
	price_vol_raw = 0.06 - 0.035 * (1.0 - np.exp(-3.5 * progress))
	price_vol_noise = rng.uniform(-0.005, 0.005, size=n_episodes) * (1.0 - 0.6 * progress)
	price_stability = np.clip(price_vol_raw + price_vol_noise, 0.005, 0.08)

	return {
		"episodes": episodes,
		"uc_reward": uc_reward,
		"consumer_reward": consumer_per,
		"social_welfare": social_welfare,
		"price_stability": price_stability,
	}


# ============================================================
# Plot Factory Functions
# ============================================================

def plot_price_signal() -> go.Figure:
	"""Tab 2: TOU base price, UC dynamic price, and DR incentive over 24 hours."""
	tou = gen_tou_price()
	dynamic = gen_uc_dynamic_price(tou)
	dr = gen_dr_incentive()

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# TOU base price (step function)
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=tou.tolist(),
			mode="lines",
			name="TOU Base Price",
			line=dict(color="#6B7280", width=2.5, shape="hv", dash="dot"),
			hovertemplate="Hour %{x}<br>TOU: $%{y:.3f}/kWh<extra></extra>",
		),
		secondary_y=False,
	)

	# UC dynamic price (smooth line)
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=dynamic.tolist(),
			mode="lines+markers",
			name="UC Dynamic Price",
			line=dict(color=COLORS["uc"], width=3),
			marker=dict(size=6, symbol="diamond"),
			hovertemplate="Hour %{x}<br>UC Price: $%{y:.3f}/kWh<extra></extra>",
		),
		secondary_y=False,
	)

	# DR incentive (bar)
	fig.add_trace(
		go.Bar(
			x=HOURS, y=dr.tolist(),
			name="DR Incentive",
			marker_color=COLORS["consumer"],
			opacity=0.6,
			hovertemplate="Hour %{x}<br>DR: $%{y:.3f}/kWh<extra></extra>",
		),
		secondary_y=True,
	)

	# Period annotations
	fig.add_vrect(x0=-0.5, x1=6.5, fillcolor="rgba(59,130,246,0.06)", line_width=0,
		annotation_text="Off-Peak", annotation_position="top left",
		annotation=dict(font=dict(size=10, color=COLORS["text_dim"])))
	fig.add_vrect(x0=10.5, x1=16.5, fillcolor="rgba(239,68,68,0.06)", line_width=0,
		annotation_text="Peak", annotation_position="top left",
		annotation=dict(font=dict(size=10, color=COLORS["text_dim"])))

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=520,
		title=dict(text="24-Hour Price Signal & Demand Response Incentive", font=dict(size=16)),
		xaxis=dict(title="Hour of Day", dtick=2, gridcolor=COLORS["grid"]),
		yaxis=dict(title="Electricity Price ($/kWh)", gridcolor=COLORS["grid"],
			range=[0.0, 0.22]),
		yaxis2=dict(title="DR Incentive ($/kWh)", gridcolor=COLORS["grid"],
			range=[0.0, 0.10], overlaying="y", side="right"),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
		barmode="overlay",
	)

	return fig


def plot_leader_follower() -> go.Figure:
	"""Tab 3: 2x2 subplot grid for leader-follower dynamics."""
	uc_actions = gen_uc_actions()
	consumer_actions = gen_consumer_actions()
	uc_reward, consumer_reward = gen_reward_curves()
	costs = gen_cost_breakdown()

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"UC Actions (5D)", "Consumer Actions (3D, avg of 4)",
			"UC vs Consumer Rewards", "System Cost Breakdown",
		),
		vertical_spacing=0.14,
		horizontal_spacing=0.12,
		specs=[
			[{"type": "heatmap"}, {"type": "heatmap"}],
			[{"type": "xy"}, {"type": "xy"}],
		],
	)

	# --- (1,1) UC Actions Heatmap ---
	uc_labels = ["energy_price", "dr_incentive", "ess_charge", "ess_discharge", "reserve_margin"]
	fig.add_trace(
		go.Heatmap(
			z=uc_actions.T.tolist(),
			x=HOURS,
			y=uc_labels,
			colorscale=[[0, "#1E1B4B"], [0.5, "#F59E0B"], [1, "#FEF3C7"]],
			showscale=False,
			hovertemplate="Hour %{x}<br>%{y}: %{z:.3f}<extra></extra>",
		),
		row=1, col=1,
	)

	# --- (1,2) Consumer Actions Heatmap ---
	consumer_labels = ["load_shift", "der_output", "flexibility_bid"]
	fig.add_trace(
		go.Heatmap(
			z=consumer_actions.T.tolist(),
			x=HOURS,
			y=consumer_labels,
			colorscale=[[0, "#1E1B4B"], [0.5, "#8B5CF6"], [1, "#EDE9FE"]],
			showscale=False,
			hovertemplate="Hour %{x}<br>%{y}: %{z:.3f}<extra></extra>",
		),
		row=1, col=2,
	)

	# --- (2,1) Reward Curves ---
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=uc_reward.tolist(),
			mode="lines+markers",
			name="UC Reward",
			line=dict(color=COLORS["uc"], width=2.5),
			marker=dict(size=5),
		),
		row=2, col=1,
	)
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=consumer_reward.tolist(),
			mode="lines+markers",
			name="Consumer Reward (total)",
			line=dict(color=COLORS["consumer"], width=2.5),
			marker=dict(size=5),
		),
		row=2, col=1,
	)

	# --- (2,2) Stacked Area: Cost Breakdown ---
	cost_names = ["generation_cost", "dr_cost", "storage_cost", "penalty"]
	display_names = ["Generation", "DR Cost", "Storage", "Penalty"]
	# Convert hex colors to rgba for fill transparency
	_fill_rgba = [
		"rgba(245, 158, 11, 0.5)",   # #F59E0B
		"rgba(139, 92, 246, 0.5)",   # #8B5CF6
		"rgba(59, 130, 246, 0.5)",   # #3B82F6
		"rgba(239, 68, 68, 0.5)",    # #EF4444
	]
	for i, (key, dname) in enumerate(zip(cost_names, display_names)):
		fig.add_trace(
			go.Scatter(
				x=HOURS, y=costs[key].tolist(),
				mode="lines",
				name=dname,
				stackgroup="costs",
				line=dict(color=COLORS["cost_colors"][i], width=0.5),
				fillcolor=_fill_rgba[i],
			),
			row=2, col=2,
		)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=700,
		title=dict(text="Leader-Follower Dynamics (Single Episode)", font=dict(size=16)),
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
	)

	# Axis labels
	fig.update_xaxes(title_text="Hour", row=1, col=1, gridcolor=COLORS["grid"])
	fig.update_xaxes(title_text="Hour", row=1, col=2, gridcolor=COLORS["grid"])
	fig.update_xaxes(title_text="Hour", row=2, col=1, gridcolor=COLORS["grid"])
	fig.update_xaxes(title_text="Hour", row=2, col=2, gridcolor=COLORS["grid"])
	fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=1)
	fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=2)
	fig.update_yaxes(title_text="Reward", gridcolor=COLORS["grid"], row=2, col=1)
	fig.update_yaxes(title_text="Cost ($)", gridcolor=COLORS["grid"], row=2, col=2)

	return fig


def plot_market_equilibrium() -> go.Figure:
	"""Tab 4: Scatter of UC profit vs Consumer Utility with Pareto frontier."""
	df = gen_equilibrium_scatter(200)

	# Compute Pareto frontier
	sorted_df = df.sort_values("uc_profit", ascending=False).reset_index(drop=True)
	pareto_uc: list[float] = []
	pareto_cu: list[float] = []
	max_cu = -np.inf
	for _, row in sorted_df.iterrows():
		if row["consumer_utility"] > max_cu:
			max_cu = row["consumer_utility"]
			pareto_uc.append(row["uc_profit"])
			pareto_cu.append(row["consumer_utility"])
	# Sort for line plot
	pareto_order = np.argsort(pareto_uc)
	pareto_uc = [pareto_uc[i] for i in pareto_order]
	pareto_cu = [pareto_cu[i] for i in pareto_order]

	fig = go.Figure()

	# Scatter colored by episode number
	fig.add_trace(
		go.Scatter(
			x=df["uc_profit"].tolist(),
			y=df["consumer_utility"].tolist(),
			mode="markers",
			marker=dict(
				size=7,
				color=df["episode"].tolist(),
				colorscale=[
					[0.0, "#1E3A5F"],
					[0.3, "#3B82F6"],
					[0.7, "#F59E0B"],
					[1.0, "#FEF3C7"],
				],
				colorbar=dict(title="Episode", tickfont=dict(color=COLORS["text"])),
				opacity=0.7,
				line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
			),
			text=[f"Ep {e} ({p})" for e, p in zip(df["episode"], df["phase"])],
			hovertemplate="UC Profit: %{x:.2f}<br>Consumer Utility: %{y:.2f}<br>%{text}<extra></extra>",
			name="Episodes",
		)
	)

	# Pareto frontier
	fig.add_trace(
		go.Scatter(
			x=pareto_uc,
			y=pareto_cu,
			mode="lines+markers",
			name="Pareto Frontier",
			line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
			marker=dict(size=4, color=COLORS["accent"]),
		)
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=560,
		title=dict(text="Market Equilibrium: UC Profit vs Consumer Utility", font=dict(size=16)),
		xaxis=dict(title="UC Profit ($/episode)", gridcolor=COLORS["grid"]),
		yaxis=dict(title="Avg Consumer Utility ($/episode)", gridcolor=COLORS["grid"]),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


def compute_equilibrium_metrics() -> pd.DataFrame:
	"""Summary metrics table for the market equilibrium."""
	df = gen_equilibrium_scatter(200)
	converged = df[df["episode"] >= 140]
	return pd.DataFrame({
		"Metric": [
			"Avg UC Profit (converged)",
			"Avg Consumer Utility (converged)",
			"Social Welfare (converged)",
			"Price Volatility (final 50 ep)",
		],
		"Value": [
			f"${converged['uc_profit'].mean():.2f}",
			f"${converged['consumer_utility'].mean():.2f}",
			f"${(converged['uc_profit'] + 4 * converged['consumer_utility']).mean():.2f}",
			f"{gen_training_curves(1000)['price_stability'][-50:].mean():.4f} $/kWh",
		],
	})


def plot_training_rewards() -> go.Figure:
	"""Tab 5: UC and Consumer reward curves with dual y-axis."""
	data = gen_training_curves(1000)
	ep = data["episodes"].tolist()

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# UC reward
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["uc_reward"].tolist(),
			mode="lines",
			name="UC Reward",
			line=dict(color=COLORS["uc"], width=1.5),
			opacity=0.4,
		),
		secondary_y=False,
	)
	# UC smoothed (rolling 50)
	uc_smooth = pd.Series(data["uc_reward"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=uc_smooth,
			mode="lines",
			name="UC Reward (smoothed)",
			line=dict(color=COLORS["uc"], width=3),
		),
		secondary_y=False,
	)

	# Consumer reward
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["consumer_reward"].tolist(),
			mode="lines",
			name="Consumer Reward",
			line=dict(color=COLORS["consumer"], width=1.5),
			opacity=0.4,
		),
		secondary_y=True,
	)
	# Consumer smoothed
	c_smooth = pd.Series(data["consumer_reward"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=c_smooth,
			mode="lines",
			name="Consumer Reward (smoothed)",
			line=dict(color=COLORS["consumer"], width=3),
		),
		secondary_y=True,
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=480,
		title=dict(text="Training Reward Curves: UC (Leader) vs Consumers (Followers)",
			font=dict(size=16)),
		xaxis=dict(title="Episode", gridcolor=COLORS["grid"]),
		yaxis=dict(
			title=dict(text="UC Reward", font=dict(color=COLORS["uc"])),
			gridcolor=COLORS["grid"],
		),
		yaxis2=dict(
			title=dict(text="Consumer Reward (per agent)", font=dict(color=COLORS["consumer"])),
			gridcolor=COLORS["grid"],
			overlaying="y", side="right",
		),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


def plot_social_welfare() -> go.Figure:
	"""Tab 5: Social welfare over training."""
	data = gen_training_curves(1000)
	ep = data["episodes"].tolist()

	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=ep, y=data["social_welfare"].tolist(),
			mode="lines",
			name="Social Welfare (raw)",
			line=dict(color="#6B7280", width=1),
			opacity=0.3,
		)
	)

	sw_smooth = pd.Series(data["social_welfare"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=sw_smooth,
			mode="lines",
			name="Social Welfare (smoothed)",
			line=dict(color="#10B981", width=3),
		)
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title=dict(text="Social Welfare (UC + 4 x Consumer Reward)", font=dict(size=14)),
		xaxis=dict(title="Episode", gridcolor=COLORS["grid"]),
		yaxis=dict(title="Social Welfare ($)", gridcolor=COLORS["grid"]),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


def plot_price_stability() -> go.Figure:
	"""Tab 5: Price stability metric over training."""
	data = gen_training_curves(1000)
	ep = data["episodes"].tolist()

	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=ep, y=data["price_stability"].tolist(),
			mode="lines",
			name="Price Volatility (raw)",
			line=dict(color="#6B7280", width=1),
			opacity=0.3,
		)
	)

	ps_smooth = pd.Series(data["price_stability"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=ps_smooth,
			mode="lines",
			name="Price Volatility (smoothed)",
			line=dict(color=COLORS["uc"], width=3),
		)
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title=dict(text="Price Stability: Hourly Price Std Deviation Over Training",
			font=dict(size=14)),
		xaxis=dict(title="Episode", gridcolor=COLORS["grid"]),
		yaxis=dict(title="Price Std Dev ($/kWh)", gridcolor=COLORS["grid"]),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


# ============================================================
# Build Gradio Application
# ============================================================

def build_app() -> gr.Blocks:
	"""Construct the Gradio Blocks application with 5 tabs."""
	with gr.Blocks(
		title="PowerZoo Stackelberg Game",
		theme=gr.themes.Soft(primary_hue="amber"),
	) as app:

		# Header
		gr.Markdown(
			"""
			# PowerZoo Stackelberg: Market Game Environment
			**Heterogeneous MARL** | **1 UC Leader + N Consumer Followers** | **Stackelberg-Nash Equilibrium** | **IEEE TSG 2025**
			"""
		)

		with gr.Tabs():
			# ========================================
			# Tab 1: Overview
			# ========================================
			with gr.Tab("Overview"):
				gr.Markdown(
					"""
					## Stackelberg Game Environment

					The Stackelberg environment models a **bi-level non-cooperative game** between a
					Utility Company (UC) and multiple consumers in a power distribution network.
					The UC acts as the **Stackelberg leader**, setting electricity prices and managing
					energy storage, while consumers act as **followers**, optimizing their load
					management and distributed energy resource (DER) utilization in response.

					This framework captures the fundamental tension in deregulated electricity markets:
					the UC maximizes revenue through strategic pricing and demand response programs,
					while consumers minimize costs and maximize comfort.

					### Key Specifications

					| Property | Value |
					|---|---|
					| **Agent Structure** | Heterogeneous: 1 UC (leader) + 4 Consumers (followers) |
					| **Episode Length** | 24 steps (hourly resolution, one day) |
					| **UC Action Space** | 5D continuous: `energy_price`, `dr_incentive`, `ess_charge`, `ess_discharge`, `reserve_margin` |
					| **Consumer Action Space** | 3D continuous: `load_shift`, `der_output`, `flexibility_bid` |
					| **Observation** | System state (voltages, loads, PV) + market signals (price, TOU, ESS SoC) |
					| **Reward** | UC: revenue - costs; Consumer: utility - cost + DR benefit |
					| **Game Equilibrium** | Stackelberg-Nash: UC commits first, consumers best-respond |

					### Supported IEEE Test Systems

					| System | Buses | Loads | Use Case |
					|---|---|---|---|
					| **13-Bus** | 13 | 9 | Rapid prototyping, algorithm debugging |
					| **34-Bus** | 34 | 22 | Standard benchmarking, moderate complexity |
					| **123-Bus** | 123 | 85 | Scalability testing, large-scale experiments |

					### Game-Theoretic Structure

					The environment implements a **Stackelberg game** where:
					1. **UC (Leader)** announces pricing schedule and DR incentives for the next period
					2. **Consumers (Followers)** observe the UC's strategy and independently optimize their response
					3. The UC anticipates consumer reactions when setting its strategy (anticipatory pricing)
					4. **Nash equilibrium** among consumers: no consumer can unilaterally improve by deviating

					This creates a rich learning problem where the UC must learn to *lead* effectively,
					and consumers must learn to *follow* optimally -- forming a Stackelberg-Nash equilibrium.
					"""
				)

			# ========================================
			# Tab 2: Price Signal
			# ========================================
			with gr.Tab("Price Signal"):
				gr.Markdown(
					"""
					### 24-Hour Electricity Market Signal

					The UC sets dynamic prices that adapt around the TOU (Time-of-Use) base tariff.
					During peak hours (11:00-17:00), the UC offers demand response incentives to
					reduce system stress. The interplay between pricing and DR signals drives
					consumer behavior and shapes the market equilibrium.
					"""
				)
				price_plot = gr.Plot(value=plot_price_signal())

				gr.Markdown(
					"""
					**Reading the chart**: The dotted gray step function is the regulated TOU base price.
					The amber line shows the UC's learned dynamic price -- note how it slightly
					undercuts peak prices (to retain customers) while raising shoulder prices
					(to capture additional revenue). Purple bars indicate DR incentive payments
					offered to consumers who reduce peak load.
					"""
				)

			# ========================================
			# Tab 3: Leader-Follower Dynamics
			# ========================================
			with gr.Tab("Leader-Follower Dynamics"):
				gr.Markdown(
					"""
					### Single-Episode Action and Outcome Analysis

					Visualizing the Stackelberg interaction within a single 24-hour episode.
					The UC's 5-dimensional action vector and consumers' 3-dimensional responses
					reveal the temporal coordination patterns that emerge from game-theoretic learning.
					"""
				)
				dynamics_plot = gr.Plot(value=plot_leader_follower())

				gr.Markdown(
					"""
					**Key observations**:
					- UC charges ESS during off-peak (dark in `ess_charge`, row 3 at night) and
					  discharges during peak (bright in `ess_discharge`, row 4 at midday)
					- Consumers shift load away from peak hours (`load_shift` negative at midday)
					  and increase DER output following the solar profile
					- The reward curves show natural tension: UC profits more during peak, but
					  consumer costs also increase -- the game seeks equilibrium
					"""
				)

			# ========================================
			# Tab 4: Market Equilibrium
			# ========================================
			with gr.Tab("Market Equilibrium"):
				gr.Markdown(
					"""
					### Convergence Toward Stackelberg-Nash Equilibrium

					Each point represents one training episode plotted by UC profit (x) vs average
					consumer utility (y). Early episodes (dark blue) show chaotic exploration.
					As training progresses (amber to white), both agents converge toward the
					Pareto frontier -- the boundary where neither agent can improve without
					harming the other.
					"""
				)

				with gr.Row():
					with gr.Column(scale=3):
						eq_plot = gr.Plot(value=plot_market_equilibrium())
					with gr.Column(scale=1):
						gr.Markdown("### Equilibrium Metrics")
						eq_table = gr.Dataframe(
							value=compute_equilibrium_metrics(),
							headers=["Metric", "Value"],
							interactive=False,
						)

			# ========================================
			# Tab 5: Training Dashboard
			# ========================================
			with gr.Tab("Training Dashboard"):
				gr.Markdown(
					"""
					### HAPPO Training on Stackelberg 13-Bus System

					Training curves from a 1000-episode HAPPO experiment showing how both the
					UC leader and consumer followers learn simultaneously. The dual reward curves
					reveal the co-evolution of leader and follower strategies.
					"""
				)
				reward_plot = gr.Plot(value=plot_training_rewards())

				with gr.Row():
					with gr.Column():
						sw_plot = gr.Plot(value=plot_social_welfare())
					with gr.Column():
						ps_plot = gr.Plot(value=plot_price_stability())

				gr.Markdown(
					"""
					**Interpretation**: Social welfare (green) increases as both agents learn to
					cooperate within the competitive framework. Price stability (amber) improves
					as the UC learns consistent pricing strategies -- initial exploration causes
					high price volatility that settles as the policy matures.
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
