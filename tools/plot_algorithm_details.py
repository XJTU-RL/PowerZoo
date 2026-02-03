"""
PowerZoo Individual Algorithm Architecture Diagrams.

Generates internal architecture / data-flow diagrams for each of the
7 algorithm families in PowerZoo using Plotly shapes + annotations.

Output:
  - HTML files (CDN plotly.js) for GitHub Pages embedding
  - JSON files for HuggingFace Space gr.Plot()
  - PNG static images

Usage:
	python tools/plot_algorithm_details.py
"""
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# ── Shared palette ──────────────────────────────────────────────
C = {
	"obs": "#E3F2FD",        # light blue  – observations
	"policy": "#42A5F5",     # blue        – policy / actor
	"critic": "#EF5350",     # red         – critic / value
	"action": "#66BB6A",     # green       – actions
	"loss": "#FFA726",       # orange      – loss / update
	"module": "#AB47BC",     # purple      – special modules
	"buffer": "#FFEE58",     # yellow      – replay buffer
	"env": "#26A69A",        # teal        – environment
	"target": "#BDBDBD",     # grey        – target network
	"text": "#212121",       # dark
	"arrow": "#546E7A",      # blue-grey
	"bg": "#FFFFFF",
	"border": "#78909C",
}


# ── Helper: build a block-diagram figure ─────────────────────────
def _box(fig: go.Figure, x0: float, y0: float, w: float, h: float,
         fill: str, label: str, sub: str = "", bold: bool = True,
         border: str = C["border"], opacity: float = 1.0) -> None:
	"""Draw a rounded rectangle with centred text."""
	fig.add_shape(
		type="rect", x0=x0, y0=y0, x1=x0 + w, y1=y0 + h,
		fillcolor=fill, opacity=opacity,
		line=dict(color=border, width=1.5),
		layer="below",
	)
	txt = f"<b>{label}</b>" if bold else label
	if sub:
		txt += f"<br><span style='font-size:10px;color:#555'>{sub}</span>"
	fig.add_annotation(
		x=x0 + w / 2, y=y0 + h / 2, text=txt,
		showarrow=False, font=dict(size=12, color=C["text"]),
		align="center",
	)


def _arrow(fig: go.Figure, x0: float, y0: float, x1: float, y1: float,
           label: str = "", color: str = C["arrow"], dash: str = "solid") -> None:
	"""Draw an arrow between two points."""
	fig.add_annotation(
		x=x1, y=y1, ax=x0, ay=y0,
		xref="x", yref="y", axref="x", ayref="y",
		showarrow=True,
		arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
		arrowcolor=color,
		standoff=4, startstandoff=4,
	)
	if label:
		mx, my = (x0 + x1) / 2, (y0 + y1) / 2
		fig.add_annotation(
			x=mx, y=my, text=f"<i>{label}</i>",
			showarrow=False, font=dict(size=9, color="#666"),
			bgcolor="rgba(255,255,255,0.85)", borderpad=2,
		)


def _dashed_arrow(fig, x0, y0, x1, y1, label="", color=C["target"]):
	"""Dashed arrow for target network / soft update."""
	# Use a shape line since annotation arrows don't support dash
	fig.add_shape(
		type="line", x0=x0, y0=y0, x1=x1, y1=y1,
		line=dict(color=color, width=1.5, dash="dot"),
	)
	if label:
		mx, my = (x0 + x1) / 2, (y0 + y1) / 2
		fig.add_annotation(
			x=mx, y=my, text=f"<i>{label}</i>",
			showarrow=False, font=dict(size=9, color="#999"),
			bgcolor="rgba(255,255,255,0.85)", borderpad=2,
		)


def _base_fig(title: str, width: int = 900, height: int = 520,
              xrange: tuple = (-0.5, 10.5), yrange: tuple = (-0.5, 7)) -> go.Figure:
	"""Create a blank figure with invisible axes."""
	fig = go.Figure()
	fig.update_layout(
		title=dict(text=title, font=dict(size=16, color=C["text"]), x=0.5),
		width=width, height=height,
		xaxis=dict(range=xrange, visible=False, fixedrange=True),
		yaxis=dict(range=yrange, visible=False, fixedrange=True, scaleanchor="x"),
		plot_bgcolor=C["bg"],
		paper_bgcolor=C["bg"],
		margin=dict(l=20, r=20, t=50, b=20),
		showlegend=False,
	)
	# Invisible trace so Plotly renders the axes range properly
	fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", hoverinfo="skip"))
	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 1: HAPPO Family (HAPPO / HATRPO / HAA2C)
# ══════════════════════════════════════════════════════════════════
def create_happo_family() -> go.Figure:
	fig = _base_fig("HAPPO Family: On-Policy Heterogeneous Actor-Critic",
	                height=560, yrange=(-0.5, 8))

	# ── Environment
	_box(fig, 0, 6.5, 2, 0.9, C["env"], "Environment", "OpenDSS / Gym")

	# ── Per-Agent obs
	_box(fig, 3, 6.5, 1.8, 0.9, C["obs"], "Obs Agent 1")
	_box(fig, 5.2, 6.5, 1.8, 0.9, C["obs"], "Obs Agent 2")
	_box(fig, 7.4, 6.5, 1.8, 0.9, C["obs"], "Obs Agent N")
	_arrow(fig, 2, 6.95, 3, 6.95, "obs")

	# ── Per-Agent Actor (StochasticPolicy)
	_box(fig, 3, 4.8, 1.8, 1.2, C["policy"], "Actor 1", "StochasticPolicy<br>MLPBase→RNN→ACT")
	_box(fig, 5.2, 4.8, 1.8, 1.2, C["policy"], "Actor 2", "StochasticPolicy<br>(separate params)")
	_box(fig, 7.4, 4.8, 1.8, 1.2, C["policy"], "Actor N", "StochasticPolicy<br>(separate params)")
	_arrow(fig, 3.9, 6.5, 3.9, 6.0)
	_arrow(fig, 6.1, 6.5, 6.1, 6.0)
	_arrow(fig, 8.3, 6.5, 8.3, 6.0)

	# ── Actions
	_box(fig, 3, 3.3, 1.8, 0.9, C["action"], "Action 1", "a₁ ~ π₁(·|o₁)")
	_box(fig, 5.2, 3.3, 1.8, 0.9, C["action"], "Action 2", "a₂ ~ π₂(·|o₂)")
	_box(fig, 7.4, 3.3, 1.8, 0.9, C["action"], "Action N", "aₙ ~ πₙ(·|oₙ)")
	_arrow(fig, 3.9, 4.8, 3.9, 4.2)
	_arrow(fig, 6.1, 4.8, 6.1, 4.2)
	_arrow(fig, 8.3, 4.8, 8.3, 4.2)

	# ── Centralized Critic (VCritic)
	_box(fig, 0, 3.3, 2.5, 1.2, C["critic"], "VCritic", "V(s) — centralized<br>global state input")

	# ── Sequential Update block
	_box(fig, 0.5, 1.2, 8.5, 1.5, C["loss"], "Sequential Update (HAPPO)",
	     "Agent 1 → compute factor → Agent 2 → … → Agent N<br>"
	     "M̂ᵢ = min(rᵢ·M̂ᵢ₋₁, clip(rᵢ)·M̂ᵢ₋₁)  |  HATRPO: KL constraint  |  HAA2C: no clipping",
	     opacity=0.85)

	# ── Arrows from actions/critic to update
	_arrow(fig, 1.25, 3.3, 1.25, 2.7, "advantage")
	_arrow(fig, 3.9, 3.3, 3.9, 2.7, "log π, action")
	_arrow(fig, 6.1, 3.3, 6.1, 2.7)
	_arrow(fig, 8.3, 3.3, 8.3, 2.7)

	# ── GAE label
	_box(fig, 0, 0, 3.5, 0.8, "#E8EAF6", "GAE(γ, λ)",
	     "Generalized Advantage Estimation")
	_arrow(fig, 1.25, 1.2, 1.25, 0.8, "returns")

	# ── Algorithm variant labels
	fig.add_annotation(x=9.5, y=1.95, text=(
		"<b>Variants:</b><br>"
		"<span style='color:#42A5F5'>HAPPO</span>: PPO clip + sequential factor<br>"
		"<span style='color:#42A5F5'>HATRPO</span>: conjugate gradient + KL<br>"
		"<span style='color:#42A5F5'>HAA2C</span>: A2C loss (no clipping)"
	), showarrow=False, font=dict(size=10), align="left",
		bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1, borderpad=6)

	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 2: MAPPO / SN-MAPPO
# ══════════════════════════════════════════════════════════════════
def create_mappo_family() -> go.Figure:
	fig = _base_fig("MAPPO / SN-MAPPO: Centralized Critic with Parameter Sharing",
	                height=560, yrange=(-0.5, 8))

	# ── Environment
	_box(fig, 0, 6.5, 2, 0.9, C["env"], "Environment")

	# ── Global state for critic
	_box(fig, 0, 4.5, 2.2, 1, C["obs"], "Global State", "s = concat(o₁,...,oₙ)")
	_arrow(fig, 1, 6.5, 1, 5.5)

	# ── Shared / Per-agent observations
	_box(fig, 3.5, 6.5, 5.5, 0.9, C["obs"], "Per-Agent Observations",
	     "o₁, o₂, ..., oₙ  (identical obs space)")
	_arrow(fig, 2, 6.95, 3.5, 6.95, "obs")

	# ── Shared Actor (MAPPO)
	_box(fig, 3.5, 4.5, 2.5, 1.2, C["policy"], "Shared Actor",
	     "StochasticPolicy<br>share_param=True<br>same weights ∀ agents")
	_arrow(fig, 6.25, 6.5, 4.75, 5.7, "oᵢ")

	# ── OR separate actors (SN-MAPPO)
	_box(fig, 6.5, 4.5, 3, 1.2, C["module"], "SN-MAPPO",
	     "Leader Actor (lr_leader)<br>Follower Actors (lr_follower)<br>Stackelberg-Nash update")
	_arrow(fig, 6.25, 6.5, 8, 5.7, "oᵢ")

	# ── Centralized Critic
	_box(fig, 0, 2.8, 2.2, 1, C["critic"], "VCritic", "V(s) centralized")
	_arrow(fig, 1.1, 4.5, 1.1, 3.8)

	# ── Actions
	_box(fig, 3.5, 2.8, 2.5, 1, C["action"], "Actions",
	     "a₁=a₂=…=aₙ ~ π_shared")
	_arrow(fig, 4.75, 4.5, 4.75, 3.8)

	# ── PPO Update
	_box(fig, 0.5, 1, 5, 1.2, C["loss"], "PPO Clipped Update",
	     "L = min(r·Â, clip(r,1±ε)·Â) − c₁·V_loss + c₂·H(π)<br>"
	     "Single shared gradient for all agents")
	_arrow(fig, 1.1, 2.8, 1.5, 2.2, "Â")
	_arrow(fig, 4.75, 2.8, 4, 2.2, "log π")

	# ── SN-MAPPO update
	_box(fig, 6, 1, 3.5, 1.2, C["loss"], "SN-MAPPO Update",
	     "Leader: PPO + best_response_loss<br>"
	     "Followers: PPO with follower LR")
	_arrow(fig, 8, 4.5, 8, 2.2)

	# ── Key differences
	fig.add_annotation(x=5, y=0.2, text=(
		"<b>MAPPO</b>: all agents share one policy network  |  "
		"<b>SN-MAPPO</b>: leader-follower hierarchy with separate LR scales"
	), showarrow=False, font=dict(size=10, color="#555"), align="center")

	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 3: DAN-HAPPO
# ══════════════════════════════════════════════════════════════════
def create_dan_happo() -> go.Figure:
	fig = _base_fig("DAN-HAPPO: Dynamic Attention Network + HAPPO",
	                height=560, yrange=(-0.5, 8))

	# ── Env + Obs
	_box(fig, 0, 6.5, 2, 0.9, C["env"], "Environment")
	_box(fig, 3, 6.5, 2, 0.9, C["obs"], "Agent i Obs", "oᵢ")
	_box(fig, 5.5, 6.5, 3.5, 0.9, C["obs"], "Neighbor Obs",
	     "o_neighbors = {oⱼ : j ∈ N(i)}")
	_arrow(fig, 2, 6.95, 3, 6.95)

	# ── DAN Module
	_box(fig, 5.5, 4.5, 3.5, 1.4, C["module"], "DAN Module",
	     "Dynamic Attention Network<br>"
	     "Self-Attention over neighbor obs<br>"
	     "→ aggregated neighborhood embedding")
	_arrow(fig, 7.25, 6.5, 7.25, 5.9)

	# ── Concatenation
	_box(fig, 3, 4.5, 2, 0.9, C["obs"], "Concat",
	     "oᵢ ⊕ DAN(neighbors)")
	_arrow(fig, 4, 6.5, 4, 5.4)
	_arrow(fig, 5.5, 5.2, 5, 5.0, "embed")

	# ── Actor
	_box(fig, 3, 2.8, 2, 1.2, C["policy"], "Actor i",
	     "StochasticPolicy<br>augmented input")
	_arrow(fig, 4, 4.5, 4, 4.0)

	# ── Action
	_box(fig, 3, 1.3, 2, 0.9, C["action"], "Action aᵢ", "aᵢ ~ πᵢ(·|oᵢ, DAN)")
	_arrow(fig, 4, 2.8, 4, 2.2)

	# ── Critic + HAPPO update
	_box(fig, 0, 2.8, 2.5, 1.2, C["critic"], "VCritic", "V(s) centralized")
	_box(fig, 0.3, 0.2, 8.5, 0.8, C["loss"], "HAPPO Sequential Update",
	     "Same as HAPPO + separate DAN optimizer for attention weights")

	_arrow(fig, 1.25, 2.8, 1.25, 1.0, "Â")
	_arrow(fig, 4, 1.3, 4, 1.0, "log π")

	# ── Key feature
	fig.add_annotation(x=8.5, y=3.5, text=(
		"<b>Key:</b><br>"
		"DAN enables agent-to-agent<br>"
		"communication via learned<br>"
		"attention over neighbors"
	), showarrow=False, font=dict(size=10), align="left",
		bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1, borderpad=6)

	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 4: DDPG Family (HADDPG / HATD3 / MADDPG / MATD3)
# ══════════════════════════════════════════════════════════════════
def create_ddpg_family() -> go.Figure:
	fig = _base_fig("DDPG Family: Off-Policy Deterministic Actor-Critic",
	                height=580, yrange=(-0.5, 8.5))

	# ── Env
	_box(fig, 0, 7, 2, 0.9, C["env"], "Environment")

	# ── Replay Buffer
	_box(fig, 0, 5.2, 2.5, 1.2, C["buffer"], "Replay Buffer",
	     "(s, a, r, s', done)<br>uniform / PER sampling")
	_arrow(fig, 1, 7, 1, 6.4, "transition")

	# ── Actor (Deterministic)
	_box(fig, 3.5, 6.5, 2.5, 1.2, C["policy"], "Actor μ",
	     "DeterministicPolicy<br>PlainMLP<br>a = μ(o) + noise")
	_arrow(fig, 2, 7.5, 3.5, 7.1, "obs")

	# ── Target Actor
	_box(fig, 3.5, 4.8, 2.5, 1, C["target"], "Target Actor μ'",
	     "Soft update: τ·μ + (1-τ)·μ'")
	_dashed_arrow(fig, 4.75, 6.5, 4.75, 5.8, "soft update τ")

	# ── Critic Q(s,a)
	_box(fig, 7, 6.5, 2.5, 1.2, C["critic"], "Critic Q",
	     "ContinuousQCritic<br>Q(s, a₁, ..., aₙ)")

	# ── Target Critic
	_box(fig, 7, 4.8, 2.5, 1, C["target"], "Target Critic Q'",
	     "Soft update: τ·Q + (1-τ)·Q'")
	_dashed_arrow(fig, 8.25, 6.5, 8.25, 5.8, "soft update τ")

	# ── TD3 twin Q extension
	_box(fig, 7, 3.2, 2.5, 1.2, C["critic"], "Twin Q (TD3)",
	     "Q₁, Q₂ — take min<br>Target smoothing noise<br>Delayed policy update")

	# ── Actor Loss
	_box(fig, 3, 2.2, 3, 1.2, C["loss"], "Actor Loss",
	     "L_actor = −Q(s, μ(o))<br>Maximize Q via policy gradient")
	_arrow(fig, 4.75, 4.8, 4.5, 3.4, "a'=μ'(o')")
	_arrow(fig, 7, 7.1, 6, 7.1)
	_arrow(fig, 7, 3.8, 6, 3.0, "∂Q/∂a")

	# ── Critic Loss
	_box(fig, 7, 1.5, 2.5, 1.2, C["loss"], "Critic Loss",
	     "L_critic = (Q − y)²<br>y = r + γ·Q'(s', μ'(s'))")
	_arrow(fig, 8.25, 3.2, 8.25, 2.7, "Q values")

	# ── Variant labels
	fig.add_annotation(x=1.5, y=0.3, text=(
		"<b>Variants:</b><br>"
		"<span style='color:#42A5F5'>HADDPG</span>: per-agent actor, centralized Q  |  "
		"<span style='color:#42A5F5'>HATD3</span>: + twin Q + delayed update<br>"
		"<span style='color:#42A5F5'>MADDPG</span>: shared actor structure  |  "
		"<span style='color:#42A5F5'>MATD3</span>: MADDPG + TD3 tricks"
	), showarrow=False, font=dict(size=10), align="left",
		xanchor="left",
		bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1, borderpad=6)

	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 5: HASAC
# ══════════════════════════════════════════════════════════════════
def create_hasac() -> go.Figure:
	fig = _base_fig("HASAC: Maximum Entropy RL with Auto Temperature",
	                height=560, yrange=(-0.5, 8))

	# ── Env + Buffer
	_box(fig, 0, 6.5, 2, 0.9, C["env"], "Environment")
	_box(fig, 0, 4.5, 2.5, 1.2, C["buffer"], "Replay Buffer",
	     "Off-policy transitions<br>(s, a, r, s', done)")
	_arrow(fig, 1, 6.5, 1, 5.7)

	# ── Stochastic Actor (SquashedGaussian)
	_box(fig, 3.5, 6, 2.8, 1.5, C["policy"], "Actor π",
	     "SquashedGaussianPolicy<br>μ, σ = f(o)<br>a = tanh(μ + σ·ε)<br>ε ~ N(0,1)")
	_arrow(fig, 2, 6.95, 3.5, 6.75, "obs")

	# ── Twin Q Critics
	_box(fig, 7, 6, 2.5, 1.5, C["critic"], "Twin Q Critics",
	     "SoftTwinContinuousQ<br>Q₁(s,a), Q₂(s,a)<br>use min(Q₁, Q₂)")

	# ── Auto-alpha
	_box(fig, 7, 4, 2.5, 1.2, C["module"], "Auto α (Temperature)",
	     "α = exp(log_alpha)<br>Target entropy = −dim(A)<br>L_α = −α·(log π + H̄)")

	# ── Target Q
	_box(fig, 7, 2.2, 2.5, 1.2, C["target"], "Target Twin Q'",
	     "Soft update τ")
	_dashed_arrow(fig, 8.25, 6.0, 8.25, 3.4, "soft update τ")

	# ── Actor loss
	_box(fig, 3, 3.5, 3, 1.2, C["loss"], "Actor Loss",
	     "L_π = α·log π(a|s) − min(Q₁,Q₂)<br>Maximize entropy + Q-value")
	_arrow(fig, 4.75, 6.0, 4.5, 4.7, "a, log π")
	_arrow(fig, 7, 6.75, 6.3, 6.75)

	# ── Critic loss
	_box(fig, 3, 1.5, 3, 1.2, C["loss"], "Critic Loss",
	     "L_Q = (Qᵢ − y)²<br>y = r + γ·(min Q' − α·log π')")
	_arrow(fig, 4.5, 3.5, 4.5, 2.7, "Q values")

	# ── Key feature
	fig.add_annotation(x=1, y=1.5, text=(
		"<b>Key:</b> entropy bonus<br>"
		"encourages exploration,<br>"
		"α auto-tuned to target H̄"
	), showarrow=False, font=dict(size=10), align="left",
		bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1, borderpad=6)

	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 6: QMIX / HAD3QN (Value Decomposition)
# ══════════════════════════════════════════════════════════════════
def create_value_decomposition() -> go.Figure:
	fig = _base_fig("QMIX / HAD3QN: Value Decomposition Methods",
	                height=580, yrange=(-0.5, 8.5))

	# ── Left side: QMIX
	fig.add_annotation(x=2.2, y=8, text="<b>QMIX</b>",
	                   showarrow=False, font=dict(size=14, color=C["text"]))

	_box(fig, 0, 6, 1.5, 1, C["obs"], "o₁")
	_box(fig, 1.8, 6, 1.5, 1, C["obs"], "o₂")
	_box(fig, 3.6, 6, 1.5, 1, C["obs"], "oₙ")

	_box(fig, 0, 4.2, 1.5, 1.2, C["policy"], "Q₁ Net", "Qᵢ(oᵢ, aᵢ)")
	_box(fig, 1.8, 4.2, 1.5, 1.2, C["policy"], "Q₂ Net", "per-agent Q")
	_box(fig, 3.6, 4.2, 1.5, 1.2, C["policy"], "Qₙ Net", "per-agent Q")
	_arrow(fig, 0.75, 6, 0.75, 5.4)
	_arrow(fig, 2.55, 6, 2.55, 5.4)
	_arrow(fig, 4.35, 6, 4.35, 5.4)

	_box(fig, 0.5, 2.2, 4, 1.4, C["module"], "QMixer",
	     "Monotonic mixing network<br>"
	     "Q_tot = f(Q₁,...,Qₙ; s)<br>"
	     "∂Q_tot/∂Qᵢ ≥ 0 (monotonicity)")
	_arrow(fig, 0.75, 4.2, 1.5, 3.6, "Q₁")
	_arrow(fig, 2.55, 4.2, 2.5, 3.6, "Q₂")
	_arrow(fig, 4.35, 4.2, 3.5, 3.6, "Qₙ")

	_box(fig, 0.5, 0.5, 4, 1, C["loss"], "TD Loss",
	     "L = (Q_tot − y)²,  y = r + γ·Q_tot'")
	_arrow(fig, 2.5, 2.2, 2.5, 1.5, "Q_tot")

	# ── Right side: HAD3QN
	fig.add_annotation(x=8, y=8, text="<b>HAD3QN</b>",
	                   showarrow=False, font=dict(size=14, color=C["text"]))

	_box(fig, 6.5, 6, 3, 1, C["obs"], "Agent i Obs", "oᵢ")

	_box(fig, 6.5, 4, 3, 1.5, C["policy"], "Dueling DQN",
	     "V(oᵢ) — state value stream<br>"
	     "A(oᵢ,a) — advantage stream<br>"
	     "Q = V + A − mean(A)")
	_arrow(fig, 8, 6, 8, 5.5)

	_box(fig, 6.5, 2.2, 3, 1.2, C["module"], "Double Q-Learning",
	     "Select: a* = argmax Q_online<br>"
	     "Evaluate: Q_target(s', a*)")
	_arrow(fig, 8, 4, 8, 3.4)

	_box(fig, 6.5, 0.5, 3, 1, C["loss"], "TD Loss",
	     "Per-agent TD error<br>ε-greedy exploration")
	_arrow(fig, 8, 2.2, 8, 1.5)

	# ── Divider
	fig.add_shape(type="line", x0=5.5, y0=0, x1=5.5, y1=8.2,
	              line=dict(color="#ddd", width=1, dash="dash"))

	return fig


# ══════════════════════════════════════════════════════════════════
# Figure 7: 2TS-VVC (Two-Timescale VVC)
# ══════════════════════════════════════════════════════════════════
def create_twots_vvc() -> go.Figure:
	fig = _base_fig("2TS-VVC: Two-Timescale Volt-VAR Control",
	                height=580, yrange=(-0.5, 8.5))

	# ── Environment
	_box(fig, 3.5, 7, 3, 1, C["env"], "Power Grid (OpenDSS)",
	     "Real-time co-simulation")

	# ── Coordinator
	_box(fig, 3.5, 5.2, 3, 1.2, C["module"], "TwoTS Coordinator",
	     "Orchestrates slow & fast agents<br>"
	     "Reward/Obs processing<br>"
	     "Timescale synchronization")
	_arrow(fig, 5, 7, 5, 6.4, "obs, reward")

	# ── Slow Agent (SACD) - left
	_box(fig, 0, 3, 3.5, 1.8, C["policy"], "Slow Agent (SACD)",
	     "Discrete actions (hourly)<br>"
	     "Capacitor switching<br>"
	     "Regulator tap positions<br>"
	     "SAC + discrete (Gumbel-Softmax)")
	_arrow(fig, 3.5, 5.8, 1.75, 4.8, "slow obs")

	# ── Fast Agent (DDPG) - right
	_box(fig, 6.5, 3, 3.5, 1.8, C["critic"], "Fast Agent (DDPG)",
	     "Continuous actions (per-step)<br>"
	     "Battery charge/discharge<br>"
	     "PV curtailment<br>"
	     "Deterministic policy")
	_arrow(fig, 6.5, 5.8, 8.25, 4.8, "fast obs")

	# ── Slow action output
	_box(fig, 0, 1.2, 3.5, 1.2, C["action"], "Slow Actions",
	     "Updated every T_slow steps<br>"
	     "Capacitor ON/OFF, Tap ±")
	_arrow(fig, 1.75, 3, 1.75, 2.4)

	# ── Fast action output
	_box(fig, 6.5, 1.2, 3.5, 1.2, C["action"], "Fast Actions",
	     "Updated every step<br>"
	     "Battery P, PV curtail %")
	_arrow(fig, 8.25, 3, 8.25, 2.4)

	# ── Combined actions back to env
	_arrow(fig, 1.75, 1.2, 4, 0.5, "discrete")
	_arrow(fig, 8.25, 1.2, 6, 0.5, "continuous")
	_box(fig, 3.5, 0, 3, 0.7, C["env"], "Joint Action → Grid", "")

	# ── Timescale annotation
	fig.add_annotation(x=5, y=8.2, text=(
		"<b>Timescales:</b>  "
		"Slow (hourly): discrete device switching  |  "
		"Fast (per-step): continuous power dispatch"
	), showarrow=False, font=dict(size=10, color="#555"))

	return fig


# ══════════════════════════════════════════════════════════════════
# Main: generate all figures
# ══════════════════════════════════════════════════════════════════
FIGURES = {
	"happo_family": ("HAPPO / HATRPO / HAA2C", create_happo_family),
	"mappo_family": ("MAPPO / SN-MAPPO", create_mappo_family),
	"dan_happo": ("DAN-HAPPO", create_dan_happo),
	"ddpg_family": ("DDPG Family", create_ddpg_family),
	"hasac": ("HASAC", create_hasac),
	"value_decomposition": ("QMIX / HAD3QN", create_value_decomposition),
	"twots_vvc": ("2TS-VVC", create_twots_vvc),
}


def main():
	output_dir = Path(__file__).resolve().parent.parent
	gh_dir = output_dir / "docs" / "assets"
	hf_dir = output_dir / "huggingface_space" / "data"
	gh_dir.mkdir(parents=True, exist_ok=True)
	hf_dir.mkdir(parents=True, exist_ok=True)

	for key, (name, factory) in FIGURES.items():
		print(f"Generating: {name} ...")
		fig = factory()

		# HTML for GitHub Pages
		fig.write_html(
			str(gh_dir / f"{key}.html"),
			include_plotlyjs="cdn",
			full_html=True,
			config={"displayModeBar": False, "staticPlot": False},
		)

		# JSON for HuggingFace Space
		with open(hf_dir / f"{key}.json", "w") as f:
			json.dump(json.loads(fig.to_json()), f)

		# PNG static image
		try:
			fig.write_image(str(gh_dir / f"{key}.png"), width=900, height=580, scale=2)
		except Exception as e:
			print(f"  PNG export skipped ({e})")

		print(f"  ✓ {key}.html / .json / .png")

	print(f"\nDone. Output dirs:\n  GitHub Pages: {gh_dir}\n  HF Space:     {hf_dir}")


if __name__ == "__main__":
	main()
