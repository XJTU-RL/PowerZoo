"""
PowerZoo Algorithm Architecture & Training Pipeline Visualization.

Generates interactive Plotly charts:
1. Algorithm Inheritance Hierarchy (tree layout)
2. Training Pipeline Flow (Sankey diagram)
3. Runner-Algorithm Mapping Matrix (heatmap)

Output: HTML files for GitHub Pages / HuggingFace Space integration.

Usage:
	python tools/plot_algorithm_architectures.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color Palette ──────────────────────────────────────────────
COLORS = {
	"on_policy_base": "#2196F3",   # Blue
	"on_policy": "#42A5F5",        # Light Blue
	"off_policy_base": "#FF5722",  # Deep Orange
	"off_policy": "#FF7043",       # Light Orange
	"special": "#9C27B0",          # Purple
	"twots": "#4CAF50",            # Green
	"runner_on": "#1565C0",        # Dark Blue
	"runner_off": "#BF360C",       # Dark Orange
	"runner_special": "#6A1B9A",   # Dark Purple
	"model": "#795548",            # Brown
	"env": "#009688",              # Teal
	"buffer": "#FFC107",           # Amber
	"config": "#607D8B",           # Blue Grey
	"bg": "#FAFAFA",               # Near White
	"edge": "#BDBDBD",             # Grey
	"text": "#212121",             # Dark
}


# ── Figure 1: Algorithm Inheritance Hierarchy ──────────────────
def create_algorithm_hierarchy():
	"""Create the algorithm inheritance tree using Scatter + Shapes."""

	# Node definitions: (label, x, y, color_key, description)
	nodes = [
		# Base classes (level 0)
		("OnPolicyBase", 2.5, 5, "on_policy_base",
		 "On-policy base class<br>get_actions(), evaluate_actions()<br>act(), lr_decay()"),
		("OffPolicyBase", 7.5, 5, "off_policy_base",
		 "Off-policy base class<br>soft_update(), save/restore()<br>turn_on/off_grad()"),

		# On-policy algorithms (level 1)
		("HAPPO", 0.5, 3.5, "on_policy",
		 "Heterogeneous-Agent PPO<br>Sequential update + factor_batch<br>Trust region + importance sampling"),
		("HATRPO", 1.5, 3.5, "on_policy",
		 "HA Trust Region Policy Opt<br>Conjugate gradient + line search<br>KL constraint optimization"),
		("HAA2C", 2.5, 3.5, "on_policy",
		 "HA Advantage Actor-Critic<br>Simpler than PPO (no clipping)<br>Single-step policy gradient"),
		("MAPPO", 3.5, 3.5, "on_policy",
		 "Multi-Agent PPO<br>Simultaneous update (all agents)<br>Supports share_param mode"),
		("SHOM", 4.5, 3.5, "on_policy",
		 "Sequential Heterogeneous<br>Order Mechanism<br>PPO-based with factor_batch"),

		# On-policy level 2
		("DAN_HAPPO", 0.5, 2, "on_policy",
		 "Dynamic Agent Network + HAPPO<br>Self-attention neighborhood<br>Separate DAN optimizer"),
		("SN_MAPPO", 3.5, 2, "on_policy",
		 "Stackelberg-Nash MAPPO<br>Leader-follower hierarchy<br>Dual learning rates"),

		# Off-policy algorithms (level 1)
		("HADDPG", 6, 3.5, "off_policy",
		 "HA Deep DPG<br>Deterministic policy gradient<br>Target network + soft update"),
		("HASAC", 7.5, 3.5, "off_policy",
		 "HA Soft Actor-Critic<br>Entropy regularization<br>Auto temperature tuning"),
		("HAD3QN", 9, 3.5, "off_policy",
		 "HA Dueling Double DQN<br>Discrete action space only<br>Epsilon-greedy exploration"),

		# Off-policy level 2
		("HATD3", 5.5, 2, "off_policy",
		 "HA Twin Delayed DDPG<br>Twin Q-networks<br>Delayed policy update"),
		("MADDPG", 6.5, 2, "off_policy",
		 "Multi-Agent DDPG<br>Simultaneous update<br>Centralized critic"),

		# Off-policy level 3
		("MATD3", 5.5, 0.5, "off_policy",
		 "Multi-Agent TD3<br>Twin Q + delayed update<br>Simultaneous agent updates"),

		# Special algorithms
		("M_QMix", 8.5, 0.5, "special",
		 "Multi-Agent QMix<br>Value decomposition<br>Hypernetwork mixing"),
		("TwoTSVVC", 10, 2, "twots",
		 "Two-Timescale VVC<br>Coordinator pattern<br>SACD(slow) + DDPG(fast)"),
	]

	# Edges: (parent_label, child_label)
	edges = [
		("OnPolicyBase", "HAPPO"),
		("OnPolicyBase", "HATRPO"),
		("OnPolicyBase", "HAA2C"),
		("OnPolicyBase", "MAPPO"),
		("OnPolicyBase", "SHOM"),
		("HAPPO", "DAN_HAPPO"),
		("MAPPO", "SN_MAPPO"),
		("OffPolicyBase", "HADDPG"),
		("OffPolicyBase", "HASAC"),
		("OffPolicyBase", "HAD3QN"),
		("HADDPG", "HATD3"),
		("HADDPG", "MADDPG"),
		("HATD3", "MATD3"),
	]

	# Build lookup
	node_map = {n[0]: n for n in nodes}

	fig = go.Figure()

	# Draw edges first (behind nodes)
	for parent_label, child_label in edges:
		p = node_map[parent_label]
		c = node_map[child_label]
		fig.add_trace(go.Scatter(
			x=[p[1], c[1]], y=[p[2], c[2]],
			mode="lines",
			line=dict(color=COLORS["edge"], width=2),
			hoverinfo="skip",
			showlegend=False,
		))

	# Draw nodes grouped by category for legend
	categories = {
		"On-Policy Base": ([n for n in nodes if n[3] == "on_policy_base"], COLORS["on_policy_base"], "circle"),
		"On-Policy Algorithms": ([n for n in nodes if n[3] == "on_policy"], COLORS["on_policy"], "circle"),
		"Off-Policy Base": ([n for n in nodes if n[3] == "off_policy_base"], COLORS["off_policy_base"], "diamond"),
		"Off-Policy Algorithms": ([n for n in nodes if n[3] == "off_policy"], COLORS["off_policy"], "diamond"),
		"Value Decomposition": ([n for n in nodes if n[3] == "special"], COLORS["special"], "star"),
		"Two-Timescale": ([n for n in nodes if n[3] == "twots"], COLORS["twots"], "hexagon"),
	}

	for cat_name, (cat_nodes, color, symbol) in categories.items():
		if not cat_nodes:
			continue
		fig.add_trace(go.Scatter(
			x=[n[1] for n in cat_nodes],
			y=[n[2] for n in cat_nodes],
			mode="markers+text",
			marker=dict(
				size=28,
				color=color,
				symbol=symbol,
				line=dict(color="white", width=2),
			),
			text=[n[0] for n in cat_nodes],
			textposition="top center",
			textfont=dict(size=11, color=COLORS["text"], family="Consolas, monospace"),
			hovertext=[n[4] for n in cat_nodes],
			hoverinfo="text",
			name=cat_name,
		))

	# Dashed lines for independent algorithms
	for label in ["M_QMix", "TwoTSVVC"]:
		n = node_map[label]
		fig.add_annotation(
			x=n[1], y=n[2] + 0.3,
			text="(independent)",
			font=dict(size=9, color="#757575"),
			showarrow=False,
		)

	fig.update_layout(
		title=dict(
			text="PowerZoo Algorithm Inheritance Hierarchy",
			font=dict(size=20, family="Arial Black"),
			x=0.5,
		),
		xaxis=dict(
			showgrid=False, zeroline=False, showticklabels=False,
			range=[-0.5, 11],
		),
		yaxis=dict(
			showgrid=False, zeroline=False, showticklabels=False,
			range=[-0.5, 6],
		),
		plot_bgcolor=COLORS["bg"],
		paper_bgcolor="white",
		height=650,
		width=1100,
		legend=dict(
			orientation="h",
			yanchor="bottom", y=-0.12,
			xanchor="center", x=0.5,
			font=dict(size=11),
		),
		margin=dict(l=20, r=20, t=60, b=80),
	)

	return fig


# ── Figure 2: Training Pipeline Sankey Diagram ────────────────
def create_training_pipeline():
	"""Create the training pipeline flow as a Sankey diagram."""

	# Node labels for the Sankey
	labels = [
		# 0-3: Entry & Config
		"CLI (train.py)",          # 0
		"Algorithm Config",        # 1
		"Environment Config",      # 2
		"System Config",           # 3

		# 4-9: Runners
		"OnPolicyHARunner",        # 4
		"OnPolicyMARunner",        # 5
		"OffPolicyHARunner",       # 6
		"OffPolicyMARunner",       # 7
		"QMIXRunner",              # 8
		"TwoTSRunner",             # 9

		# 10-12: Core Components
		"Actor (Policy Network)",  # 10
		"Critic (Value Network)",  # 11
		"Experience Buffer",       # 12

		# 13-16: Environments
		"PowerZoo VVC",            # 13
		"SmartGrid",               # 14
		"Stackelberg Game",        # 15
		"DSR",                     # 16

		# 17-20: Training Phases
		"Rollout (Data Collection)",  # 17
		"Advantage / Q Computation",  # 18
		"Policy Optimization",        # 19
		"Value Function Update",      # 20

		# 21-23: Outputs
		"TensorBoard Logs",        # 21
		"Model Checkpoints",       # 22
		"Evaluation Results",      # 23
	]

	# Source → Target links
	source = [
		# Config → Runner
		0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2,
		3, 3, 3, 3,

		# Runner → Components
		4, 4, 4,
		5, 5, 5,
		6, 6, 6,
		7, 7, 7,
		8, 8, 8,
		9, 9, 9,

		# Components → Environments
		10, 10, 10, 10,
		11, 11, 11, 11,

		# Training Loop
		13, 14, 15, 16,
		17, 17,
		12, 18,
		19, 19,
		20, 20, 20,
	]

	target = [
		# Config → Runner
		4, 5, 6, 7, 8, 9,
		4, 5, 6, 7, 8, 9,
		4, 5, 6, 7, 8, 9,
		4, 5, 6, 7,

		# Runner → Components
		10, 11, 12,
		10, 11, 12,
		10, 11, 12,
		10, 11, 12,
		10, 11, 12,
		10, 11, 12,

		# Components → Environments
		13, 14, 15, 16,
		13, 14, 15, 16,

		# Training Loop
		17, 17, 17, 17,
		18, 12,
		18, 19,
		20, 21,
		21, 22, 23,
	]

	value = [
		# Config → Runner
		3, 1, 2, 1, 1, 1,
		3, 1, 2, 1, 1, 1,
		3, 1, 2, 1, 1, 1,
		3, 1, 2, 1,

		# Runner → Components
		4, 4, 4,
		2, 2, 2,
		3, 3, 3,
		2, 2, 2,
		1, 1, 1,
		1, 1, 1,

		# Components → Environments
		4, 3, 2, 2,
		4, 3, 2, 2,

		# Training Loop
		4, 3, 2, 2,
		6, 5,
		5, 6,
		6, 6,
		4, 4, 4,
	]

	# Node colors
	node_colors = [
		COLORS["config"],
		COLORS["config"],
		COLORS["config"],
		COLORS["config"],
		COLORS["runner_on"],
		COLORS["runner_on"],
		COLORS["runner_off"],
		COLORS["runner_off"],
		COLORS["runner_special"],
		COLORS["runner_special"],
		COLORS["on_policy"],
		COLORS["off_policy"],
		COLORS["buffer"],
		COLORS["env"],
		COLORS["env"],
		COLORS["env"],
		COLORS["env"],
		"#E91E63",
		"#673AB7",
		COLORS["on_policy_base"],
		COLORS["off_policy_base"],
		"#FF9800",
		"#4CAF50",
		"#00BCD4",
	]

	fig = go.Figure(data=[go.Sankey(
		arrangement="snap",
		node=dict(
			pad=15,
			thickness=20,
			line=dict(color="white", width=1),
			label=labels,
			color=node_colors,
			hovertemplate="%{label}<extra></extra>",
		),
		link=dict(
			source=source,
			target=target,
			value=value,
			color="rgba(180, 180, 180, 0.3)",
			hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>"
			              "Flow: %{value}<extra></extra>",
		),
	)])

	fig.update_layout(
		title=dict(
			text="PowerZoo Training Pipeline Flow",
			font=dict(size=20, family="Arial Black"),
			x=0.5,
		),
		font=dict(size=11, family="Arial"),
		height=700,
		width=1200,
		paper_bgcolor="white",
		margin=dict(l=10, r=10, t=60, b=20),
	)

	return fig


# ── Figure 3: Runner-Algorithm Mapping Matrix ─────────────────
def create_runner_algorithm_matrix():
	"""Create the runner-algorithm compatibility heatmap."""

	runners = [
		"OnPolicyHARunner",
		"OnPolicyMARunner",
		"OffPolicyHARunner",
		"OffPolicyMARunner",
		"QMIXRunner",
		"TwoTSRunner",
	]

	algorithms = [
		"HAPPO", "HATRPO", "HAA2C", "SHOM", "DAN_HAPPO", "SN_MAPPO",
		"MAPPO",
		"HADDPG", "HATD3", "HASAC", "HAD3QN",
		"MADDPG", "MATD3",
		"M_QMix",
		"2TS_VVC",
	]

	# Mapping matrix: 1 = supported, 0 = not supported
	z = [
		[1, 1, 1, 1, 1, 1,  0,  0, 0, 0, 0,  0, 0,  0, 0],
		[0, 0, 0, 0, 0, 0,  1,  0, 0, 0, 0,  0, 0,  0, 0],
		[0, 0, 0, 0, 0, 0,  0,  1, 1, 1, 1,  0, 0,  0, 0],
		[0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0,  1, 1,  0, 0],
		[0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0,  0, 0,  1, 0],
		[0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0,  0, 0,  0, 1],
	]

	# Annotation text
	text = []
	for row in z:
		text_row = []
		for val in row:
			text_row.append("✓" if val == 1 else "")
		text.append(text_row)

	colorscale = [[0, "#F5F5F5"], [1, "#4CAF50"]]

	fig = go.Figure(data=go.Heatmap(
		z=z,
		x=algorithms,
		y=runners,
		text=text,
		texttemplate="%{text}",
		textfont=dict(size=16, color="white"),
		colorscale=colorscale,
		showscale=False,
		hovertemplate="<b>%{y}</b><br>Algorithm: %{x}<br>"
		              "Supported: %{z}<extra></extra>",
	))

	# Category separator lines
	fig.add_shape(type="line", x0=5.5, x1=5.5, y0=-0.5, y1=5.5,
	              line=dict(color="#E0E0E0", width=2, dash="dash"))
	fig.add_shape(type="line", x0=6.5, x1=6.5, y0=-0.5, y1=5.5,
	              line=dict(color="#E0E0E0", width=2, dash="dash"))
	fig.add_shape(type="line", x0=10.5, x1=10.5, y0=-0.5, y1=5.5,
	              line=dict(color="#E0E0E0", width=2, dash="dash"))
	fig.add_shape(type="line", x0=12.5, x1=12.5, y0=-0.5, y1=5.5,
	              line=dict(color="#E0E0E0", width=2, dash="dash"))

	annotations = [
		dict(x=2.5, y=6.2, text="On-Policy HA", font=dict(size=11, color=COLORS["on_policy_base"]),
		     showarrow=False, xref="x", yref="y"),
		dict(x=6, y=6.2, text="On-Policy MA", font=dict(size=11, color=COLORS["on_policy"]),
		     showarrow=False, xref="x", yref="y"),
		dict(x=8.5, y=6.2, text="Off-Policy HA", font=dict(size=11, color=COLORS["off_policy_base"]),
		     showarrow=False, xref="x", yref="y"),
		dict(x=11.5, y=6.2, text="Off-Policy MA", font=dict(size=11, color=COLORS["off_policy"]),
		     showarrow=False, xref="x", yref="y"),
		dict(x=13, y=6.2, text="Special", font=dict(size=11, color=COLORS["special"]),
		     showarrow=False, xref="x", yref="y"),
		# Update mechanism labels
		dict(x=2.5, y=-1.2, text="Sequential Update (factor_batch)",
		     font=dict(size=10, color="#757575"), showarrow=False),
		dict(x=6, y=-1.2, text="Simultaneous",
		     font=dict(size=10, color="#757575"), showarrow=False),
		dict(x=8.5, y=-1.2, text="Sequential + Soft Update",
		     font=dict(size=10, color="#757575"), showarrow=False),
		dict(x=11.5, y=-1.2, text="Simultaneous",
		     font=dict(size=10, color="#757575"), showarrow=False),
		dict(x=13.5, y=-1.2, text="Value Decomp / 2TS",
		     font=dict(size=10, color="#757575"), showarrow=False),
	]

	fig.update_layout(
		title=dict(
			text="Runner ↔ Algorithm Compatibility Matrix",
			font=dict(size=20, family="Arial Black"),
			x=0.5,
		),
		xaxis=dict(
			title="Algorithm",
			tickangle=-45,
			side="bottom",
			range=[-0.5, 14.5],
		),
		yaxis=dict(
			title="Runner",
			autorange="reversed",
			range=[-0.5, 5.5],
		),
		annotations=annotations,
		height=500,
		width=1100,
		paper_bgcolor="white",
		plot_bgcolor="white",
		margin=dict(l=150, r=20, t=60, b=120),
	)

	return fig


# ── Output Generation ──────────────────────────────────────────
def main():
	"""Generate all architecture figures and export."""
	output_dir = Path(__file__).parent.parent

	# Generate figures
	fig_hierarchy = create_algorithm_hierarchy()
	fig_pipeline = create_training_pipeline()
	fig_matrix = create_runner_algorithm_matrix()

	# Export standalone HTML files for GitHub Pages
	gh_pages_dir = output_dir / "docs" / "assets"
	gh_pages_dir.mkdir(parents=True, exist_ok=True)

	fig_hierarchy.write_html(
		str(gh_pages_dir / "algorithm_hierarchy.html"),
		include_plotlyjs="cdn",
		full_html=False,
	)
	fig_pipeline.write_html(
		str(gh_pages_dir / "training_pipeline.html"),
		include_plotlyjs="cdn",
		full_html=False,
	)
	fig_matrix.write_html(
		str(gh_pages_dir / "runner_algorithm_matrix.html"),
		include_plotlyjs="cdn",
		full_html=False,
	)

	# Export JSON data for HuggingFace Space
	hf_data_dir = output_dir / "huggingface_space" / "data"
	hf_data_dir.mkdir(parents=True, exist_ok=True)

	fig_hierarchy.write_json(str(hf_data_dir / "algorithm_hierarchy.json"))
	fig_pipeline.write_json(str(hf_data_dir / "training_pipeline.json"))
	fig_matrix.write_json(str(hf_data_dir / "runner_algorithm_matrix.json"))

	# Export static images
	try:
		img_dir = gh_pages_dir
		fig_hierarchy.write_image(str(img_dir / "algorithm_hierarchy.png"), width=1100, height=650, scale=2)
		fig_pipeline.write_image(str(img_dir / "training_pipeline.png"), width=1200, height=700, scale=2)
		fig_matrix.write_image(str(img_dir / "runner_algorithm_matrix.png"), width=1100, height=500, scale=2)
		print("Static images exported successfully.")
	except Exception as e:
		print(f"Warning: Static image export failed ({e}). HTML files are still available.")

	print(f"HTML fragments exported to: {gh_pages_dir}")
	print(f"JSON data exported to: {hf_data_dir}")
	print("Done!")


if __name__ == "__main__":
	main()
