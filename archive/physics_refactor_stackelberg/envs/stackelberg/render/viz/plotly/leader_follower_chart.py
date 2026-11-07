# -*- coding: utf-8 -*-
"""
Stackelberg 独有：Leader-Follower 博弈动态图

多子图展示 UC Leader 决策和 Consumer 响应之间的互动关系：
1. UC Leader 5D 动作时间序列
2. Consumer 平均 3D 响应时间序列
3. UC 效用 vs Consumer 平均效用对比
4. Leader 定价 vs Consumer 负荷调整散点图
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import AGENT_ROLE_COLORS, get_plotly_layout

UC_COLOR = AGENT_ROLE_COLORS.get("uc_leader", "#F59E0B")
CONSUMER_COLOR = AGENT_ROLE_COLORS.get("consumer", "#3B82F6")

# UC 动作维度颜色
UC_DIM_COLORS: List[str] = [
	"#F59E0B",   # price
	"#7C3AED",   # DR_signal
	"#3B82F6",   # ESS_charge
	"#EF4444",   # ESS_discharge
	"#6B7280",   # reserve
]

UC_DIM_LABELS: List[str] = [
	"Price", "DR Signal", "ESS Charge", "ESS Discharge", "Reserve",
]

# Consumer 动作维度颜色
CONSUMER_DIM_COLORS: List[str] = [
	"#10B981",   # load_adjustment
	"#F59E0B",   # DER_output
	"#7C3AED",   # flexibility
]

CONSUMER_DIM_LABELS: List[str] = [
	"Load Adj", "DER Output", "Flexibility",
]


def create_leader_follower_chart(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建 Leader-Follower 博弈动态 4 子图

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Leader-Follower Dynamics (No Data)", env_name="stackelberg"
		))
		return fig

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"UC Leader Actions (5D)",
			"Avg Consumer Response (3D)",
			"Utility Comparison",
			"Price vs Load Adjustment",
		),
		vertical_spacing=0.12,
		horizontal_spacing=0.10,
	)

	steps = []
	uc_actions_series: Dict[int, List[float]] = {i: [] for i in range(5)}
	consumer_avg_series: Dict[int, List[float]] = {i: [] for i in range(3)}
	uc_utilities = []
	consumer_utilities = []
	price_values = []
	load_adj_values = []

	for snap in snapshots:
		step = snap.get("step", 0)
		if step == 0:
			continue
		steps.append(step)

		market = snap.get("market_data", {})
		uc_act = market.get("uc_actions", {})
		consumer_acts = market.get("consumer_actions", [])

		# UC 动作
		uc_keys = ["price", "DR_signal", "ESS_charge", "ESS_discharge", "reserve"]
		for i, key in enumerate(uc_keys):
			uc_actions_series[i].append(uc_act.get(key, 0.0))

		# Consumer 平均动作
		c_keys = ["load_adjustment", "DER_output", "flexibility"]
		for i, key in enumerate(c_keys):
			if consumer_acts:
				vals = [c.get(key, 0.0) for c in consumer_acts]
				consumer_avg_series[i].append(float(np.mean(vals)))
			else:
				consumer_avg_series[i].append(0.0)

		# 效用
		uc_utilities.append(market.get("uc_utility", snap.get("uc_reward", 0.0)))
		consumer_utilities.append(
			market.get("avg_consumer_utility", snap.get("avg_consumer_reward", 0.0))
		)

		# 散点图数据
		price_values.append(uc_act.get("effective_price", 0.0))
		if consumer_acts:
			load_adj_values.append(
				float(np.mean([c.get("load_adjustment", 0.0) for c in consumer_acts]))
			)
		else:
			load_adj_values.append(0.0)

	# Row 1, Col 1: UC Leader Actions
	for dim_idx in range(5):
		fig.add_trace(go.Scatter(
			x=steps,
			y=uc_actions_series[dim_idx],
			mode="lines",
			name=UC_DIM_LABELS[dim_idx],
			line=dict(color=UC_DIM_COLORS[dim_idx], width=1.5),
			legendgroup="uc",
		), row=1, col=1)

	# Row 1, Col 2: Consumer Avg Response
	for dim_idx in range(3):
		fig.add_trace(go.Scatter(
			x=steps,
			y=consumer_avg_series[dim_idx],
			mode="lines+markers",
			name=CONSUMER_DIM_LABELS[dim_idx],
			line=dict(color=CONSUMER_DIM_COLORS[dim_idx], width=1.5),
			marker=dict(size=4),
			legendgroup="consumer",
		), row=1, col=2)

	# Row 2, Col 1: Utility Comparison
	fig.add_trace(go.Bar(
		x=steps, y=uc_utilities,
		name="UC Utility",
		marker_color=UC_COLOR,
		opacity=0.7,
		legendgroup="utility",
	), row=2, col=1)

	fig.add_trace(go.Scatter(
		x=steps, y=consumer_utilities,
		mode="lines+markers",
		name="Avg Consumer Utility",
		line=dict(color=CONSUMER_COLOR, width=2),
		legendgroup="utility",
	), row=2, col=1)

	# Row 2, Col 2: Price vs Load Adj Scatter
	if price_values and load_adj_values:
		fig.add_trace(go.Scatter(
			x=price_values,
			y=load_adj_values,
			mode="markers",
			name="Price-Response",
			marker=dict(
				size=8,
				color=steps,
				colorscale="YlOrRd",
				showscale=True,
				colorbar=dict(
					title="Step",
					x=1.05,
					len=0.4,
					y=0.2,
					tickfont=dict(color="#e0e0e0"),
					titlefont=dict(color="#e0e0e0"),
				),
			),
			hovertemplate=(
				"Price: %{x:.4f}<br>"
				"Load Adj: %{y:.4f}<br>"
				"Step: %{marker.color}<extra></extra>"
			),
			legendgroup="scatter",
		), row=2, col=2)

	layout = get_plotly_layout(
		title="Stackelberg Leader-Follower Game Dynamics",
		height=700,
		env_name="stackelberg",
	)
	layout.update(hovermode="closest")
	fig.update_layout(**layout)

	fig.update_yaxes(title_text="Action Value", row=1, col=1)
	fig.update_yaxes(title_text="Action Value", row=1, col=2)
	fig.update_yaxes(title_text="Utility", row=2, col=1)
	fig.update_yaxes(title_text="Load Adjustment", row=2, col=2)
	fig.update_xaxes(title_text="Step", row=2, col=1)
	fig.update_xaxes(title_text="Effective Price", row=2, col=2)

	return fig


def create_action_heatmap(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建动作热力图

	展示所有 agent 在所有时间步的动作值。

	Args:
		snapshots: 快照列表

	Returns:
		Plotly 热力图
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Action Heatmap (No Data)", env_name="stackelberg"
		))
		return fig

	# 收集动作数据
	all_actions: List[List[float]] = []
	steps = []
	agent_labels: List[str] = []

	for snap in snapshots:
		step = snap.get("step", 0)
		if step == 0:
			continue
		steps.append(step)

		actions = snap.get("actions", [])
		if not actions:
			continue

		flat_row: List[float] = []
		for agent_idx, agent_actions in enumerate(actions):
			if isinstance(agent_actions, (list, tuple)):
				flat_row.extend(agent_actions)
			else:
				flat_row.append(float(agent_actions))

			# 构建标签 (仅第一次)
			if not agent_labels:
				if agent_idx == 0:
					for i, label in enumerate(UC_DIM_LABELS):
						agent_labels.append(f"UC:{label}")
				else:
					for i, label in enumerate(CONSUMER_DIM_LABELS):
						agent_labels.append(f"C{agent_idx}:{label}")

		all_actions.append(flat_row)

	if not all_actions or not agent_labels:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Action Heatmap (No Data)", env_name="stackelberg"
		))
		return fig

	# 转换为矩阵 (action_dim x steps)
	max_dim = len(agent_labels)
	matrix = np.zeros((max_dim, len(steps)))
	for j, row in enumerate(all_actions):
		for i in range(min(len(row), max_dim)):
			matrix[i, j] = row[i]

	fig = go.Figure(data=go.Heatmap(
		z=matrix,
		x=[str(s) for s in steps],
		y=agent_labels,
		colorscale="RdBu",
		zmid=0,
		colorbar=dict(
			title="Value",
			tickfont=dict(color="#e0e0e0"),
			titlefont=dict(color="#e0e0e0"),
		),
		hovertemplate="Dim: %{y}<br>Step: %{x}<br>Value: %{z:.4f}<extra></extra>",
	))

	layout = get_plotly_layout(
		title="All Agent Actions Heatmap",
		height=max(400, max_dim * 25 + 100),
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(title="Step"),
		yaxis=dict(title="Action Dimension"),
	)
	fig.update_layout(**layout)

	return fig
