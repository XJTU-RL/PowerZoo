"""
奖励分解图
堆叠柱状图 (6 分量 per agent, over time) + 雷达图 (各分量对比)。
"""

import math
from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.district_dispatch.render.viz.theme import (
	ZONE_COLORS,
	get_plotly_layout,
)

# 奖励分量定义: (key, display_name, color)
REWARD_COMPONENTS: List[tuple[str, str, str]] = [
	("economic", "Economic", "#10B981"),
	("voltage", "Voltage", "#F59E0B"),
	("loss", "Loss", "#EF4444"),
	("carbon", "Carbon", "#6B7280"),
	("exchange", "Exchange", "#3B82F6"),
	("storage", "Storage", "#7C3AED"),
]


def _extract_reward_timeseries(
	snapshots: List[Dict[str, Any]],
) -> tuple[list[float], dict[str, list[float]], list[str]]:
	"""从快照序列提取奖励分量时间序列。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		tuple: (timestamps, component_series, agent_names)
	"""
	timestamps: list[float] = []
	component_series: dict[str, list[float]] = {k: [] for k, _, _ in REWARD_COMPONENTS}
	agent_names: list[str] = []

	for i, snap in enumerate(snapshots):
		timestamps.append(snap.get("timestamp_h", i * 0.25))
		rc = snap.get("reward_components", {})

		for key, _, _ in REWARD_COMPONENTS:
			component_series[key].append(rc.get(key, 0.0))

		# 收集 agent 信息
		if not agent_names:
			rewards = snap.get("rewards", [])
			agent_names = [f"Agent {j}" for j in range(len(rewards))]

	return timestamps, component_series, agent_names


def create_reward_stacked(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建奖励分解堆叠柱状图。

	6 个奖励分量 (economic, voltage, loss, carbon, exchange, storage)
	按时间步堆叠显示。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		go.Figure: Plotly 堆叠柱状图
	"""
	timestamps, component_series, _ = _extract_reward_timeseries(snapshots)

	if not timestamps:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout("Reward Breakdown (No Data)"))
		return fig

	time_labels = [f"{t:.2f}" for t in timestamps]

	fig = go.Figure()

	for key, name, color in REWARD_COMPONENTS:
		values = component_series[key]
		fig.add_trace(go.Bar(
			x=time_labels,
			y=values,
			name=name,
			marker_color=color,
			opacity=0.85,
			hovertemplate=f"{name}<br>Time: %{{x}}h<br>Reward: %{{y:.3f}}<extra></extra>",
		))

	layout = get_plotly_layout(
		title="Reward Breakdown (Stacked Bar)",
		height=500,
	)
	layout.update(
		barmode="relative",
		xaxis=dict(title="Time (h)"),
		yaxis=dict(title="Reward"),
		hovermode="x unified",
	)
	fig.update_layout(**layout)

	return fig


def create_reward_radar(
	snapshot: Dict[str, Any],
) -> go.Figure:
	"""创建奖励分量雷达图。

	展示单时间步各奖励分量的对比，支持多 agent。

	Args:
		snapshot: 单时间步快照数据

	Returns:
		go.Figure: Plotly 雷达图
	"""
	rc = snapshot.get("reward_components", {})
	rewards = snapshot.get("rewards", [])

	categories = [name for _, name, _ in REWARD_COMPONENTS]
	component_keys = [key for key, _, _ in REWARD_COMPONENTS]

	# 单 agent 或汇总模式
	if isinstance(rc, dict) and not any(isinstance(v, list) for v in rc.values()):
		# 全局奖励分量
		values = [rc.get(k, 0.0) for k in component_keys]

		# 归一化到 [0, 1] 用于雷达图
		abs_max = max(abs(v) for v in values) if values else 1.0
		if abs_max == 0:
			abs_max = 1.0
		normalized = [v / abs_max for v in values]

		fig = go.Figure()
		fig.add_trace(go.Scatterpolar(
			r=normalized + [normalized[0]],  # 闭合
			theta=categories + [categories[0]],
			fill="toself",
			fillcolor="rgba(79,70,229,0.2)",
			line=dict(color="#4F46E5", width=2),
			name="Total Reward",
			hovertemplate="Component: %{theta}<br>Value: %{r:.3f}<extra></extra>",
		))
	else:
		# 多 agent 模式
		fig = go.Figure()
		n_agents = len(rewards) if rewards else 1
		agent_colors = ZONE_COLORS + ["#EF4444", "#7C3AED", "#3B82F6"]

		for agent_idx in range(n_agents):
			values = []
			for k in component_keys:
				comp_val = rc.get(k, 0.0)
				if isinstance(comp_val, list) and agent_idx < len(comp_val):
					values.append(comp_val[agent_idx])
				elif isinstance(comp_val, (int, float)):
					values.append(comp_val)
				else:
					values.append(0.0)

			abs_max = max(abs(v) for v in values) if values else 1.0
			if abs_max == 0:
				abs_max = 1.0
			normalized = [v / abs_max for v in values]
			color = agent_colors[agent_idx % len(agent_colors)]

			fig.add_trace(go.Scatterpolar(
				r=normalized + [normalized[0]],
				theta=categories + [categories[0]],
				fill="toself",
				fillcolor=_to_rgba(color, 0.15),
				line=dict(color=color, width=2),
				name=f"Agent {agent_idx}",
			))

	step = snapshot.get("step", 0)
	layout = get_plotly_layout(
		title=f"Reward Radar (Step {step})",
		height=500,
	)
	layout.update(
		polar=dict(
			bgcolor="rgba(15,15,35,0.8)",
			radialaxis=dict(
				visible=True,
				range=[-1, 1],
				gridcolor="rgba(255,255,255,0.1)",
			),
			angularaxis=dict(
				gridcolor="rgba(255,255,255,0.1)",
			),
		),
	)
	fig.update_layout(**layout)

	return fig


def _to_rgba(hex_color: str, alpha: float) -> str:
	"""将十六进制颜色转为 rgba 字符串。

	Args:
		hex_color: 十六进制颜色, 如 "#EF4444"
		alpha: 透明度 (0-1)

	Returns:
		str: rgba 格式颜色字符串
	"""
	h = hex_color.lstrip("#")
	r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
	return f"rgba({r},{g},{b},{alpha})"
