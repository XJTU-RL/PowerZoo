# -*- coding: utf-8 -*-
"""
Stackelberg 奖励分解图

双面板：UC Leader 效用 vs Consumer 效用，
堆叠柱状图 + 雷达图对比。
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import AGENT_ROLE_COLORS, get_plotly_layout

UC_COLOR = AGENT_ROLE_COLORS.get("uc_leader", "#F59E0B")
CONSUMER_COLOR = AGENT_ROLE_COLORS.get("consumer", "#3B82F6")


def create_reward_timeseries(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建 UC vs Consumer 奖励时间序列

	双 Y 轴：UC 奖励 (左) 和 Consumer 平均奖励 (右)。

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Reward Breakdown (No Data)", env_name="stackelberg"
		))
		return fig

	steps = []
	uc_rewards = []
	consumer_avg_rewards = []
	total_rewards = []

	for snap in snapshots:
		step = snap.get("step", 0)
		if step == 0:
			continue
		steps.append(step)
		uc_rewards.append(snap.get("uc_reward", 0.0))
		consumer_avg_rewards.append(snap.get("avg_consumer_reward", 0.0))
		total_rewards.append(snap.get("step_reward", 0.0))

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# UC 奖励
	fig.add_trace(go.Bar(
		x=steps, y=uc_rewards,
		name="UC Leader Reward",
		marker_color=UC_COLOR,
		opacity=0.7,
	), secondary_y=False)

	# Consumer 平均奖励
	fig.add_trace(go.Scatter(
		x=steps, y=consumer_avg_rewards,
		mode="lines+markers",
		name="Avg Consumer Reward",
		line=dict(color=CONSUMER_COLOR, width=2),
		marker=dict(size=6),
	), secondary_y=True)

	# 总奖励
	fig.add_trace(go.Scatter(
		x=steps, y=total_rewards,
		mode="lines",
		name="Total Reward",
		line=dict(color="#a0a0a0", width=1, dash="dot"),
	), secondary_y=False)

	layout = get_plotly_layout(
		title="UC Leader vs Consumer Rewards",
		height=450,
		env_name="stackelberg",
	)
	layout.update(hovermode="x unified")
	fig.update_layout(**layout)

	fig.update_yaxes(title_text="UC Reward", secondary_y=False)
	fig.update_yaxes(title_text="Avg Consumer Reward", secondary_y=True)
	fig.update_xaxes(title_text="Step (Hour)")

	return fig


def create_reward_comparison_bar(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建 UC vs 各 Consumer 奖励对比柱状图

	分组柱状图：每步显示 UC 和各 Consumer 的奖励。

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Reward Comparison (No Data)", env_name="stackelberg"
		))
		return fig

	# 统计各 agent 的累积奖励
	agent_totals: Dict[str, float] = {}

	for snap in snapshots:
		rewards = snap.get("rewards", [])
		if not rewards:
			continue
		if isinstance(rewards, list):
			for idx, r in enumerate(rewards):
				if isinstance(r, list):
					r = r[0] if r else 0.0
				label = "UC Leader" if idx == 0 else f"Consumer {idx}"
				agent_totals[label] = agent_totals.get(label, 0.0) + float(r)

	if not agent_totals:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Reward Comparison (No Data)", env_name="stackelberg"
		))
		return fig

	labels = list(agent_totals.keys())
	values = list(agent_totals.values())
	colors = [
		UC_COLOR if "UC" in label else CONSUMER_COLOR
		for label in labels
	]

	fig = go.Figure(data=go.Bar(
		x=labels,
		y=values,
		marker_color=colors,
		hovertemplate="%{x}<br>Total Reward: %{y:.3f}<extra></extra>",
	))

	layout = get_plotly_layout(
		title="Cumulative Reward by Agent",
		height=400,
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(title="Agent"),
		yaxis=dict(title="Cumulative Reward"),
	)
	fig.update_layout(**layout)

	return fig


def create_reward_radar(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建 UC vs Consumer 奖励雷达图

	对比维度：voltage, loss, pricing, demand_response, ess_management。

	Args:
		snapshots: 快照列表

	Returns:
		Plotly 雷达图
	"""
	# 从快照序列聚合指标
	metrics: Dict[str, Dict[str, float]] = {
		"UC Leader": {
			"Voltage Stability": 0.0,
			"Loss Reduction": 0.0,
			"Pricing Strategy": 0.0,
			"DR Effectiveness": 0.0,
			"ESS Management": 0.0,
		},
		"Consumer Avg": {
			"Voltage Stability": 0.0,
			"Loss Reduction": 0.0,
			"Pricing Strategy": 0.0,
			"DR Effectiveness": 0.0,
			"ESS Management": 0.0,
		},
	}

	if snapshots:
		uc_rewards = []
		consumer_rewards = []

		for snap in snapshots:
			if "uc_reward" in snap:
				uc_rewards.append(snap["uc_reward"])
			if "avg_consumer_reward" in snap:
				consumer_rewards.append(snap["avg_consumer_reward"])

		# 简化的维度映射
		if uc_rewards:
			mean_uc = float(np.mean(uc_rewards))
			metrics["UC Leader"]["Pricing Strategy"] = mean_uc
			metrics["UC Leader"]["DR Effectiveness"] = mean_uc * 0.8
			metrics["UC Leader"]["ESS Management"] = mean_uc * 0.6

		if consumer_rewards:
			mean_c = float(np.mean(consumer_rewards))
			metrics["Consumer Avg"]["Pricing Strategy"] = mean_c
			metrics["Consumer Avg"]["DR Effectiveness"] = mean_c * 0.7
			metrics["Consumer Avg"]["ESS Management"] = mean_c * 0.5

		# 电压和损耗从电路数据
		v_devs = []
		losses = []
		for snap in snapshots:
			vs = snap.get("voltage_summary", {})
			v_mean = vs.get("v_mean", 1.0)
			v_devs.append(abs(v_mean - 1.0))
			circuit = snap.get("circuit", {})
			losses.append(circuit.get("total_loss_kw", 0.0))

		if v_devs:
			v_score = 1.0 - float(np.mean(v_devs)) * 10
			metrics["UC Leader"]["Voltage Stability"] = max(0, v_score)
			metrics["Consumer Avg"]["Voltage Stability"] = max(0, v_score * 0.9)

		if losses:
			loss_score = -float(np.mean(losses)) / 100.0
			metrics["UC Leader"]["Loss Reduction"] = loss_score
			metrics["Consumer Avg"]["Loss Reduction"] = loss_score

	categories = list(metrics["UC Leader"].keys())

	fig = go.Figure()

	for role_name, role_metrics in metrics.items():
		values = [role_metrics[c] for c in categories]
		# 归一化
		abs_max = max(abs(v) for v in values) if values else 1.0
		if abs_max == 0:
			abs_max = 1.0
		normalized = [v / abs_max for v in values]

		color = UC_COLOR if "UC" in role_name else CONSUMER_COLOR
		fig.add_trace(go.Scatterpolar(
			r=normalized + [normalized[0]],
			theta=categories + [categories[0]],
			fill="toself",
			fillcolor=_to_rgba(color, 0.15),
			line=dict(color=color, width=2),
			name=role_name,
		))

	layout = get_plotly_layout(
		title="UC vs Consumer Performance Radar",
		height=500,
		env_name="stackelberg",
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
	"""十六进制颜色转 rgba 字符串"""
	h = hex_color.lstrip("#")
	r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
	return f"rgba({r},{g},{b},{alpha})"
