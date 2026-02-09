# -*- coding: utf-8 -*-
"""
VVC 奖励分解图 (Plotly)

展示 VVC 奖励的三个分量随时间步的变化:
power_loss, voltage_violation, control_cost
使用堆叠面积图。
"""

from typing import Any, Dict, List

import plotly.graph_objects as go

from envs.render_common.viz.theme import COLORS, get_plotly_layout


# 奖励分量配色
_COMPONENT_COLORS = {
	"power_loss": "#EF4444",
	"power_loss_ratio": "#EF4444",
	"voltage_violation": "#F59E0B",
	"voltage_penalty": "#F59E0B",
	"control_cost": "#3B82F6",
	"switching_cost": "#3B82F6",
}

_COMPONENT_LABELS = {
	"power_loss": "Power Loss",
	"power_loss_ratio": "Power Loss",
	"voltage_violation": "Voltage Violation",
	"voltage_penalty": "Voltage Violation",
	"control_cost": "Control Cost",
	"switching_cost": "Control Cost",
}


def create_reward_breakdown(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建奖励分解堆叠面积图。

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure 对象
	"""
	if not snapshots:
		fig = go.Figure()
		layout = get_plotly_layout(
			title="Reward Breakdown (No Data)", height=400, env_name="vvc"
		)
		fig.update_layout(**layout)
		return fig

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	# 发现所有奖励分量
	component_keys: List[str] = []
	seen: set = set()
	for snap in snapshots:
		rc = snap.get("reward_components", {})
		for key in rc:
			if key not in seen:
				seen.add(key)
				component_keys.append(key)

	fig = go.Figure()

	if component_keys:
		# 堆叠面积图
		for key in component_keys:
			values = []
			for snap in snapshots:
				rc = snap.get("reward_components", {})
				val = rc.get(key, 0.0)
				if isinstance(val, (list, tuple)):
					val = sum(val) / len(val) if val else 0.0
				values.append(float(val))

			color = _COMPONENT_COLORS.get(key, COLORS["info"])
			label = _COMPONENT_LABELS.get(key, key.replace("_", " ").title())

			fig.add_trace(go.Scatter(
				x=steps,
				y=values,
				mode="lines",
				name=label,
				line=dict(color=color, width=1),
				fill="tonexty" if len(fig.data) > 0 else "tozeroy",
				fillcolor=_rgba(color, 0.3),
				stackgroup="rewards",
			))

	# 总奖励曲线 (叠加)
	total_rewards = []
	for snap in snapshots:
		reward = snap.get("step_reward", 0.0)
		if isinstance(reward, (int, float)):
			total_rewards.append(reward)
		else:
			total_rewards.append(0.0)

	if any(r != 0 for r in total_rewards):
		fig.add_trace(go.Scatter(
			x=steps,
			y=total_rewards,
			mode="lines+markers",
			name="Total Reward",
			line=dict(color=COLORS["primary"], width=2.5),
			marker=dict(size=4),
		))

	# 累计奖励 (第二 Y 轴)
	cumulative = []
	cum_sum = 0.0
	for r in total_rewards:
		cum_sum += r
		cumulative.append(cum_sum)

	if cumulative:
		fig.add_trace(go.Scatter(
			x=steps,
			y=cumulative,
			mode="lines",
			name="Cumulative Reward",
			line=dict(color=COLORS["secondary"], width=2, dash="dash"),
			yaxis="y2",
		))

	layout = get_plotly_layout(
		title="Reward Breakdown",
		height=450,
		env_name="vvc",
	)
	layout.update({
		"xaxis_title": "Step",
		"yaxis_title": "Reward Component",
		"yaxis2": dict(
			title="Cumulative Reward",
			overlaying="y",
			side="right",
			gridcolor="rgba(255,255,255,0.05)",
		),
		"legend": dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1,
			bgcolor="rgba(0,0,0,0.3)",
		),
	})
	fig.update_layout(**layout)

	return fig


def _rgba(hex_color: str, alpha: float) -> str:
	"""将 hex 颜色转为 rgba 字符串。

	Args:
		hex_color: 十六进制颜色
		alpha: 透明度

	Returns:
		rgba 字符串
	"""
	h = hex_color.lstrip("#")
	r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
	return f"rgba({r},{g},{b},{alpha})"
