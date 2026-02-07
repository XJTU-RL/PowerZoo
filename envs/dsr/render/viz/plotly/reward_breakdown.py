"""
DSR Reward Breakdown
奖励分解图

展示 DSR 环境奖励组成:
- restoration_rate: 恢复率奖励 (主要)
- voltage_penalty: 电压违规惩罚
- overload_penalty: 过载惩罚
- done_reward: 恢复完成奖励
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import COLORS, DEVICE_COLORS, get_plotly_layout


# 奖励分量配色
REWARD_COLORS = {
	"restore_reward": COLORS["success"],
	"voltage_penalty": COLORS["warning"],
	"overload_penalty": COLORS["danger"],
	"done_reward": COLORS["info"],
	"total_reward": COLORS["primary"],
}


def create_reward_breakdown(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建奖励分解图

	上方: 堆叠面积图展示各奖励分量随时间变化
	下方: 累计总奖励曲线

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure 对象
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Reward Breakdown (No Data)", env_name="dsr"))
		return fig

	steps = []
	components: Dict[str, List[float]] = {
		"restore_reward": [],
		"voltage_penalty": [],
		"overload_penalty": [],
		"done_reward": [],
	}
	total_rewards = []
	cumulative_rewards = []

	for snap in snapshots:
		step = snap.get("step", 0)
		rc = snap.get("reward_components", {})

		if not rc and step == 0:
			continue

		steps.append(step)
		for key in components:
			components[key].append(float(rc.get(key, 0.0)))

		step_reward = snap.get("step_reward", 0.0)
		total_rewards.append(step_reward)
		cumulative_rewards.append(snap.get("cumulative_reward", 0.0))

	if not steps:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Reward Breakdown (No Reward Data)", env_name="dsr"))
		return fig

	from plotly.subplots import make_subplots
	fig = make_subplots(
		rows=2, cols=1,
		subplot_titles=["Reward Components per Step", "Cumulative Reward"],
		vertical_spacing=0.12,
		shared_xaxes=True,
		row_heights=[0.6, 0.4],
	)

	# 堆叠柱状图（DSR 步数少，柱状图比面积图更清晰）
	for key, values in components.items():
		if all(v == 0.0 for v in values):
			continue
		display_name = key.replace("_", " ").title()
		fig.add_trace(go.Bar(
			x=steps, y=values,
			name=display_name,
			marker_color=REWARD_COLORS.get(key, COLORS["secondary"]),
			hovertemplate=f"{display_name}<br>Step: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>",
		), row=1, col=1)

	fig.update_layout(barmode="relative")

	# 总奖励折线（叠加在柱状图上）
	fig.add_trace(go.Scatter(
		x=steps, y=total_rewards,
		mode="lines+markers",
		name="Step Total",
		line=dict(color="white", width=2, dash="dot"),
		marker=dict(size=5),
	), row=1, col=1)

	# 累计奖励
	fig.add_trace(go.Scatter(
		x=steps, y=cumulative_rewards,
		mode="lines+markers",
		name="Cumulative",
		line=dict(color=COLORS["primary"], width=2.5),
		marker=dict(size=6),
		fill="tozeroy",
		fillcolor="rgba(79,70,229,0.1)",
		hovertemplate="Step: %{x}<br>Cumulative: %{y:.3f}<extra></extra>",
	), row=2, col=1)

	# 标注最终累计奖励
	if cumulative_rewards:
		final = cumulative_rewards[-1]
		fig.add_annotation(
			x=steps[-1], y=final,
			text=f"Total: {final:.2f}",
			showarrow=True,
			arrowhead=2,
			font=dict(size=11, color=COLORS["primary"]),
			row=2, col=1,
		)

	layout = get_plotly_layout(title="Reward Breakdown", height=600, env_name="dsr")
	fig.update_layout(**layout)
	fig.update_yaxes(title_text="Reward", row=1, col=1)
	fig.update_yaxes(title_text="Cumulative", row=2, col=1)
	fig.update_xaxes(title_text="Step", row=2, col=1)

	return fig
