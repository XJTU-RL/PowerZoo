"""
SmartGrid Reward Breakdown (Plotly)
奖励分解图

显示 CMDP 框架下的 objective (功率损耗) 和 constraint violation (电压违规)
奖励组件时间序列。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import get_plotly_layout, COLORS, DEVICE_COLORS

logger = logging.getLogger(__name__)


def create_reward_breakdown(
	snapshots: List[Dict[str, Any]],
	height: int = 500,
) -> go.Figure:
	"""创建奖励分解图

	CMDP 框架: 显示 objective + constraint violation + 总奖励

	Args:
		snapshots: 快照列表
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		layout = get_plotly_layout(title="Reward Breakdown", height=height, env_name="smartgrid")
		fig.update_layout(**layout)
		return fig

	fig = make_subplots(
		rows=2, cols=1,
		shared_xaxes=True,
		vertical_spacing=0.08,
		subplot_titles=["Reward Components", "Cumulative Reward"],
	)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	x_labels = [f"Day {s + 1}" for s in steps]

	# 提取奖励组件
	component_series = _extract_reward_series(snapshots)

	# Row 1: 奖励组件
	component_colors = {
		"vol_reward": COLORS["success"],
		"ctrl_reward": COLORS["info"],
		"power_loss_reward": DEVICE_COLORS["objective_cost"],
		"cost_voltage": DEVICE_COLORS["constraint_violation"],
		"lagrangian_penalty": DEVICE_COLORS["lagrangian_lambda"],
		"cap_penalty": DEVICE_COLORS["capacitor_on"],
		"reg_penalty": DEVICE_COLORS["regulator_tap"],
		"soc_penalty": DEVICE_COLORS["battery_soc"],
	}

	for comp_name, series in component_series.items():
		if not series or all(v == 0 for v in series):
			continue
		color = component_colors.get(comp_name, COLORS["text_secondary"])
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(series)],
				y=series,
				mode="lines",
				name=comp_name,
				line={"color": color, "width": 1.5},
				hovertemplate=f"{comp_name}<br>%{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
			),
			row=1, col=1,
		)

	# Row 2: 累计奖励
	step_rewards = _extract_step_rewards(snapshots)
	if step_rewards:
		cumulative = np.cumsum(step_rewards).tolist()
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(step_rewards)],
				y=step_rewards,
				mode="lines",
				name="Step Reward",
				line={"color": COLORS["primary"], "width": 1.5},
				hovertemplate="Step Reward<br>%{x}<br>R = %{y:.4f}<extra></extra>",
			),
			row=2, col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(cumulative)],
				y=cumulative,
				mode="lines",
				name="Cumulative",
				line={"color": COLORS["success"], "width": 2},
				hovertemplate="Cumulative<br>%{x}<br>Sum = %{y:.2f}<extra></extra>",
			),
			row=2, col=1,
		)

	layout = get_plotly_layout(
		title="CMDP Reward Breakdown",
		height=height,
		env_name="smartgrid",
	)
	layout.pop("xaxis", None)
	layout.pop("yaxis", None)
	fig.update_layout(**layout)
	fig.update_xaxes(title_text="Day of Year", row=2, col=1)
	fig.update_yaxes(title_text="Component Value", row=1, col=1)
	fig.update_yaxes(title_text="Reward", row=2, col=1)

	return fig


def _extract_reward_series(
	snapshots: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
	"""从快照中提取各奖励组件时间序列"""
	series: Dict[str, List[float]] = {}

	for snap in snapshots:
		rc = snap.get("reward_components", {})
		for key, val in rc.items():
			if key not in series:
				series[key] = []
			series[key].append(float(val) if isinstance(val, (int, float)) else 0.0)

	return series


def _extract_step_rewards(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步总奖励"""
	rewards: List[float] = []
	for snap in snapshots:
		r = snap.get("step_reward", 0.0)
		if r == 0.0:
			raw = snap.get("rewards")
			if isinstance(raw, list) and raw:
				r = float(sum(v for v in raw if isinstance(v, (int, float))))
			elif isinstance(raw, (int, float)):
				r = float(raw)
		rewards.append(r)
	return rewards
