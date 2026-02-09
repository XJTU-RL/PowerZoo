"""
SmartGrid Lagrangian Trajectory (Plotly) -- SmartGrid 独有
拉格朗日乘子轨迹图

双 Y 轴:
- 左 Y 轴: Lambda (拉格朗日乘子) 值
- 右 Y 轴: 约束违反量 (constraint violation)
- X 轴: Day of Year (0-360)

用于分析 CMDP 框架下 lambda 的收敛行为和约束满足情况。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import get_plotly_layout, COLORS, DEVICE_COLORS

logger = logging.getLogger(__name__)


def create_lagrangian_trajectory(
	snapshots: List[Dict[str, Any]],
	lambda_history: Optional[List[float]] = None,
	target_cost: float = 0.01,
	height: int = 500,
) -> go.Figure:
	"""创建 Lagrangian 乘子收敛轨迹图

	双 Y 轴设计:
	- 左轴 (主): Lambda 值随时间变化
	- 右轴 (辅): 约束违反量（电压违规率）

	Args:
		snapshots: 快照列表
		lambda_history: 外部传入的 lambda 历史（可选，来自 env 跨 episode）
		target_cost: 目标约束值
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	if not snapshots:
		layout = get_plotly_layout(
			title="Lagrangian Trajectory", height=height, env_name="smartgrid",
		)
		fig.update_layout(**layout)
		return fig

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	x_labels = [f"Day {s + 1}" for s in steps]

	# 从快照 info 中提取 lambda 和约束违反
	lambda_from_info: List[float] = []
	cost_voltage: List[float] = []
	violation_rate: List[float] = []

	for snap in snapshots:
		info = snap.get("info", {})
		rc = snap.get("reward_components", {})
		merged = {**rc, **info}

		lmbda = merged.get("lambda")
		if lmbda is not None:
			lambda_from_info.append(float(lmbda))

		cost = merged.get("cost_voltage")
		if cost is not None:
			cost_voltage.append(float(cost))

		viol = merged.get("voltage_violation_rate_buses",
				  merged.get("voltage_violation_rate", None))
		if viol is not None:
			violation_rate.append(float(viol))

	# 优先使用外部 lambda_history (跨 episode)
	lambda_vals = lambda_history if lambda_history else lambda_from_info

	# 左 Y 轴: Lambda
	if lambda_vals:
		# Lambda 可能是逐 episode 更新的，x 轴需要适配
		if len(lambda_vals) != len(x_labels):
			lambda_x = [f"Update {i + 1}" for i in range(len(lambda_vals))]
		else:
			lambda_x = x_labels

		fig.add_trace(
			go.Scatter(
				x=lambda_x,
				y=lambda_vals,
				mode="lines+markers",
				name="Lambda (λ)",
				line={"color": DEVICE_COLORS["lagrangian_lambda"], "width": 2.5},
				marker={"size": 4},
				hovertemplate="Lambda<br>%{x}<br>λ = %{y:.4f}<extra></extra>",
			),
			secondary_y=False,
		)

	# 右 Y 轴: 约束违反量
	if cost_voltage:
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(cost_voltage)],
				y=cost_voltage,
				mode="lines",
				name="Cost (Voltage)",
				line={"color": DEVICE_COLORS["constraint_violation"], "width": 1.5},
				hovertemplate="Cost<br>%{x}<br>C = %{y:.4f}<extra></extra>",
			),
			secondary_y=True,
		)

	if violation_rate:
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(violation_rate)],
				y=violation_rate,
				mode="lines",
				name="Violation Rate",
				line={"color": COLORS["warning"], "width": 1.5, "dash": "dot"},
				hovertemplate="Violation<br>%{x}<br>Rate = %{y:.4f}<extra></extra>",
			),
			secondary_y=True,
		)

	# 目标成本参考线
	if cost_voltage:
		fig.add_hline(
			y=target_cost,
			line_dash="dash",
			line_color=COLORS["success"],
			annotation_text=f"Target = {target_cost}",
			secondary_y=True,
		)

	# 布局
	layout = get_plotly_layout(
		title="Lagrangian Trajectory (CMDP)",
		height=height,
		env_name="smartgrid",
	)
	layout.pop("yaxis", None)
	fig.update_layout(**layout)

	fig.update_xaxes(title_text="Day of Year")
	fig.update_yaxes(
		title_text="Lambda (λ)",
		secondary_y=False,
		titlefont_color=DEVICE_COLORS["lagrangian_lambda"],
	)
	fig.update_yaxes(
		title_text="Constraint Violation",
		secondary_y=True,
		titlefont_color=DEVICE_COLORS["constraint_violation"],
	)

	return fig


def create_lambda_convergence_summary(
	lambda_history: List[float],
	cost_history: List[float],
	target_cost: float = 0.01,
	height: int = 400,
) -> go.Figure:
	"""创建 Lambda 收敛汇总图（跨 episode 视角）

	Args:
		lambda_history: 各 episode 结束时的 lambda 值
		cost_history: 各 episode 的平均约束违反成本
		target_cost: 目标成本
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	episodes = list(range(1, len(lambda_history) + 1))

	if lambda_history:
		fig.add_trace(
			go.Scatter(
				x=episodes,
				y=lambda_history,
				mode="lines+markers",
				name="Lambda",
				line={"color": DEVICE_COLORS["lagrangian_lambda"], "width": 2},
				marker={"size": 5},
			),
			secondary_y=False,
		)

	if cost_history:
		fig.add_trace(
			go.Scatter(
				x=episodes[:len(cost_history)],
				y=cost_history,
				mode="lines+markers",
				name="Avg Cost",
				line={"color": DEVICE_COLORS["constraint_violation"], "width": 2},
				marker={"size": 5},
			),
			secondary_y=True,
		)

		fig.add_hline(
			y=target_cost,
			line_dash="dash",
			line_color=COLORS["success"],
			annotation_text=f"Target = {target_cost}",
			secondary_y=True,
		)

	layout = get_plotly_layout(
		title="Lambda Convergence (Cross-Episode)",
		height=height,
		env_name="smartgrid",
	)
	layout.pop("yaxis", None)
	fig.update_layout(**layout)

	fig.update_xaxes(title_text="Episode")
	fig.update_yaxes(title_text="Lambda", secondary_y=False)
	fig.update_yaxes(title_text="Avg Constraint Cost", secondary_y=True)

	return fig
