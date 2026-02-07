"""
DSR Restoration Progress (Unique)
恢复进度追踪图 -- DSR 环境独有

X 轴: Step
主 Y 轴: 恢复百分比 (0-100%)
次 Y 轴: 恢复的负荷 kW
额外轨迹: 按优先级分类的恢复进度
标注: 每步恢复的线路/负荷
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from envs.render_common.utils.color_scales import restoration_to_color
from envs.render_common.viz.theme import (
	COLORS, DEVICE_COLORS, RESTORATION_COLORSCALE, get_plotly_layout,
)


# 优先级颜色
PRIORITY_COLORS = {
	"critical": DEVICE_COLORS["priority_critical"],
	"high": DEVICE_COLORS["priority_high"],
	"medium": DEVICE_COLORS["priority_medium"],
	"low": DEVICE_COLORS["priority_low"],
	"none": "#6B7280",
}


def create_restoration_progress(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建恢复进度追踪图

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure 对象
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Restoration Progress (No Data)", env_name="dsr"))
		return fig

	steps: List[int] = []
	restoration_pcts: List[float] = []
	restored_kws: List[float] = []
	n_energized: List[int] = []
	n_faults: List[int] = []

	# 按优先级分类的恢复数据
	priority_pcts: Dict[str, List[float]] = {
		"critical": [], "high": [], "medium": [], "low": [],
	}

	# 事件标注
	annotations_data: List[Dict[str, Any]] = []

	prev_energized: set = set()
	prev_restored_names: set = set()

	for snap in snapshots:
		step = snap.get("step", 0)
		steps.append(step)

		restoration = snap.get("restoration_data", {})
		pct = restoration.get("restoration_pct", 0.0)
		restoration_pcts.append(pct)
		restored_kws.append(restoration.get("total_restored_kw", 0.0))

		energized_buses = set(restoration.get("energized_buses", []))
		n_energized.append(len(energized_buses))
		n_faults.append(len(restoration.get("fault_lines", [])))

		# 优先级分解
		breakdown = restoration.get("priority_breakdown", {})
		for priority in priority_pcts:
			stats = breakdown.get(priority, {})
			priority_pcts[priority].append(stats.get("pct", 0.0))

		# 检测恢复事件
		current_restored = set(
			l.get("name", "") for l in restoration.get("restored_loads", [])
		)
		new_energized = energized_buses - prev_energized
		new_restored = current_restored - prev_restored_names

		if step > 0 and (new_energized or new_restored):
			events: List[str] = []
			if new_energized:
				events.append(f"+{len(new_energized)} buses")
			if new_restored:
				events.append(f"+{len(new_restored)} loads")
			annotations_data.append({
				"step": step,
				"pct": pct,
				"text": ", ".join(events),
			})

		prev_energized = energized_buses
		prev_restored_names = current_restored

	# 创建双 Y 轴图
	from plotly.subplots import make_subplots
	fig = make_subplots(
		specs=[[{"secondary_y": True}]],
	)

	# 主轴: 恢复百分比
	# 进度条着色
	marker_colors = [restoration_to_color(p / 100.0) for p in restoration_pcts]

	fig.add_trace(go.Scatter(
		x=steps, y=restoration_pcts,
		mode="lines+markers",
		name="Restoration %",
		line=dict(color=COLORS["success"], width=3),
		marker=dict(size=10, color=marker_colors, line=dict(width=1, color="white")),
		fill="tozeroy",
		fillcolor="rgba(16,185,129,0.1)",
		hovertemplate="Step: %{x}<br>Restored: %{y:.1f}%<extra></extra>",
	), secondary_y=False)

	# 次轴: 恢复功率 kW
	fig.add_trace(go.Scatter(
		x=steps, y=restored_kws,
		mode="lines+markers",
		name="Restored kW",
		line=dict(color=COLORS["info"], width=2, dash="dash"),
		marker=dict(size=6, symbol="diamond"),
		hovertemplate="Step: %{x}<br>Power: %{y:.1f} kW<extra></extra>",
	), secondary_y=True)

	# 按优先级分类的恢复进度
	for priority, pcts in priority_pcts.items():
		if all(p == 0.0 for p in pcts):
			continue
		fig.add_trace(go.Scatter(
			x=steps, y=pcts,
			mode="lines",
			name=f"{priority.title()} Priority",
			line=dict(
				color=PRIORITY_COLORS.get(priority, "#6B7280"),
				width=1.5,
				dash="dot",
			),
			hovertemplate=f"{priority.title()} Priority<br>Step: %{{x}}<br>Restored: %{{y:.1f}}%<extra></extra>",
		), secondary_y=False)

	# 恢复事件标注
	for ann in annotations_data:
		fig.add_annotation(
			x=ann["step"], y=ann["pct"],
			text=ann["text"],
			showarrow=True,
			arrowhead=2,
			arrowsize=0.8,
			font=dict(size=9, color=COLORS["success"]),
			bgcolor="rgba(16,185,129,0.15)",
			borderpad=2,
		)

	# 100% 完成线
	fig.add_hline(
		y=100, line_dash="dash",
		line_color=COLORS["success"], opacity=0.5,
		annotation_text="100% Restored",
		annotation_position="top left",
		secondary_y=False,
	)

	# 故障数标注
	if n_faults and n_faults[0] > 0:
		fig.add_annotation(
			text=f"Faults: {n_faults[0]}",
			xref="paper", yref="paper",
			x=0.02, y=0.98,
			showarrow=False,
			font=dict(size=12, color=COLORS["danger"]),
			bgcolor="rgba(239,68,68,0.15)",
			borderpad=4,
		)

	layout = get_plotly_layout(
		title="Restoration Progress",
		height=500,
		env_name="dsr",
	)

	fig.update_layout(**layout)
	fig.update_yaxes(title_text="Restoration (%)", range=[0, 110], secondary_y=False)
	fig.update_yaxes(title_text="Restored Power (kW)", secondary_y=True)
	fig.update_xaxes(title_text="Step", dtick=1)

	return fig
