"""
DSR Network State Chart (Unique)
网络状态分布图 -- DSR 环境独有

两个子图:
1. 堆叠面积图: energized / de-energized / faulted 母线数量随步骤变化
2. 饼图: 优先级负荷分布 (critical / high / medium / low) 及恢复状态
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import COLORS, DEVICE_COLORS, get_plotly_layout


# 母线状态颜色
STATE_COLORS = {
	"energized": COLORS["success"],
	"de_energized": "#6B7280",
	"faulted": COLORS["danger"],
}

# 优先级颜色
PRIORITY_COLORS = {
	"critical": DEVICE_COLORS["priority_critical"],
	"high": DEVICE_COLORS["priority_high"],
	"medium": DEVICE_COLORS["priority_medium"],
	"low": DEVICE_COLORS["priority_low"],
	"none": "#6B7280",
}


def create_network_state_chart(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建网络状态分布图

	左侧: 堆叠面积图展示母线通电/断电/故障状态随时间变化
	右侧: 饼图展示最终步骤的优先级负荷恢复分布

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure 对象
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Network State (No Data)", env_name="dsr"))
		return fig

	fig = make_subplots(
		rows=1, cols=2,
		column_widths=[0.6, 0.4],
		subplot_titles=["Bus State Distribution", "Priority Load Breakdown"],
		specs=[[{"type": "scatter"}, {"type": "pie"}]],
	)

	# --- 左侧: 堆叠面积图 ---
	steps: List[int] = []
	n_energized: List[int] = []
	n_deenergized: List[int] = []
	n_faulted_buses: List[int] = []

	for snap in snapshots:
		step = snap.get("step", 0)
		steps.append(step)

		restoration = snap.get("restoration_data", {})
		n_en = len(restoration.get("energized_buses", []))
		n_de = len(restoration.get("de_energized_buses", []))
		n_faults = len(restoration.get("fault_lines", []))

		n_energized.append(n_en)
		n_deenergized.append(n_de)
		n_faulted_buses.append(n_faults)

	# Energized (绿色，底部)
	fig.add_trace(go.Scatter(
		x=steps, y=n_energized,
		mode="lines",
		name="Energized",
		line=dict(width=0),
		fillcolor=_rgba(STATE_COLORS["energized"], 0.6),
		fill="tozeroy",
		stackgroup="buses",
		hovertemplate="Step: %{x}<br>Energized: %{y}<extra></extra>",
	), row=1, col=1)

	# De-energized (灰色，中间)
	fig.add_trace(go.Scatter(
		x=steps, y=n_deenergized,
		mode="lines",
		name="De-energized",
		line=dict(width=0),
		fillcolor=_rgba(STATE_COLORS["de_energized"], 0.6),
		fill="tonexty",
		stackgroup="buses",
		hovertemplate="Step: %{x}<br>De-energized: %{y}<extra></extra>",
	), row=1, col=1)

	# Faulted 标注 (不堆叠，单独线条)
	if any(n > 0 for n in n_faulted_buses):
		fig.add_trace(go.Scatter(
			x=steps, y=n_faulted_buses,
			mode="lines+markers",
			name="Fault Lines",
			line=dict(color=STATE_COLORS["faulted"], width=2, dash="dot"),
			marker=dict(size=6, symbol="x"),
			hovertemplate="Step: %{x}<br>Faults: %{y}<extra></extra>",
		), row=1, col=1)

	fig.update_xaxes(title_text="Step", dtick=1, row=1, col=1)
	fig.update_yaxes(title_text="Count", row=1, col=1)

	# --- 右侧: 优先级负荷饼图 ---
	final_snap = snapshots[-1]
	restoration = final_snap.get("restoration_data", {})
	breakdown = restoration.get("priority_breakdown", {})

	pie_labels: List[str] = []
	pie_values: List[float] = []
	pie_colors: List[str] = []
	pie_text: List[str] = []

	for priority in ["critical", "high", "medium", "low"]:
		stats = breakdown.get(priority, {})
		total_kw = stats.get("total_kw", 0.0)
		restored_kw = stats.get("restored_kw", 0.0)
		total_count = stats.get("total_count", 0)
		restored_count = stats.get("restored_count", 0)

		if total_count == 0:
			continue

		# 已恢复部分
		if restored_kw > 0:
			pie_labels.append(f"{priority.title()} (Restored)")
			pie_values.append(restored_kw)
			pie_colors.append(PRIORITY_COLORS.get(priority, "#6B7280"))
			pie_text.append(f"{restored_count}/{total_count} loads<br>{restored_kw:.0f} kW")

		# 未恢复部分
		unrestored_kw = total_kw - restored_kw
		if unrestored_kw > 0:
			pie_labels.append(f"{priority.title()} (Pending)")
			pie_values.append(unrestored_kw)
			# 半透明版本
			pie_colors.append(_rgba(PRIORITY_COLORS.get(priority, "#6B7280"), 0.3))
			unrestored_count = total_count - restored_count
			pie_text.append(f"{unrestored_count} pending<br>{unrestored_kw:.0f} kW")

	if pie_values:
		fig.add_trace(go.Pie(
			labels=pie_labels,
			values=pie_values,
			marker=dict(colors=pie_colors),
			textinfo="label+percent",
			textposition="inside",
			hoverinfo="text",
			hovertext=pie_text,
			hole=0.3,
		), row=1, col=2)
	else:
		# 无数据时的占位
		fig.add_trace(go.Pie(
			labels=["No Priority Data"],
			values=[1],
			marker=dict(colors=["#3a3a5a"]),
			textinfo="label",
			hole=0.3,
		), row=1, col=2)

	layout = get_plotly_layout(
		title="Network State Distribution",
		height=450,
		env_name="dsr",
	)
	fig.update_layout(**layout)

	return fig


def _rgba(hex_color: str, alpha: float) -> str:
	"""将 hex 颜色转换为 rgba 字符串

	Args:
		hex_color: '#RRGGBB' 格式
		alpha: 透明度 0-1

	Returns:
		'rgba(r,g,b,a)' 字符串
	"""
	h = hex_color.lstrip("#")
	if len(h) == 6:
		r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
		return f"rgba({r},{g},{b},{alpha})"
	return hex_color
