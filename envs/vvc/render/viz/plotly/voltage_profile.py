# -*- coding: utf-8 -*-
"""
VVC 电压剖面图 (Plotly)

单时间步各母线电压条形图，标注安全区间 [0.95, 1.05] pu。
"""

from typing import Any, Dict, List, Optional, Union

import plotly.graph_objects as go

from envs.render_common.utils.color_scales import voltage_to_color
from envs.render_common.viz.theme import COLORS, get_plotly_layout


def create_voltage_profile(
	snapshot_or_bus_data: Union[Dict[str, Dict[str, Any]], Dict[str, Any]],
	step: Optional[int] = None,
) -> go.Figure:
	"""创建单步电压剖面条形图。

	Args:
		snapshot_or_bus_data: 快照字典或直接的 bus_data 字典
		step: 时间步编号 (用于标题)

	Returns:
		Plotly Figure 对象
	"""
	# 兼容两种输入格式
	if "buses" in snapshot_or_bus_data:
		bus_data = snapshot_or_bus_data["buses"]
		if step is None:
			step = snapshot_or_bus_data.get("step")
	else:
		bus_data = snapshot_or_bus_data

	if not bus_data:
		fig = go.Figure()
		layout = get_plotly_layout(
			title="Voltage Profile (No Data)", height=400, env_name="vvc"
		)
		fig.update_layout(**layout)
		return fig

	# 排序母线名并计算平均电压
	bus_names = sorted(bus_data.keys())
	avg_voltages: List[float] = []
	bar_colors: List[str] = []
	hover_texts: List[str] = []

	for bus_name in bus_names:
		bd = bus_data[bus_name]
		v_pu = bd.get("v_mag_pu", [1.0])
		avg_v = sum(v_pu) / len(v_pu) if v_pu else 1.0
		avg_voltages.append(avg_v)
		bar_colors.append(voltage_to_color(avg_v))

		phases_str = ", ".join(f"{v:.4f}" for v in v_pu)
		hover_texts.append(
			f"Bus: {bus_name}<br>"
			f"V_avg: {avg_v:.4f} pu<br>"
			f"Phases: [{phases_str}]<br>"
			f"Base kV: {bd.get('base_kv', 0):.2f}"
		)

	fig = go.Figure()

	# 电压条形图
	fig.add_trace(go.Bar(
		x=bus_names,
		y=avg_voltages,
		marker_color=bar_colors,
		hoverinfo="text",
		hovertext=hover_texts,
		name="Voltage",
	))

	# 安全区间
	fig.add_hline(
		y=1.05, line_dash="dash",
		line_color=COLORS["warning"],
		annotation_text="1.05 pu (upper)",
		annotation_position="top right",
	)
	fig.add_hline(
		y=0.95, line_dash="dash",
		line_color=COLORS["warning"],
		annotation_text="0.95 pu (lower)",
		annotation_position="bottom right",
	)
	fig.add_hline(
		y=1.00, line_dash="dot",
		line_color=COLORS["text_secondary"],
		opacity=0.5,
	)

	# 安全区间填充
	fig.add_hrect(
		y0=0.95, y1=1.05,
		fillcolor=COLORS["success"],
		opacity=0.08,
		line_width=0,
	)

	title = "Voltage Profile"
	if step is not None:
		title += f" (Step {step})"

	layout = get_plotly_layout(title=title, height=400, env_name="vvc")
	layout.update({
		"xaxis_title": "Bus",
		"yaxis_title": "Voltage (pu)",
		"yaxis_range": [0.88, 1.12],
		"bargap": 0.2,
	})
	fig.update_layout(**layout)

	return fig
