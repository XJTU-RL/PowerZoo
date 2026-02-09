# -*- coding: utf-8 -*-
"""
VVC 电压热图 (Plotly)

以时间步为 X 轴、母线为 Y 轴的热图，
色标表示各母线各时间步的电压标幺值，安全区间 [0.95, 1.05] 外高亮。
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from envs.render_common.viz.theme import VOLTAGE_COLORSCALE, get_plotly_layout


def create_voltage_heatmap(
	snapshots: List[Dict[str, Any]],
	bus_names: Optional[List[str]] = None,
) -> go.Figure:
	"""创建母线电压热图。

	Args:
		snapshots: 快照列表
		bus_names: 指定母线名称列表 (None 则自动提取)

	Returns:
		Plotly Figure 对象
	"""
	# 收集所有母线名称
	if bus_names is None:
		name_set: set = set()
		for snap in snapshots:
			buses = snap.get("buses", {})
			name_set.update(buses.keys())
		bus_names = sorted(name_set)

	if not bus_names or not snapshots:
		fig = go.Figure()
		layout = get_plotly_layout(
			title="Voltage Heatmap (No Data)", height=400, env_name="vvc"
		)
		fig.update_layout(**layout)
		return fig

	# 构建矩阵: rows=buses, cols=steps
	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	n_steps = len(steps)
	n_buses = len(bus_names)

	z_matrix: List[List[float]] = []

	for bus_name in bus_names:
		row: List[float] = []
		for snap in snapshots:
			buses = snap.get("buses", {})
			bd = buses.get(bus_name, {})
			v_pu = bd.get("v_mag_pu", [1.0])
			avg_v = sum(v_pu) / len(v_pu) if v_pu else 1.0
			row.append(avg_v)
		z_matrix.append(row)

	# 将电压值归一化到 [0, 1] 以映射到色标
	# 色标: 0.90 -> 0.0, 1.00 -> 0.5, 1.10 -> 1.0
	z_normalized: List[List[float]] = []
	for row in z_matrix:
		z_normalized.append([(v - 0.90) / 0.20 for v in row])

	fig = go.Figure(data=go.Heatmap(
		z=z_matrix,
		x=steps,
		y=bus_names,
		colorscale=VOLTAGE_COLORSCALE,
		zmin=0.90,
		zmax=1.10,
		colorbar=dict(
			title="V (pu)",
			titleside="right",
			tickvals=[0.90, 0.95, 1.00, 1.05, 1.10],
			ticktext=["0.90", "0.95", "1.00", "1.05", "1.10"],
		),
		hovertemplate=(
			"Bus: %{y}<br>"
			"Step: %{x}<br>"
			"Voltage: %{z:.4f} pu<br>"
			"<extra></extra>"
		),
	))

	# 安全区间参考线
	fig.add_hline(y=-0.5, line_dash="dot", opacity=0)  # 占位

	layout = get_plotly_layout(
		title="Bus Voltage Heatmap",
		height=max(400, n_buses * 20 + 100),
		env_name="vvc",
	)
	layout.update({
		"xaxis_title": "Step",
		"yaxis_title": "Bus",
		"yaxis": dict(
			autorange="reversed",
			dtick=1,
			gridcolor="rgba(255,255,255,0.05)",
		),
	})
	fig.update_layout(**layout)

	return fig
