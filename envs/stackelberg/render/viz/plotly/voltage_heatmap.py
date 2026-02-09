# -*- coding: utf-8 -*-
"""
电压热力图

以时间步为 X 轴、母线为 Y 轴，用色标展示系统电压分布。
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import get_plotly_layout


def create_voltage_heatmap(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建电压热力图

	Args:
		snapshots: 快照列表

	Returns:
		Plotly 热力图
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Voltage Heatmap (No Data)", env_name="stackelberg"
		))
		return fig

	# 收集所有母线名称
	bus_names: List[str] = []
	seen: set = set()
	for snap in snapshots:
		buses = snap.get("buses", {})
		for name in buses:
			if name not in seen:
				seen.add(name)
				bus_names.append(name)

	if not bus_names:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Voltage Heatmap (No Bus Data)", env_name="stackelberg"
		))
		return fig

	# 构建电压矩阵 (bus x step)
	n_buses = len(bus_names)
	n_steps = len(snapshots)
	v_matrix = np.ones((n_buses, n_steps))

	steps = []
	for j, snap in enumerate(snapshots):
		steps.append(snap.get("step", j))
		buses = snap.get("buses", {})
		for i, bus_name in enumerate(bus_names):
			bus_data = buses.get(bus_name, {})
			v_pu = bus_data.get("v_mag_pu", [1.0])
			if isinstance(v_pu, list) and v_pu:
				v_matrix[i, j] = sum(v_pu) / len(v_pu)
			elif isinstance(v_pu, (int, float)):
				v_matrix[i, j] = float(v_pu)

	# 电压色标
	colorscale = [
		[0.0, "#EF4444"],     # 低压 (红)
		[0.40, "#F59E0B"],    # 偏低 (琥珀)
		[0.45, "#10B981"],    # 正常下限
		[0.55, "#10B981"],    # 正常上限
		[0.60, "#F59E0B"],    # 偏高 (琥珀)
		[1.0, "#EF4444"],     # 高压 (红)
	]

	fig = go.Figure(data=go.Heatmap(
		z=v_matrix,
		x=[str(s) for s in steps],
		y=bus_names,
		colorscale=colorscale,
		zmin=0.90,
		zmax=1.10,
		colorbar=dict(
			title="Vpu",
			titleside="right",
			tickfont=dict(color="#e0e0e0"),
			titlefont=dict(color="#e0e0e0"),
		),
		hovertemplate=(
			"Bus: %{y}<br>"
			"Step: %{x}<br>"
			"Vpu: %{z:.4f}<extra></extra>"
		),
	))

	# 安全限值参考线
	for v_limit in [0.95, 1.05]:
		fig.add_hline(
			y=v_limit, line_dash="dot",
			line_color="rgba(255,255,255,0.3)",
			annotation_text=f"V={v_limit}",
			annotation_position="top right",
			annotation_font=dict(color="#a0a0a0", size=9),
		)

	layout = get_plotly_layout(
		title="Bus Voltage Heatmap",
		height=max(400, n_buses * 20 + 100),
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(title="Step"),
		yaxis=dict(title="Bus", autorange="reversed"),
	)
	fig.update_layout(**layout)

	return fig
