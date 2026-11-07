# -*- coding: utf-8 -*-
"""
电压分布图

展示指定时间步所有母线的电压标幺值，
包含 0.95/1.05 安全限值标注。
"""

from typing import Any, Dict, List

import plotly.graph_objects as go

from envs.render_common.utils.color_scales import voltage_to_color
from envs.render_common.viz.theme import get_plotly_layout


def create_voltage_profile(
	snapshot: Dict[str, Any],
) -> go.Figure:
	"""创建电压分布柱状图

	Args:
		snapshot: 单时间步快照

	Returns:
		Plotly 柱状图
	"""
	buses = snapshot.get("buses", {})
	step = snapshot.get("step", 0)

	if not buses:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Voltage Profile (No Data)", env_name="stackelberg"
		))
		return fig

	bus_names: List[str] = []
	v_values: List[float] = []
	colors: List[str] = []

	for bus_name in sorted(buses.keys()):
		bus_data = buses[bus_name]
		v_pu = bus_data.get("v_mag_pu", [1.0])
		if isinstance(v_pu, list) and v_pu:
			avg_v = sum(v_pu) / len(v_pu)
		else:
			avg_v = float(v_pu) if isinstance(v_pu, (int, float)) else 1.0

		bus_names.append(bus_name)
		v_values.append(avg_v)
		colors.append(voltage_to_color(avg_v))

	fig = go.Figure()

	fig.add_trace(go.Bar(
		x=bus_names,
		y=v_values,
		marker_color=colors,
		hovertemplate="Bus: %{x}<br>Vpu: %{y:.4f}<extra></extra>",
	))

	# 安全限值
	fig.add_hline(
		y=0.95, line_dash="dash", line_color="#EF4444",
		annotation_text="V_min=0.95",
		annotation_position="bottom right",
		annotation_font=dict(color="#EF4444", size=10),
	)
	fig.add_hline(
		y=1.05, line_dash="dash", line_color="#EF4444",
		annotation_text="V_max=1.05",
		annotation_position="top right",
		annotation_font=dict(color="#EF4444", size=10),
	)
	fig.add_hline(
		y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
	)

	layout = get_plotly_layout(
		title=f"Voltage Profile (Step {step})",
		height=450,
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(title="Bus", tickangle=-45),
		yaxis=dict(title="Voltage (pu)", range=[0.90, 1.10]),
	)
	fig.update_layout(**layout)

	return fig


def create_voltage_timeseries(
	snapshots: List[Dict[str, Any]],
	bus_names: List[str] | None = None,
) -> go.Figure:
	"""创建电压时间序列折线图

	Args:
		snapshots: 快照列表
		bus_names: 要显示的母线名称列表 (None 显示电压摘要)

	Returns:
		Plotly 折线图
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Voltage Timeseries (No Data)", env_name="stackelberg"
		))
		return fig

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	fig = go.Figure()

	if bus_names:
		# 按母线显示
		for bus_name in bus_names:
			v_series = []
			for snap in snapshots:
				buses = snap.get("buses", {})
				bus_data = buses.get(bus_name, {})
				v_pu = bus_data.get("v_mag_pu", [1.0])
				if isinstance(v_pu, list) and v_pu:
					v_series.append(sum(v_pu) / len(v_pu))
				else:
					v_series.append(float(v_pu) if isinstance(v_pu, (int, float)) else 1.0)

			fig.add_trace(go.Scatter(
				x=steps,
				y=v_series,
				mode="lines+markers",
				name=bus_name,
				hovertemplate=f"Bus {bus_name}<br>Step: %{{x}}<br>Vpu: %{{y:.4f}}<extra></extra>",
			))
	else:
		# 显示 V_min / V_mean / V_max
		v_min_series = []
		v_mean_series = []
		v_max_series = []
		for snap in snapshots:
			vs = snap.get("voltage_summary", {})
			v_min_series.append(vs.get("v_min", 1.0))
			v_mean_series.append(vs.get("v_mean", 1.0))
			v_max_series.append(vs.get("v_max", 1.0))

		fig.add_trace(go.Scatter(
			x=steps, y=v_min_series,
			mode="lines", name="V_min",
			line=dict(color="#EF4444", width=1, dash="dash"),
		))
		fig.add_trace(go.Scatter(
			x=steps, y=v_mean_series,
			mode="lines+markers", name="V_mean",
			line=dict(color="#F59E0B", width=2),
		))
		fig.add_trace(go.Scatter(
			x=steps, y=v_max_series,
			mode="lines", name="V_max",
			line=dict(color="#3B82F6", width=1, dash="dash"),
		))

	# 安全限值
	fig.add_hline(y=0.95, line_dash="dot", line_color="rgba(239,68,68,0.5)")
	fig.add_hline(y=1.05, line_dash="dot", line_color="rgba(239,68,68,0.5)")

	layout = get_plotly_layout(
		title="Voltage Timeseries",
		height=400,
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(title="Step"),
		yaxis=dict(title="Voltage (pu)"),
	)
	fig.update_layout(**layout)

	return fig
