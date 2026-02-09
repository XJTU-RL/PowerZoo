# -*- coding: utf-8 -*-
"""
VVC 拓扑图 (Plotly)

绘制馈线网络拓扑图，节点按电压着色，线路按负载率着色，
叠加电容器/调压器/电池/PV 设备标记。
"""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

from envs.render_common.utils.color_scales import (
	loading_to_color,
	voltage_to_color,
)
from envs.render_common.viz.theme import COLORS, DEVICE_COLORS, get_plotly_layout


def create_topology_figure(
	bus_data: Dict[str, Dict[str, Any]],
	line_data: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	device_data: Optional[Dict[str, Dict[str, Any]]] = None,
	step: Optional[int] = None,
) -> go.Figure:
	"""创建 VVC 馈线拓扑图。

	Args:
		bus_data: {bus_name: {v_mag_pu, ...}} 母线数据
		line_data: {line_name: {bus1, bus2, loading_pct, ...}} 线路数据
		bus_coords: {bus_name: (x, y)} 坐标
		device_data: {capacitors: {...}, regulators: {...}, ...} 设备数据
		step: 当前时间步 (显示在标题中)

	Returns:
		Plotly Figure 对象
	"""
	fig = go.Figure()

	# --- 线路 (边) ---
	for line_name, ld in line_data.items():
		bus1 = ld.get("bus1", "")
		bus2 = ld.get("bus2", "")

		if bus1 not in bus_coords or bus2 not in bus_coords:
			continue

		x1, y1 = bus_coords[bus1]
		x2, y2 = bus_coords[bus2]

		loading = ld.get("loading_pct", 0.0)
		line_color = loading_to_color(loading)
		width = max(1.0, min(loading / 25.0, 5.0))

		fig.add_trace(go.Scatter(
			x=[x1, x2, None],
			y=[y1, y2, None],
			mode="lines",
			line=dict(color=line_color, width=width),
			hoverinfo="text",
			text=f"Line: {line_name}<br>Loading: {loading:.1f}%<br>Loss: {ld.get('losses_kw', 0):.2f} kW",
			showlegend=False,
		))

	# --- 母线节点 ---
	bus_x: List[float] = []
	bus_y: List[float] = []
	bus_colors: List[str] = []
	bus_sizes: List[int] = []
	bus_texts: List[str] = []

	for bus_name, coords in bus_coords.items():
		bus_x.append(coords[0])
		bus_y.append(coords[1])

		bd = bus_data.get(bus_name, {})
		v_pu = bd.get("v_mag_pu", [1.0])
		avg_v = sum(v_pu) / len(v_pu) if v_pu else 1.0

		bus_colors.append(voltage_to_color(avg_v))
		bus_sizes.append(12)

		n_phases = bd.get("n_phases", 0)
		bus_texts.append(
			f"Bus: {bus_name}<br>"
			f"V_avg: {avg_v:.4f} pu<br>"
			f"Phases: {n_phases}"
		)

	fig.add_trace(go.Scatter(
		x=bus_x,
		y=bus_y,
		mode="markers+text",
		marker=dict(
			size=bus_sizes,
			color=bus_colors,
			line=dict(width=1, color=COLORS["border"]),
		),
		text=[name for name in bus_coords],
		textposition="top center",
		textfont=dict(size=8, color=COLORS["text_secondary"]),
		hoverinfo="text",
		hovertext=bus_texts,
		name="Buses",
		showlegend=False,
	))

	# --- 设备标记 ---
	if device_data:
		_add_device_markers(fig, device_data, bus_coords)

	# --- 布局 ---
	title = "VVC Network Topology"
	if step is not None:
		title += f" (Step {step})"

	layout = get_plotly_layout(title=title, height=600, env_name="vvc")
	layout.update({
		"xaxis": dict(
			showgrid=False, zeroline=False, showticklabels=False,
			gridcolor=COLORS["grid"],
		),
		"yaxis": dict(
			showgrid=False, zeroline=False, showticklabels=False,
			scaleanchor="x", scaleratio=1,
			gridcolor=COLORS["grid"],
		),
		"hovermode": "closest",
	})
	fig.update_layout(**layout)

	return fig


def _add_device_markers(
	fig: go.Figure,
	device_data: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""在拓扑图上添加设备标记。

	Args:
		fig: Plotly Figure
		device_data: 设备数据字典
		bus_coords: 母线坐标
	"""
	# 电容器 (方形)
	caps = device_data.get("capacitors", {})
	if caps:
		cap_x, cap_y, cap_text = [], [], []
		cap_colors = []
		for name, cd in caps.items():
			bus = cd.get("bus", "")
			if bus in bus_coords:
				x, y = bus_coords[bus]
				cap_x.append(x + 0.3)
				cap_y.append(y + 0.3)
				is_on = cd.get("is_on", False)
				cap_colors.append(
					DEVICE_COLORS["capacitor_on"] if is_on
					else DEVICE_COLORS["capacitor_off"]
				)
				cap_text.append(
					f"Cap: {name}<br>"
					f"State: {'ON' if is_on else 'OFF'}<br>"
					f"kvar: {cd.get('kvar', 0):.0f}"
				)

		if cap_x:
			fig.add_trace(go.Scatter(
				x=cap_x, y=cap_y,
				mode="markers",
				marker=dict(
					size=14, color=cap_colors,
					symbol="square",
					line=dict(width=1, color="white"),
				),
				hoverinfo="text",
				hovertext=cap_text,
				name="Capacitors",
			))

	# 调压器 (三角形)
	regs = device_data.get("regulators", {})
	if regs:
		reg_x, reg_y, reg_text = [], [], []
		for name, rd in regs.items():
			bus = rd.get("bus", "")
			if bus in bus_coords:
				x, y = bus_coords[bus]
				reg_x.append(x - 0.3)
				reg_y.append(y + 0.3)
				reg_text.append(
					f"Reg: {name}<br>"
					f"Tap: {rd.get('tap', 0)}<br>"
					f"Vreg: {rd.get('forward_vreg', 0):.1f}V"
				)

		if reg_x:
			fig.add_trace(go.Scatter(
				x=reg_x, y=reg_y,
				mode="markers",
				marker=dict(
					size=14, color=DEVICE_COLORS["regulator_tap"],
					symbol="triangle-up",
					line=dict(width=1, color="white"),
				),
				hoverinfo="text",
				hovertext=reg_text,
				name="Regulators",
			))

	# 电池 (菱形)
	bats = device_data.get("batteries", {})
	if bats:
		bat_x, bat_y, bat_text = [], [], []
		bat_colors = []
		for name, bd in bats.items():
			bus = bd.get("bus", "")
			if bus in bus_coords:
				x, y = bus_coords[bus]
				bat_x.append(x + 0.3)
				bat_y.append(y - 0.3)
				kw = bd.get("kw", 0)
				bat_colors.append(
					DEVICE_COLORS["storage_discharge"] if kw > 0
					else DEVICE_COLORS["storage_charge"]
				)
				bat_text.append(
					f"Battery: {name}<br>"
					f"SOC: {bd.get('soc', 0):.1%}<br>"
					f"Power: {kw:.1f} kW<br>"
					f"State: {bd.get('state', 'IDLE')}"
				)

		if bat_x:
			fig.add_trace(go.Scatter(
				x=bat_x, y=bat_y,
				mode="markers",
				marker=dict(
					size=14, color=bat_colors,
					symbol="diamond",
					line=dict(width=1, color="white"),
				),
				hoverinfo="text",
				hovertext=bat_text,
				name="Batteries",
			))

	# PV (圆形/太阳)
	pvs = device_data.get("pvsystems", {})
	if pvs:
		pv_x, pv_y, pv_text = [], [], []
		for name, pd in pvs.items():
			bus = pd.get("bus", "")
			if bus in bus_coords:
				x, y = bus_coords[bus]
				pv_x.append(x - 0.3)
				pv_y.append(y - 0.3)
				pv_text.append(
					f"PV: {name}<br>"
					f"Output: {pd.get('kw', 0):.1f} kW<br>"
					f"Pmpp: {pd.get('pmpp', 0):.1f} kW<br>"
					f"Curtail: {pd.get('curtail_pct', 0):.1f}%"
				)

		if pv_x:
			fig.add_trace(go.Scatter(
				x=pv_x, y=pv_y,
				mode="markers",
				marker=dict(
					size=14, color=DEVICE_COLORS["pv"],
					symbol="circle",
					line=dict(width=1, color="white"),
				),
				hoverinfo="text",
				hovertext=pv_text,
				name="PV Systems",
			))
