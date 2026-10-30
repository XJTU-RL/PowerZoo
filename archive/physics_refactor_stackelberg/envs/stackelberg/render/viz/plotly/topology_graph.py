# -*- coding: utf-8 -*-
"""
Stackelberg 电路拓扑交互式网络图

节点按电压着色，线路按负载率着色，
设备图标标注 (PV, ESS, Load, Transformer)。
支持 13Bus / 34Bus / 123Bus 系统。
"""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

from envs.render_common.utils.color_scales import loading_to_color, voltage_to_color
from envs.render_common.viz.theme import get_plotly_layout
from envs.stackelberg.render.assets.bus_coordinates import (
	load_bus_coordinates,
	load_topology_edges,
)

# 设备图标映射
DEVICE_ICONS: Dict[str, str] = {
	"pv": "\u2600",          # ☀
	"storage": "\u26a1",     # ⚡
	"load": "\U0001f3e0",    # 🏠
	"transformer": "\U0001f504",  # 🔄
}


def _get_bus_vpu_avg(bus_data: Dict[str, Any]) -> float:
	"""计算母线三相平均电压标幺值

	Args:
		bus_data: 母线数据字典

	Returns:
		三相平均电压标幺值
	"""
	v_mag_pu = bus_data.get("v_mag_pu", [1.0])
	if not v_mag_pu:
		return 1.0
	if isinstance(v_mag_pu, (int, float)):
		return float(v_mag_pu)
	return sum(v_mag_pu) / len(v_mag_pu)


def _build_bus_hover(bus_name: str, bus_data: Dict[str, Any]) -> str:
	"""构建母线 hover 文本

	Args:
		bus_name: 母线名称
		bus_data: 母线数据字典

	Returns:
		格式化的 hover 信息
	"""
	lines = [f"<b>Bus {bus_name}</b>"]

	v_mag_pu = bus_data.get("v_mag_pu", [])
	if v_mag_pu:
		if isinstance(v_mag_pu, list):
			vpu_str = ", ".join(f"{v:.4f}" for v in v_mag_pu)
			lines.append(f"Vpu: [{vpu_str}]")
		else:
			lines.append(f"Vpu: {v_mag_pu:.4f}")

	v_angle = bus_data.get("v_angle_deg", [])
	if v_angle and isinstance(v_angle, list):
		ang_str = ", ".join(f"{a:.1f}" for a in v_angle)
		lines.append(f"Angle: [{ang_str}] deg")

	n_phases = bus_data.get("n_phases", 0)
	if n_phases:
		lines.append(f"Phases: {n_phases}")

	return "<br>".join(lines)


def _build_line_hover(line_key: str, line_data: Dict[str, Any]) -> str:
	"""构建线路 hover 文本

	Args:
		line_key: 线路标识
		line_data: 线路数据字典

	Returns:
		格式化的 hover 信息
	"""
	name = line_data.get("name", line_key)
	lines = [f"<b>Line {name}</b>"]

	current_mag = line_data.get("current_mag", [])
	if current_mag:
		i_str = ", ".join(f"{i:.1f}" for i in current_mag)
		lines.append(f"I: [{i_str}] A")

	power_kw = line_data.get("power_kw", [])
	if power_kw:
		p_str = ", ".join(f"{p:.1f}" for p in power_kw)
		lines.append(f"P: [{p_str}] kW")

	losses = line_data.get("losses", {})
	if losses:
		lines.append(f"Losses: {losses.get('kw', 0):.2f} kW")

	loading = line_data.get("loading_pct", 0.0)
	lines.append(f"Loading: {loading:.1f}%")

	return "<br>".join(lines)


def create_topology_figure(
	snapshot: Dict[str, Any],
	bus_coords: Optional[Dict[str, Tuple[float, float]]] = None,
	system_name: str = "13Bus",
) -> go.Figure:
	"""创建电路拓扑交互式网络图

	节点按电压着色，线路按负载率着色，设备图标标注。

	Args:
		snapshot: 单时间步快照数据
		bus_coords: 母线坐标字典 (None 则自动加载)
		system_name: IEEE 系统名称

	Returns:
		Plotly 交互式网络图
	"""
	if bus_coords is None:
		bus_coords = load_bus_coordinates(system_name)
	topology_edges = load_topology_edges(system_name)

	fig = go.Figure()

	buses_data = snapshot.get("buses", {})
	lines_data = snapshot.get("lines", {})
	devices_data = snapshot.get("devices", {})

	# 1. 线路 (边)
	for from_bus, to_bus in topology_edges:
		if from_bus not in bus_coords or to_bus not in bus_coords:
			continue

		x0, y0 = bus_coords[from_bus]
		x1, y1 = bus_coords[to_bus]

		line_key = f"{from_bus}_{to_bus}"
		rev_key = f"{to_bus}_{from_bus}"
		line_data = lines_data.get(line_key, lines_data.get(rev_key, {}))
		loading = line_data.get("loading_pct", 0.0)
		line_color = loading_to_color(loading)
		hover_text = _build_line_hover(line_key, line_data)

		fig.add_trace(go.Scatter(
			x=[x0, x1, None],
			y=[y0, y1, None],
			mode="lines",
			line=dict(color=line_color, width=max(1, loading / 30)),
			hoverinfo="text",
			hovertext=hover_text,
			showlegend=False,
		))

	# 2. 母线节点
	node_x, node_y, node_colors, node_hovers, node_texts = [], [], [], [], []
	for bus_name, (x, y) in bus_coords.items():
		bus_data = buses_data.get(bus_name, buses_data.get(bus_name.lower(), {}))
		vpu_avg = _get_bus_vpu_avg(bus_data)
		color = voltage_to_color(vpu_avg)
		hover = _build_bus_hover(bus_name, bus_data)

		node_x.append(x)
		node_y.append(y)
		node_colors.append(color)
		node_hovers.append(hover)
		node_texts.append(bus_name)

	fig.add_trace(go.Scatter(
		x=node_x,
		y=node_y,
		mode="markers+text",
		marker=dict(
			size=14,
			color=node_colors,
			line=dict(width=1, color="rgba(255,255,255,0.3)"),
		),
		text=node_texts,
		textposition="top center",
		textfont=dict(size=8, color="#e0e0e0"),
		hoverinfo="text",
		hovertext=node_hovers,
		showlegend=False,
		name="Buses",
	))

	# 3. 设备图标
	_add_device_markers(fig, devices_data, bus_coords)

	# 4. Layout
	step = snapshot.get("step", 0)
	layout = get_plotly_layout(
		title=f"{system_name} Topology (Step {step})",
		height=700,
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(
			showgrid=False, zeroline=False, showticklabels=False,
			gridcolor="rgba(0,0,0,0)",
		),
		yaxis=dict(
			showgrid=False, zeroline=False, showticklabels=False,
			scaleanchor="x", scaleratio=1,
			gridcolor="rgba(0,0,0,0)",
		),
		hovermode="closest",
	)
	fig.update_layout(**layout)

	return fig


def _add_device_markers(
	fig: go.Figure,
	devices_data: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""添加设备图标到拓扑图

	Args:
		fig: Plotly Figure 对象
		devices_data: 设备数据字典
		bus_coords: 母线坐标字典
	"""
	offset_y = 0.025

	# PV 设备
	pv_list = devices_data.get("pv", [])
	for pv in pv_list:
		bus = pv.get("bus", "")
		if bus in bus_coords:
			x, y = bus_coords[bus]
			output = pv.get("output_kw", 0.0)
			hover = (
				f"<b>{DEVICE_ICONS['pv']} PV {pv.get('name', '')}</b><br>"
				f"Output: {output:.1f} kW<br>"
				f"Pmpp: {pv.get('pmpp', 0):.1f} kW"
			)
			fig.add_trace(go.Scatter(
				x=[x], y=[y + offset_y],
				mode="markers",
				marker=dict(size=16, color="#F59E0B", symbol="star"),
				hoverinfo="text",
				hovertext=hover,
				showlegend=False,
			))

	# Storage 设备
	storage_list = devices_data.get("storage", [])
	for sto in storage_list:
		bus = sto.get("bus", "")
		if bus in bus_coords:
			x, y = bus_coords[bus]
			soc = sto.get("soc", 0.0)
			power = sto.get("power_kw", 0.0)
			state = sto.get("state", "idle")
			hover = (
				f"<b>{DEVICE_ICONS['storage']} ESS {sto.get('name', '')}</b><br>"
				f"SOC: {soc * 100:.1f}%<br>"
				f"Power: {power:.1f} kW<br>"
				f"State: {state}"
			)
			fig.add_trace(go.Scatter(
				x=[x + 0.02], y=[y + offset_y],
				mode="markers",
				marker=dict(size=16, color="#3B82F6", symbol="diamond"),
				hoverinfo="text",
				hovertext=hover,
				showlegend=False,
			))

	# Transformers
	xfm_list = devices_data.get("transformers", [])
	for xfm in xfm_list:
		name = xfm.get("name", "")
		hover = (
			f"<b>{DEVICE_ICONS['transformer']} Xfm {name}</b><br>"
			f"Loss: {xfm.get('loss_kw', 0):.2f} kW<br>"
			f"Power: {xfm.get('power_kw', 0):.1f} kW"
		)
		# Transformer 不一定有 bus 坐标，跳过无坐标的
