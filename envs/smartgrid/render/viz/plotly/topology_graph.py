"""
SmartGrid Topology Graph (Plotly)
拓扑图可视化

绘制电网拓扑图，节点着色表示电压幅值，
边着色表示线路类型（线路/变压器），
标注可控设备（电容器、调压器、电池、PV）位置。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from envs.render_common.utils.color_scales import voltage_to_color, loading_to_color
from envs.render_common.viz.theme import (
	get_plotly_layout,
	COLORS,
	DEVICE_COLORS,
)

logger = logging.getLogger(__name__)


def create_topology_graph(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
	show_devices: bool = True,
	show_labels: bool = False,
	v_min: float = 0.95,
	v_max: float = 1.05,
	height: int = 600,
) -> go.Figure:
	"""创建电网拓扑图

	Args:
		snapshot: 快照字典
		bus_coords: {bus_name: (x, y)} 坐标
		show_devices: 是否显示设备标注
		show_labels: 是否显示母线标签
		v_min: 电压下限 (p.u.)
		v_max: 电压上限 (p.u.)
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	fig = go.Figure()

	buses = snapshot.get("buses", {})
	lines = snapshot.get("lines", {})
	devices = snapshot.get("devices", {})
	step = snapshot.get("step", 0)

	# 绘制边（线路和变压器）
	_add_edges(fig, lines, bus_coords)

	# 绘制节点（母线）
	_add_bus_nodes(fig, buses, bus_coords, v_min, v_max, show_labels)

	# 绘制设备标记
	if show_devices:
		_add_device_markers(fig, devices, bus_coords)

	# 布局
	layout = get_plotly_layout(
		title=f"SmartGrid Topology (Step {step})",
		height=height,
		env_name="smartgrid",
	)
	layout["xaxis"] = {
		"showgrid": False,
		"zeroline": False,
		"showticklabels": False,
		"scaleanchor": "y",
	}
	layout["yaxis"] = {
		"showgrid": False,
		"zeroline": False,
		"showticklabels": False,
	}
	layout["showlegend"] = True
	fig.update_layout(**layout)

	return fig


def _add_edges(
	fig: go.Figure,
	lines: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""添加线路和变压器边"""
	line_x: List[Optional[float]] = []
	line_y: List[Optional[float]] = []
	xfmr_x: List[Optional[float]] = []
	xfmr_y: List[Optional[float]] = []

	for name, info in lines.items():
		bus1 = info.get("bus1", "")
		bus2 = info.get("bus2", "")
		if bus1 not in bus_coords or bus2 not in bus_coords:
			continue

		x0, y0 = bus_coords[bus1]
		x1, y1 = bus_coords[bus2]
		edge_type = info.get("type", "line")

		if edge_type == "transformer":
			xfmr_x.extend([x0, x1, None])
			xfmr_y.extend([y0, y1, None])
		else:
			line_x.extend([x0, x1, None])
			line_y.extend([y0, y1, None])

	if line_x:
		fig.add_trace(go.Scatter(
			x=line_x, y=line_y,
			mode="lines",
			line={"color": COLORS["text_secondary"], "width": 1.5},
			name="Lines",
			hoverinfo="skip",
		))

	if xfmr_x:
		fig.add_trace(go.Scatter(
			x=xfmr_x, y=xfmr_y,
			mode="lines",
			line={"color": COLORS["danger"], "width": 2.5, "dash": "dash"},
			name="Transformers",
			hoverinfo="skip",
		))


def _add_bus_nodes(
	fig: go.Figure,
	buses: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	v_min: float,
	v_max: float,
	show_labels: bool,
) -> None:
	"""添加母线节点"""
	x_vals: List[float] = []
	y_vals: List[float] = []
	colors: List[str] = []
	texts: List[str] = []
	hover_texts: List[str] = []

	for bus_name, coord in bus_coords.items():
		bus_info = buses.get(bus_name, {})
		v_pu_list = bus_info.get("v_mag_pu", [])
		v_mean = float(np.mean(v_pu_list)) if v_pu_list else 1.0

		x_vals.append(coord[0])
		y_vals.append(coord[1])
		colors.append(voltage_to_color(v_mean, v_min, v_max))
		texts.append(bus_name if show_labels else "")
		hover_texts.append(
			f"<b>{bus_name}</b><br>"
			f"V = {v_mean:.4f} pu<br>"
			f"Phases: {len(v_pu_list)}"
		)

	fig.add_trace(go.Scatter(
		x=x_vals, y=y_vals,
		mode="markers+text" if show_labels else "markers",
		marker={
			"size": 10,
			"color": colors,
			"line": {"width": 0.5, "color": COLORS["border"]},
		},
		text=texts,
		textposition="top center",
		textfont={"size": 8},
		hovertext=hover_texts,
		hoverinfo="text",
		name="Buses",
	))


def _add_device_markers(
	fig: go.Figure,
	devices: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""添加设备标记"""
	device_types = [
		("capacitors", "Cap", DEVICE_COLORS["capacitor_on"], "diamond"),
		("regulators", "Reg", DEVICE_COLORS["regulator_tap"], "star"),
		("batteries", "Bat", DEVICE_COLORS["battery_soc"], "square"),
		("pvs", "PV", DEVICE_COLORS["pv"], "triangle-up"),
	]

	for dev_key, prefix, color, symbol in device_types:
		dev_dict = devices.get(dev_key, {})
		if not dev_dict:
			continue

		x_vals: List[float] = []
		y_vals: List[float] = []
		hover_texts: List[str] = []

		for name, dev in dev_dict.items():
			bus = dev.get("bus", dev.get("bus1", ""))
			if bus not in bus_coords:
				continue

			bx, by = bus_coords[bus]
			# 偏移设备标记避免重叠
			offset_map = {
				"capacitors": (15.0, 15.0),
				"regulators": (-15.0, 15.0),
				"batteries": (15.0, -15.0),
				"pvs": (-15.0, -15.0),
			}
			dx, dy = offset_map.get(dev_key, (0.0, 0.0))
			x_vals.append(bx + dx)
			y_vals.append(by + dy)

			hover = f"<b>{name}</b><br>"
			if dev_key == "capacitors":
				hover += f"Status: {'ON' if dev.get('status') else 'OFF'}"
			elif dev_key == "regulators":
				hover += f"Tap: {dev.get('tap', 0)}"
			elif dev_key == "batteries":
				hover += f"SOC: {dev.get('soc', 0):.1%}<br>Power: {dev.get('power_kw', 0):.1f} kW"
			elif dev_key == "pvs":
				hover += f"Power: {dev.get('power_ratio', 0):.1%}<br>PF: {dev.get('power_factor', 1):.3f}"
			hover_texts.append(hover)

		if x_vals:
			fig.add_trace(go.Scatter(
				x=x_vals, y=y_vals,
				mode="markers",
				marker={
					"size": 14,
					"color": color,
					"symbol": symbol,
					"line": {"width": 1, "color": "white"},
				},
				name=prefix,
				hovertext=hover_texts,
				hoverinfo="text",
			))
