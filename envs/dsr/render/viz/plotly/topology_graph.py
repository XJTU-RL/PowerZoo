"""
DSR Topology Graph
网络拓扑图 -- 故障线路红色虚线 + 断电母线灰色 + 恢复母线绿色

DSR 拓扑图的关键特殊处理:
- 故障线路: 红色虚线 + "FAULT" 标签
- 断电母线: 灰色
- 已恢复母线: 绿色
- 带电母线: 正常电压着色
- 开关操作动画提示
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from envs.render_common.utils.color_scales import voltage_to_color, restoration_to_color
from envs.render_common.viz.theme import (
	COLORS, DEVICE_COLORS, get_plotly_layout,
)
from envs.dsr.render.assets.bus_coordinates import (
	get_bus_coordinates, try_load_from_opendss, generate_fallback_coords,
)

# 母线状态颜色
BUS_COLORS = {
	"energized": None,  # 使用电压着色
	"de_energized": "#6B7280",  # 灰色
	"faulted_bus": "#DC2626",  # 红色
	"restored": COLORS["success"],  # 绿色
}

# 线路状态样式
LINE_STYLES = {
	"normal": {"color": "rgba(255,255,255,0.3)", "dash": "solid", "width": 1.5},
	"faulted": {"color": DEVICE_COLORS["fault_line"], "dash": "dash", "width": 3.0},
	"switch_open": {"color": DEVICE_COLORS["switch_open"], "dash": "dot", "width": 2.0},
	"switch_closed": {"color": DEVICE_COLORS["switch_closed"], "dash": "solid", "width": 2.0},
	"overloaded": {"color": COLORS["warning"], "dash": "solid", "width": 2.5},
}


def create_topology_graph(
	snapshot: Dict[str, Any],
	system_name: str = "13Bus",
	env: Any = None,
	show_labels: bool = True,
	highlight_faults: bool = True,
) -> go.Figure:
	"""创建 DSR 网络拓扑图

	Args:
		snapshot: 快照字典
		system_name: 系统名称
		env: 可选的 DSREnv 实例（用于获取坐标）
		show_labels: 是否显示标签
		highlight_faults: 是否高亮故障线路

	Returns:
		Plotly Figure 对象
	"""
	fig = go.Figure()

	# 获取坐标
	coords = get_bus_coordinates(system_name)
	if not coords and env is not None:
		coords = try_load_from_opendss(env)
	buses = snapshot.get("buses", {})
	if not coords:
		coords = generate_fallback_coords(list(buses.keys()))

	lines = snapshot.get("lines", {})
	restoration = snapshot.get("restoration_data", {})
	energized_set = set(restoration.get("energized_buses", []))
	fault_lines_set = set(restoration.get("fault_lines", []))

	# 绘制线路
	_draw_lines(fig, lines, coords, fault_lines_set, highlight_faults, show_labels)

	# 绘制母线
	_draw_buses(fig, buses, coords, energized_set, fault_lines_set, show_labels)

	# 绘制设备标记
	devices = snapshot.get("devices", {})
	_draw_device_markers(fig, devices, coords)

	# 恢复进度标注
	rest_pct = restoration.get("restoration_pct", 0.0)
	step = snapshot.get("step", 0)
	title = f"Network Topology - Step {step} (Restored: {rest_pct:.1f}%)"

	layout = get_plotly_layout(title=title, height=550, env_name="dsr")
	layout["showlegend"] = True
	layout["xaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False}
	layout["yaxis"] = {"showgrid": False, "zeroline": False, "showticklabels": False, "scaleanchor": "x"}

	fig.update_layout(**layout)

	return fig


def _draw_lines(
	fig: go.Figure,
	lines: Dict[str, Dict[str, Any]],
	coords: Dict,
	fault_lines: set,
	highlight_faults: bool,
	show_labels: bool,
) -> None:
	"""绘制线路

	Args:
		fig: Figure 对象
		lines: 线路数据
		coords: 母线坐标
		fault_lines: 故障线路集合
		highlight_faults: 是否高亮故障
		show_labels: 是否显示标签
	"""
	for line_name, data in lines.items():
		from_bus = data.get("from_bus", "")
		to_bus = data.get("to_bus", "")

		if from_bus not in coords or to_bus not in coords:
			continue

		x0, y0 = coords[from_bus]
		x1, y1 = coords[to_bus]

		# 确定线路样式
		is_faulted = data.get("is_faulted", False) or line_name in fault_lines
		is_open = data.get("is_open", False)
		is_overloaded = data.get("is_overloaded", False)

		if is_faulted and highlight_faults:
			style = LINE_STYLES["faulted"]
		elif is_open:
			style = LINE_STYLES["switch_open"]
		elif is_overloaded:
			style = LINE_STYLES["overloaded"]
		elif data.get("is_switch", False):
			style = LINE_STYLES["switch_closed"]
		else:
			style = LINE_STYLES["normal"]

		hover_text = f"Line: {line_name}<br>From: {from_bus} -> To: {to_bus}"
		if is_faulted:
			hover_text += "<br><b>FAULTED</b>"
		if is_open:
			hover_text += "<br>Switch: OPEN"
		loading = data.get("loading_pct", 0)
		if loading > 0:
			hover_text += f"<br>Loading: {loading:.1f}%"

		fig.add_trace(go.Scatter(
			x=[x0, x1, None],
			y=[y0, y1, None],
			mode="lines",
			line=dict(
				color=style["color"],
				width=style["width"],
				dash=style["dash"],
			),
			hoverinfo="text",
			hovertext=hover_text,
			showlegend=False,
		))

		# 故障标签
		if is_faulted and highlight_faults and show_labels:
			mid_x = (x0 + x1) / 2
			mid_y = (y0 + y1) / 2
			fig.add_annotation(
				x=mid_x, y=mid_y,
				text="<b>FAULT</b>",
				showarrow=False,
				font=dict(color=DEVICE_COLORS["fault_line"], size=10),
				bgcolor="rgba(220,38,38,0.2)",
				borderpad=2,
			)


def _draw_buses(
	fig: go.Figure,
	buses: Dict[str, Dict[str, Any]],
	coords: Dict,
	energized_set: set,
	fault_lines: set,
	show_labels: bool,
) -> None:
	"""绘制母线节点

	Args:
		fig: Figure 对象
		buses: 母线数据
		coords: 母线坐标
		energized_set: 带电母线集合
		fault_lines: 故障线路集合
		show_labels: 是否显示标签
	"""
	# 分类母线
	energized_x, energized_y, energized_text, energized_colors = [], [], [], []
	deenergized_x, deenergized_y, deenergized_text = [], [], []

	for bus_name, bus_data in buses.items():
		if bus_name not in coords:
			continue

		x, y = coords[bus_name]
		is_energized = bus_data.get("is_energized", bus_name in energized_set)

		v_pu_list = bus_data.get("v_mag_pu", [])
		v_pu = v_pu_list[0] if v_pu_list else 0.0

		hover = f"Bus: {bus_name}<br>Energized: {'Yes' if is_energized else 'No'}"
		if v_pu > 0:
			hover += f"<br>V: {v_pu:.4f} pu"

		if is_energized:
			energized_x.append(x)
			energized_y.append(y)
			energized_text.append(bus_name if show_labels else "")
			energized_colors.append(voltage_to_color(v_pu) if v_pu > 0.1 else COLORS["success"])
		else:
			deenergized_x.append(x)
			deenergized_y.append(y)
			deenergized_text.append(bus_name if show_labels else "")

	# 带电母线
	if energized_x:
		fig.add_trace(go.Scatter(
			x=energized_x, y=energized_y,
			mode="markers+text" if show_labels else "markers",
			marker=dict(
				size=12,
				color=energized_colors,
				line=dict(width=1, color="white"),
			),
			text=energized_text,
			textposition="top center",
			textfont=dict(size=8, color=COLORS["text_secondary"]),
			name="Energized",
			hoverinfo="text",
			hovertext=[f"Bus: {t}" for t in energized_text],
		))

	# 断电母线
	if deenergized_x:
		fig.add_trace(go.Scatter(
			x=deenergized_x, y=deenergized_y,
			mode="markers+text" if show_labels else "markers",
			marker=dict(
				size=10,
				color=BUS_COLORS["de_energized"],
				symbol="x",
				line=dict(width=1, color="white"),
			),
			text=deenergized_text,
			textposition="top center",
			textfont=dict(size=8, color=COLORS["text_secondary"]),
			name="De-energized",
			hoverinfo="text",
			hovertext=[f"Bus: {t} (de-energized)" for t in deenergized_text],
		))


def _draw_device_markers(
	fig: go.Figure,
	devices: Dict[str, Any],
	coords: Dict,
) -> None:
	"""在拓扑图上绘制设备标记（PV、负荷）

	Args:
		fig: Figure 对象
		devices: 设备数据
		coords: 母线坐标
	"""
	# PV 设备
	pvs = devices.get("pvs", {})
	pv_x, pv_y, pv_text = [], [], []
	for pv_name, pv_data in pvs.items():
		bus = pv_data.get("bus", "")
		if bus in coords:
			x, y = coords[bus]
			pv_x.append(x + 15)  # 偏移避免重叠
			pv_y.append(y + 15)
			ratio = pv_data.get("power_ratio", 0)
			pv_text.append(f"PV: {pv_name}<br>Power: {ratio*100:.0f}%")

	if pv_x:
		fig.add_trace(go.Scatter(
			x=pv_x, y=pv_y,
			mode="markers",
			marker=dict(
				size=8,
				color=DEVICE_COLORS["pv"],
				symbol="star-triangle-up",
			),
			name="PV",
			hoverinfo="text",
			hovertext=pv_text,
		))

	# 负荷标记（仅高优先级）
	loads = devices.get("loads", {})
	load_x, load_y, load_text, load_colors = [], [], [], []
	priority_color_map = {
		4: DEVICE_COLORS["priority_critical"],
		3: DEVICE_COLORS["priority_high"],
		2: DEVICE_COLORS["priority_medium"],
	}
	for load_name, load_data in loads.items():
		priority = load_data.get("priority", 0)
		if priority < 2:
			continue
		bus = load_data.get("bus", "")
		if bus in coords:
			x, y = coords[bus]
			load_x.append(x - 15)
			load_y.append(y - 15)
			status = "ON" if load_data.get("is_connected", False) else "OFF"
			load_text.append(f"Load: {load_name}<br>Priority: {priority}<br>Status: {status}")
			load_colors.append(priority_color_map.get(priority, DEVICE_COLORS["priority_low"]))

	if load_x:
		fig.add_trace(go.Scatter(
			x=load_x, y=load_y,
			mode="markers",
			marker=dict(
				size=7,
				color=load_colors,
				symbol="triangle-down",
			),
			name="Priority Loads",
			hoverinfo="text",
			hovertext=load_text,
		))
