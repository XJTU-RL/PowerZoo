"""
IEEE 34-Bus 电路拓扑交互式网络图
节点按电压着色，线路按负载率着色，设备图标标注，Zone 背景区域。
"""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

from envs.district_dispatch.render.assets.bus_coordinates import (
	BUS_TO_ZONE,
	ZONE_BUSES,
)
from envs.district_dispatch.render.utils.color_scales import (
	loading_to_color,
	voltage_to_color,
)
from envs.district_dispatch.render.viz.theme import (
	ZONE_COLORS,
	ZONE_NAMES,
	get_plotly_layout,
)

# 设备图标映射
DEVICE_ICONS: Dict[str, str] = {
	"pv": "\u2600",         # ☀
	"storage": "\u26a1",    # ⚡
	"ev": "\U0001f50c",     # 🔌
	"regctrl": "\U0001f504", # 🔄
}

# IEEE 34-bus 系统拓扑边列表 (from_bus, to_bus)
TOPOLOGY_EDGES: List[Tuple[str, str]] = [
	("800", "802"), ("802", "806"), ("806", "808"), ("808", "810"),
	("808", "812"), ("812", "814"), ("814", "850"), ("850", "816"),
	("816", "818"), ("818", "820"), ("820", "822"), ("816", "824"),
	("824", "826"), ("824", "828"), ("828", "830"), ("830", "854"),
	("854", "856"), ("854", "852"), ("852", "832"), ("832", "858"),
	("858", "864"), ("858", "834"), ("834", "860"), ("860", "836"),
	("836", "840"), ("836", "838"), ("838", "842"), ("842", "844"),
	("844", "846"), ("846", "848"), ("834", "862"), ("852", "888"),
	("888", "890"),
]


def _get_bus_vpu_avg(bus_data: Dict[str, Any]) -> float:
	"""计算母线三相平均电压标幺值。

	Args:
		bus_data: 母线数据字典，含 vpu 列表

	Returns:
		float: 三相平均电压标幺值
	"""
	vpu_list = bus_data.get("vpu", [1.0])
	if not vpu_list:
		return 1.0
	return sum(vpu_list) / len(vpu_list)


def _build_bus_hover(bus_name: str, bus_data: Dict[str, Any]) -> str:
	"""构建母线 hover 文本。

	Args:
		bus_name: 母线名称
		bus_data: 母线数据字典

	Returns:
		str: 格式化的 hover 信息
	"""
	vpu = bus_data.get("vpu", [])
	angle = bus_data.get("angle", [])
	kw = bus_data.get("kw_injection", 0.0)
	kvar = bus_data.get("kvar_injection", 0.0)
	devices = bus_data.get("connected_devices", [])

	lines = [f"<b>Bus {bus_name}</b>"]
	zone_id = BUS_TO_ZONE.get(bus_name, -1)
	if zone_id >= 0:
		lines.append(f"Zone: {zone_id}")

	if vpu:
		vpu_str = ", ".join(f"{v:.4f}" for v in vpu)
		lines.append(f"Vpu: [{vpu_str}]")
	if angle:
		ang_str = ", ".join(f"{a:.1f}" for a in angle)
		lines.append(f"Angle: [{ang_str}] deg")

	lines.append(f"P inj: {kw:.1f} kW")
	lines.append(f"Q inj: {kvar:.1f} kvar")

	if devices:
		lines.append(f"Devices: {', '.join(devices)}")

	return "<br>".join(lines)


def _build_line_hover(line_name: str, line_data: Dict[str, Any]) -> str:
	"""构建线路 hover 文本。

	Args:
		line_name: 线路名称
		line_data: 线路数据字典

	Returns:
		str: 格式化的 hover 信息
	"""
	current_mag = line_data.get("current_mag", [])
	power = line_data.get("power", [])
	losses = line_data.get("losses", {})
	loading = line_data.get("loading_pct", 0.0)
	length = line_data.get("length", 0.0)

	lines = [f"<b>Line {line_name}</b>"]

	if current_mag:
		i_str = ", ".join(f"{i:.1f}" for i in current_mag)
		lines.append(f"I: [{i_str}] A")
	if power:
		p_str = ", ".join(f"{p:.1f}" for p in power)
		lines.append(f"P/Q: [{p_str}]")
	if losses:
		loss_kw = losses.get("kw", 0.0)
		loss_kvar = losses.get("kvar", 0.0)
		lines.append(f"Losses: {loss_kw:.2f} kW, {loss_kvar:.2f} kvar")

	lines.append(f"Loading: {loading:.1f}%")
	if length > 0:
		lines.append(f"Length: {length:.1f} m")

	return "<br>".join(lines)


def _build_device_hover(dev_type: str, dev_data: Dict[str, Any]) -> str:
	"""构建设备 hover 文本。

	Args:
		dev_type: 设备类型 (pv, storage, ev)
		dev_data: 设备数据字典

	Returns:
		str: 格式化的 hover 信息
	"""
	icon = DEVICE_ICONS.get(dev_type, "")
	lines = [f"<b>{icon} {dev_type.upper()}</b>"]

	if dev_type == "pv":
		output = dev_data.get("output_kw", 0.0)
		curtail = dev_data.get("curtailment_kw", 0.0)
		lines.append(f"Output: {output:.1f} kW")
		lines.append(f"Curtailment: {curtail:.1f} kW")
	elif dev_type == "storage":
		soc = dev_data.get("soc", 0.0)
		power = dev_data.get("power_kw", 0.0)
		lines.append(f"SOC: {soc * 100:.1f}%")
		lines.append(f"Power: {power:.1f} kW")
	elif dev_type == "ev":
		load = dev_data.get("load_kw", 0.0)
		lines.append(f"Load: {load:.1f} kW")

	return "<br>".join(lines)


def _add_zone_backgrounds(
	fig: go.Figure,
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""添加三个 Zone 的半透明背景区域。

	Args:
		fig: Plotly Figure 对象
		bus_coords: {母线名称: (x, y)} 坐标字典
	"""
	for zone_id, bus_list in ZONE_BUSES.items():
		zone_xs = []
		zone_ys = []
		for bus in bus_list:
			if bus in bus_coords:
				x, y = bus_coords[bus]
				zone_xs.append(x)
				zone_ys.append(y)

		if not zone_xs:
			continue

		# 用矩形框住该 zone 所有母线
		pad = 0.03
		x_min, x_max = min(zone_xs) - pad, max(zone_xs) + pad
		y_min, y_max = min(zone_ys) - pad, max(zone_ys) + pad

		color = ZONE_COLORS[zone_id]
		fig.add_shape(
			type="rect",
			x0=x_min, y0=y_min, x1=x_max, y1=y_max,
			fillcolor=color,
			opacity=0.08,
			line=dict(color=color, width=1, dash="dot"),
			layer="below",
		)
		fig.add_annotation(
			x=(x_min + x_max) / 2,
			y=y_max + 0.01,
			text=ZONE_NAMES[zone_id],
			showarrow=False,
			font=dict(color=color, size=10),
		)


def _add_power_flow_arrows(
	fig: go.Figure,
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""在联络线上添加功率流向箭头。

	Args:
		fig: Plotly Figure 对象
		snapshot: 快照数据
		bus_coords: 母线坐标字典
	"""
	lines_data = snapshot.get("lines", {})
	for from_bus, to_bus in TOPOLOGY_EDGES:
		if from_bus not in bus_coords or to_bus not in bus_coords:
			continue

		# 检查是否为跨 zone 联络线
		z_from = BUS_TO_ZONE.get(from_bus, -1)
		z_to = BUS_TO_ZONE.get(to_bus, -1)
		if z_from == z_to:
			continue

		x0, y0 = bus_coords[from_bus]
		x1, y1 = bus_coords[to_bus]

		# 查找线路功率方向
		line_key = f"{from_bus}_{to_bus}"
		rev_key = f"{to_bus}_{from_bus}"
		line_data = lines_data.get(line_key, lines_data.get(rev_key, {}))
		power_list = line_data.get("power", [])
		p_flow = power_list[0] if power_list else 0.0

		# 反向功率则翻转箭头
		if p_flow < 0:
			x0, y0, x1, y1 = x1, y1, x0, y0

		fig.add_annotation(
			x=x1, y=y1,
			ax=x0, ay=y0,
			xref="x", yref="y",
			axref="x", ayref="y",
			showarrow=True,
			arrowhead=3,
			arrowsize=1.5,
			arrowwidth=2,
			arrowcolor="#F59E0B",
			opacity=0.7,
		)


def create_topology_figure(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> go.Figure:
	"""创建 IEEE 34-bus 电路拓扑交互式网络图。

	节点按电压着色 (绿色=正常, 黄色=轻微越限, 红色=严重越限)，
	线路按负载率着色 (蓝->红)，包含设备图标标注和 Zone 背景区域。

	Args:
		snapshot: 单时间步快照数据字典
		bus_coords: {母线名称: (x, y)} 归一化坐标字典

	Returns:
		go.Figure: Plotly 交互式网络图
	"""
	fig = go.Figure()

	buses_data = snapshot.get("buses", {})
	lines_data = snapshot.get("lines", {})
	devices_data = snapshot.get("devices", {})
	regulators_data = snapshot.get("regulators", {})

	# 1. Zone 背景
	_add_zone_backgrounds(fig, bus_coords)

	# 2. 线路 (边)
	for from_bus, to_bus in TOPOLOGY_EDGES:
		if from_bus not in bus_coords or to_bus not in bus_coords:
			continue

		x0, y0 = bus_coords[from_bus]
		x1, y1 = bus_coords[to_bus]

		# 查找线路数据
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

	# 3. 母线节点
	node_x, node_y, node_colors, node_hovers, node_texts = [], [], [], [], []
	for bus_name, (x, y) in bus_coords.items():
		bus_data = buses_data.get(bus_name, {})
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

	# 4. 设备图标
	_add_device_markers(fig, devices_data, regulators_data, bus_coords, buses_data)

	# 5. 功率流向箭头 (仅联络线)
	_add_power_flow_arrows(fig, snapshot, bus_coords)

	# 6. Layout
	step = snapshot.get("step", 0)
	ts = snapshot.get("timestamp_h", 0.0)
	layout = get_plotly_layout(
		title=f"IEEE 34-Bus Topology (Step {step}, t={ts:.2f}h)",
		height=700,
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
	regulators_data: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
	buses_data: Dict[str, Any],
) -> None:
	"""添加设备图标标记到拓扑图上。

	Args:
		fig: Plotly Figure 对象
		devices_data: 设备数据字典
		regulators_data: 调压器数据字典
		bus_coords: 母线坐标字典
		buses_data: 母线数据字典
	"""
	offset_y = 0.025

	# PV 设备
	pv_data = devices_data.get("pv", {})
	pv_bus = pv_data.get("bus", None)
	if pv_bus and pv_bus in bus_coords:
		x, y = bus_coords[pv_bus]
		fig.add_trace(go.Scatter(
			x=[x], y=[y + offset_y],
			mode="markers+text",
			marker=dict(size=18, color="#F59E0B", symbol="star"),
			text=[DEVICE_ICONS["pv"]],
			textposition="top center",
			hoverinfo="text",
			hovertext=_build_device_hover("pv", pv_data),
			showlegend=False,
			name="PV",
		))

	# Storage 设备
	storage_data = devices_data.get("storage", {})
	storage_bus = storage_data.get("bus", None)
	if storage_bus and storage_bus in bus_coords:
		x, y = bus_coords[storage_bus]
		fig.add_trace(go.Scatter(
			x=[x + 0.02], y=[y + offset_y],
			mode="markers+text",
			marker=dict(size=18, color="#3B82F6", symbol="diamond"),
			text=[DEVICE_ICONS["storage"]],
			textposition="top center",
			hoverinfo="text",
			hovertext=_build_device_hover("storage", storage_data),
			showlegend=False,
			name="Storage",
		))

	# EV 设备
	ev_data = devices_data.get("ev", {})
	ev_bus = ev_data.get("bus", None)
	if ev_bus and ev_bus in bus_coords:
		x, y = bus_coords[ev_bus]
		fig.add_trace(go.Scatter(
			x=[x - 0.02], y=[y + offset_y],
			mode="markers+text",
			marker=dict(size=18, color="#7C3AED", symbol="square"),
			text=[DEVICE_ICONS["ev"]],
			textposition="top center",
			hoverinfo="text",
			hovertext=_build_device_hover("ev", ev_data),
			showlegend=False,
			name="EV",
		))

	# Regulator 设备
	for reg_name, reg_data in regulators_data.items():
		reg_bus = reg_data.get("bus", None)
		if reg_bus and reg_bus in bus_coords:
			x, y = bus_coords[reg_bus]
			tap = reg_data.get("tap_number", 0)
			hover = (
				f"<b>{DEVICE_ICONS['regctrl']} Reg {reg_name}</b><br>"
				f"Tap: {tap}<br>"
				f"Vreg: {reg_data.get('vreg', 0.0):.4f} pu"
			)
			fig.add_trace(go.Scatter(
				x=[x], y=[y - offset_y],
				mode="markers+text",
				marker=dict(size=14, color="#10B981", symbol="triangle-up"),
				text=[DEVICE_ICONS["regctrl"]],
				textposition="bottom center",
				hoverinfo="text",
				hovertext=hover,
				showlegend=False,
				name=f"Reg {reg_name}",
			))
