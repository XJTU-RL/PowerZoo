"""
功率平衡 Stacked Area Chart 和功率流向覆盖图。
分层展示 Load, PV, Storage, EV, Exchange, Loss 随时间变化。
"""

from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go

from envs.district_dispatch.render.assets.bus_coordinates import BUS_TO_ZONE
from envs.district_dispatch.render.utils.color_scales import loading_to_color
from envs.district_dispatch.render.viz.theme import (
	DEVICE_COLORS,
	get_plotly_layout,
)

# 功率分量定义: (key, display_name, color)
POWER_COMPONENTS: List[Tuple[str, str, str]] = [
	("load", "Load", DEVICE_COLORS.get("load", "#6B7280")),
	("pv", "PV Generation", DEVICE_COLORS.get("pv", "#F59E0B")),
	("storage", "Storage", DEVICE_COLORS.get("storage_charge", "#3B82F6")),
	("ev", "EV Load", DEVICE_COLORS.get("ev", "#7C3AED")),
	("exchange", "Exchange", DEVICE_COLORS.get("exchange_in", "#10B981")),
	("loss", "Losses", "#EF4444"),
]


def _extract_power_timeseries(
	snapshots: List[Dict[str, Any]],
) -> Tuple[List[float], Dict[str, List[float]]]:
	"""从快照序列提取各功率分量时间序列。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		tuple: (timestamps, power_series)
			- timestamps: 时间戳列表
			- power_series: {component_key: [values_per_step]}
	"""
	timestamps: List[float] = []
	power_series: Dict[str, List[float]] = {k: [] for k, _, _ in POWER_COMPONENTS}

	for i, snap in enumerate(snapshots):
		timestamps.append(snap.get("timestamp_h", i * 0.25))
		circuit = snap.get("circuit", {})
		devices = snap.get("devices", {})

		# Load
		power_series["load"].append(circuit.get("total_load_kw", 0.0))
		# PV
		pv = devices.get("pv", {})
		power_series["pv"].append(pv.get("output_kw", 0.0))
		# Storage
		storage = devices.get("storage", {})
		power_series["storage"].append(storage.get("power_kw", 0.0))
		# EV
		ev = devices.get("ev", {})
		power_series["ev"].append(ev.get("load_kw", 0.0))
		# Exchange (inter-zone)
		power_series["exchange"].append(circuit.get("exchange_kw", 0.0))
		# Loss
		power_series["loss"].append(circuit.get("total_loss_kw", 0.0))

	return timestamps, power_series


def create_power_balance_chart(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建功率平衡堆叠面积图。

	各层: Load, PV, Storage, EV, Exchange, Loss over time。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		go.Figure: Plotly stacked area chart
	"""
	timestamps, power_series = _extract_power_timeseries(snapshots)

	if not timestamps:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout("Power Balance (No Data)"))
		return fig

	time_labels = [f"{t:.2f}" for t in timestamps]

	fig = go.Figure()

	for key, name, color in POWER_COMPONENTS:
		values = power_series[key]
		fig.add_trace(go.Scatter(
			x=time_labels,
			y=values,
			mode="lines",
			name=name,
			line=dict(width=0.5, color=color),
			stackgroup="power",
			fillcolor=_to_rgba(color, 0.6),
			hovertemplate=f"{name}<br>Time: %{{x}}h<br>Power: %{{y:.1f}} kW<extra></extra>",
		))

	layout = get_plotly_layout(
		title="Power Balance (Stacked Area)",
		height=500,
	)
	layout.update(
		xaxis=dict(title="Time (h)"),
		yaxis=dict(title="Power (kW)"),
		hovermode="x unified",
	)
	fig.update_layout(**layout)

	return fig


def create_power_flow_overlay(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> go.Figure:
	"""创建功率流向覆盖图（在拓扑图上叠加功率方向箭头）。

	线路颜色按负载率着色，箭头方向表示功率流向。

	Args:
		snapshot: 单时间步快照数据
		bus_coords: {母线名称: (x, y)} 归一化坐标字典

	Returns:
		go.Figure: Plotly 功率流向图
	"""
	fig = go.Figure()

	lines_data = snapshot.get("lines", {})

	# 按负载率绘制线路并添加方向箭头
	for line_name, line_data in lines_data.items():
		parts = line_name.split("_")
		if len(parts) < 2:
			continue

		from_bus, to_bus = parts[0], parts[1]
		if from_bus not in bus_coords or to_bus not in bus_coords:
			continue

		x0, y0 = bus_coords[from_bus]
		x1, y1 = bus_coords[to_bus]
		loading = line_data.get("loading_pct", 0.0)
		color = loading_to_color(loading)

		# 功率方向
		power_list = line_data.get("power", [])
		p_flow = power_list[0] if power_list else 0.0

		# 线路
		fig.add_trace(go.Scatter(
			x=[x0, x1, None],
			y=[y0, y1, None],
			mode="lines",
			line=dict(color=color, width=max(1.5, loading / 25)),
			hoverinfo="text",
			hovertext=(
				f"<b>{line_name}</b><br>"
				f"Loading: {loading:.1f}%<br>"
				f"P: {p_flow:.1f} kW"
			),
			showlegend=False,
		))

		# 箭头 (功率方向)
		if abs(p_flow) > 0.1:
			ax, ay = (x0, y0) if p_flow >= 0 else (x1, y1)
			target_x = (x0 + x1) / 2
			target_y = (y0 + y1) / 2
			fig.add_annotation(
				x=target_x, y=target_y,
				ax=ax, ay=ay,
				xref="x", yref="y",
				axref="x", ayref="y",
				showarrow=True,
				arrowhead=2,
				arrowsize=1.2,
				arrowwidth=1.5,
				arrowcolor=color,
				opacity=0.8,
			)

	# 母线节点
	for bus_name, (x, y) in bus_coords.items():
		zone = BUS_TO_ZONE.get(bus_name, -1)
		fig.add_trace(go.Scatter(
			x=[x], y=[y],
			mode="markers+text",
			marker=dict(size=8, color="#e0e0e0"),
			text=[bus_name],
			textposition="top center",
			textfont=dict(size=7),
			hoverinfo="text",
			hovertext=f"Bus {bus_name} (Zone {zone})",
			showlegend=False,
		))

	step = snapshot.get("step", 0)
	layout = get_plotly_layout(
		title=f"Power Flow Overlay (Step {step})",
		height=650,
	)
	layout.update(
		xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
		yaxis=dict(
			showgrid=False, zeroline=False, showticklabels=False,
			scaleanchor="x", scaleratio=1,
		),
		hovermode="closest",
	)
	fig.update_layout(**layout)

	return fig


def _to_rgba(hex_color: str, alpha: float) -> str:
	"""将十六进制颜色转为 rgba 字符串。

	Args:
		hex_color: 十六进制颜色, 如 "#EF4444"
		alpha: 透明度 (0-1)

	Returns:
		str: rgba 格式颜色字符串
	"""
	h = hex_color.lstrip("#")
	r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
	return f"rgba({r},{g},{b},{alpha})"
