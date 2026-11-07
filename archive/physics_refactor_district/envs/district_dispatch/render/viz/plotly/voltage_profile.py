"""
Distance vs Vpu 电压剖面曲线
三相电压分别绘制，Min/Max 限制线，按 Zone 分段高亮。
"""

from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go

from envs.district_dispatch.render.assets.bus_coordinates import (
	BUS_TO_ZONE,
	ZONE_BUSES,
)
from envs.district_dispatch.render.viz.theme import (
	ZONE_COLORS,
	ZONE_NAMES,
	get_plotly_layout,
)

# 三相颜色
PHASE_COLORS: List[str] = ["#EF4444", "#10B981", "#3B82F6"]
PHASE_NAMES: List[str] = ["Phase A", "Phase B", "Phase C"]


def _compute_bus_distances(
	bus_coords: Dict[str, Tuple[float, float]],
	bus_order: List[str],
) -> Dict[str, float]:
	"""计算母线沿馈线的累计距离。

	使用母线坐标计算相邻母线间的欧几里得距离，
	按给定顺序累加作为沿线距离的近似值。

	Args:
		bus_coords: {母线名称: (x, y)} 坐标字典
		bus_order: 按馈线顺序排列的母线名称列表

	Returns:
		Dict[str, float]: {母线名称: 累计距离}
	"""
	distances: Dict[str, float] = {}
	cumulative = 0.0

	for i, bus in enumerate(bus_order):
		if i == 0:
			distances[bus] = 0.0
			continue

		prev_bus = bus_order[i - 1]
		if bus in bus_coords and prev_bus in bus_coords:
			x0, y0 = bus_coords[prev_bus]
			x1, y1 = bus_coords[bus]
			dx = x1 - x0
			dy = y1 - y0
			segment = (dx ** 2 + dy ** 2) ** 0.5
			cumulative += segment
		else:
			cumulative += 0.05  # 默认间距

		distances[bus] = cumulative

	return distances


def _get_ordered_buses(
	bus_coords: Dict[str, Tuple[float, float]],
) -> List[str]:
	"""获取按 x 坐标排序的母线列表，近似馈线顺序。

	Args:
		bus_coords: {母线名称: (x, y)} 坐标字典

	Returns:
		List[str]: 排序后的母线名称列表
	"""
	return sorted(bus_coords.keys(), key=lambda b: bus_coords.get(b, (0, 0))[0])


def create_voltage_profile(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> go.Figure:
	"""创建 Distance vs Vpu 电压剖面曲线图。

	3 相电压分别绘制，Min/Max 限制线 (0.95, 1.05)，按 Zone 分段高亮。

	Args:
		snapshot: 单时间步快照数据
		bus_coords: {母线名称: (x, y)} 归一化坐标字典

	Returns:
		go.Figure: Plotly 电压剖面图
	"""
	buses_data = snapshot.get("buses", {})
	bus_order = _get_ordered_buses(bus_coords)
	distances = _compute_bus_distances(bus_coords, bus_order)

	fig = go.Figure()

	# 收集各相电压数据
	for phase_idx in range(3):
		dist_vals: List[float] = []
		vpu_vals: List[float] = []
		hover_texts: List[str] = []

		for bus in bus_order:
			if bus not in distances:
				continue
			bus_data = buses_data.get(bus, {})
			vpu_list = bus_data.get("vpu", [])

			if phase_idx < len(vpu_list):
				dist_vals.append(distances[bus])
				vpu_vals.append(vpu_list[phase_idx])
				hover_texts.append(
					f"Bus {bus}<br>{PHASE_NAMES[phase_idx]}: {vpu_list[phase_idx]:.4f} pu"
				)

		if dist_vals:
			fig.add_trace(go.Scatter(
				x=dist_vals,
				y=vpu_vals,
				mode="lines+markers",
				name=PHASE_NAMES[phase_idx],
				line=dict(color=PHASE_COLORS[phase_idx], width=2),
				marker=dict(size=5),
				hoverinfo="text",
				hovertext=hover_texts,
			))

	# 限制线
	if bus_order and distances:
		d_min = 0.0
		d_max = max(distances.values())

		fig.add_trace(go.Scatter(
			x=[d_min, d_max],
			y=[1.05, 1.05],
			mode="lines",
			name="V_max (1.05)",
			line=dict(color="#EF4444", width=1.5, dash="dash"),
		))
		fig.add_trace(go.Scatter(
			x=[d_min, d_max],
			y=[0.95, 0.95],
			mode="lines",
			name="V_min (0.95)",
			line=dict(color="#EF4444", width=1.5, dash="dash"),
		))

		# Zone 分段高亮
		for zone_id, zone_buses in ZONE_BUSES.items():
			zone_dists = [
				distances[b] for b in zone_buses
				if b in distances
			]
			if not zone_dists:
				continue

			z_min = min(zone_dists)
			z_max = max(zone_dists)
			color = ZONE_COLORS[zone_id]

			fig.add_vrect(
				x0=z_min, x1=z_max,
				fillcolor=color,
				opacity=0.08,
				line=dict(width=0),
				annotation_text=ZONE_NAMES[zone_id],
				annotation_position="top left",
				annotation_font=dict(size=9, color=color),
			)

	step = snapshot.get("step", 0)
	layout = get_plotly_layout(
		title=f"Voltage Profile (Step {step})",
		height=450,
	)
	layout.update(
		xaxis=dict(title="Distance along feeder"),
		yaxis=dict(title="Voltage (pu)", range=[0.88, 1.12]),
		hovermode="closest",
	)
	fig.update_layout(**layout)

	return fig
