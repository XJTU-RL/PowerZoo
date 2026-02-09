"""
区间功率交换 Sankey 图
Zone0 <-> Zone1 <-> Zone2，宽度=功率幅值 (kW)，颜色区分正/反向。
"""

from typing import Any, Dict, List

import plotly.graph_objects as go

from envs.district_dispatch.render.viz.theme import (
	ZONE_COLORS,
	ZONE_NAMES,
	get_plotly_layout,
)


def _extract_zone_exchanges(
	snapshot: Dict[str, Any],
) -> List[Dict[str, Any]]:
	"""从快照数据中提取区间功率交换信息。

	查找 circuit.zone_exchanges 或从线路数据推算跨 zone 功率流。

	Args:
		snapshot: 单时间步快照数据

	Returns:
		List[Dict]: 交换记录列表, 每条含 from_zone, to_zone, power_kw
	"""
	circuit = snapshot.get("circuit", {})
	exchanges = circuit.get("zone_exchanges", [])

	if exchanges:
		return exchanges

	# 从线路数据推算
	lines_data = snapshot.get("lines", {})
	from envs.district_dispatch.render.assets.bus_coordinates import BUS_TO_ZONE

	exchange_map: Dict[tuple[int, int], float] = {}

	for line_name, line_data in lines_data.items():
		parts = line_name.split("_")
		if len(parts) < 2:
			continue

		from_bus, to_bus = parts[0], parts[1]
		z_from = BUS_TO_ZONE.get(from_bus, -1)
		z_to = BUS_TO_ZONE.get(to_bus, -1)

		if z_from < 0 or z_to < 0 or z_from == z_to:
			continue

		power_list = line_data.get("power", [])
		p_flow = power_list[0] if power_list else 0.0

		if p_flow >= 0:
			key = (z_from, z_to)
		else:
			key = (z_to, z_from)
			p_flow = abs(p_flow)

		exchange_map[key] = exchange_map.get(key, 0.0) + p_flow

	result = []
	for (z_from, z_to), power in exchange_map.items():
		result.append({
			"from_zone": z_from,
			"to_zone": z_to,
			"power_kw": power,
		})

	return result


def create_exchange_sankey(
	snapshot: Dict[str, Any],
) -> go.Figure:
	"""创建区间功率交换 Sankey 图。

	Zone0 <-> Zone1 <-> Zone2 之间的功率交换，
	宽度 = 功率幅值 (kW)，颜色区分正/反向。

	Args:
		snapshot: 单时间步快照数据

	Returns:
		go.Figure: Plotly Sankey 图
	"""
	exchanges = _extract_zone_exchanges(snapshot)

	# 节点定义 (3 个 Zone, 左中右各一对 in/out)
	# 简化为 3 个节点: Zone 0, Zone 1, Zone 2
	node_labels = [ZONE_NAMES[i] for i in range(3)]
	node_colors = [ZONE_COLORS[i] for i in range(3)]

	# 添加功率汇总信息到节点标签
	circuit = snapshot.get("circuit", {})
	for i in range(3):
		zone_key = f"zone_{i}"
		zone_load = circuit.get(f"{zone_key}_load_kw", 0.0)
		zone_gen = circuit.get(f"{zone_key}_gen_kw", 0.0)
		if zone_load > 0 or zone_gen > 0:
			node_labels[i] += f"<br>Load: {zone_load:.0f} kW<br>Gen: {zone_gen:.0f} kW"

	# 构建 Sankey 链接
	sources: List[int] = []
	targets: List[int] = []
	values: List[float] = []
	link_colors: List[str] = []

	# 正向颜色 (zone_from 颜色) 和反向颜色
	forward_color = "rgba(16,185,129,0.5)"   # 绿色半透明
	reverse_color = "rgba(249,115,22,0.5)"   # 橙色半透明

	if exchanges:
		for ex in exchanges:
			z_from = ex.get("from_zone", 0)
			z_to = ex.get("to_zone", 0)
			power = abs(ex.get("power_kw", 0.0))

			if power < 0.1:
				continue

			sources.append(z_from)
			targets.append(z_to)
			values.append(power)

			# 正向 (0->1, 1->2) = 绿色, 反向 (1->0, 2->1) = 橙色
			if z_from < z_to:
				link_colors.append(forward_color)
			else:
				link_colors.append(reverse_color)
	else:
		# 无交换数据时添加占位
		sources = [0, 1]
		targets = [1, 2]
		values = [0.1, 0.1]
		link_colors = [forward_color, forward_color]

	fig = go.Figure(go.Sankey(
		node=dict(
			pad=30,
			thickness=25,
			label=node_labels,
			color=node_colors,
			line=dict(color="rgba(255,255,255,0.3)", width=1),
			hovertemplate="<b>%{label}</b><extra></extra>",
		),
		link=dict(
			source=sources,
			target=targets,
			value=values,
			color=link_colors,
			hovertemplate=(
				"From: %{source.label}<br>"
				"To: %{target.label}<br>"
				"Power: %{value:.1f} kW<extra></extra>"
			),
		),
	))

	step = snapshot.get("step", 0)
	layout = get_plotly_layout(
		title=f"Inter-Zone Power Exchange (Step {step})",
		height=450,
	)
	fig.update_layout(**layout)

	return fig
