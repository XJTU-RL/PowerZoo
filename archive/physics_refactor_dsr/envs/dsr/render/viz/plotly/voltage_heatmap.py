"""
DSR Voltage Heatmap
电压热力图

X 轴: 时间步 (10-20步，无需降采样)
Y 轴: 母线名称
颜色: 电压标幺值 (pu)

断电母线显示为灰色，故障区域特殊标注。
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import (
	VOLTAGE_COLORSCALE, get_plotly_layout,
)


def create_voltage_heatmap(
	snapshots: List[Dict[str, Any]],
	v_min: float = 0.90,
	v_max: float = 1.10,
) -> go.Figure:
	"""创建电压热力图

	Args:
		snapshots: 快照列表
		v_min: 电压色标下限
		v_max: 电压色标上限

	Returns:
		Plotly Figure 对象
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Voltage Heatmap (No Data)", env_name="dsr"))
		return fig

	# 收集所有母线名称
	all_bus_names = set()
	for snap in snapshots:
		buses = snap.get("buses", {})
		all_bus_names.update(buses.keys())

	bus_names = sorted(all_bus_names)
	if not bus_names:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Voltage Heatmap (No Bus Data)", env_name="dsr"))
		return fig

	n_steps = len(snapshots)
	n_buses = len(bus_names)
	bus_idx_map = {name: i for i, name in enumerate(bus_names)}

	# 构建热力图矩阵
	z_matrix = np.full((n_buses, n_steps), np.nan)
	custom_data = [[None] * n_steps for _ in range(n_buses)]

	for j, snap in enumerate(snapshots):
		buses = snap.get("buses", {})
		restoration = snap.get("restoration_data", {})
		energized_set = set(restoration.get("energized_buses", []))

		for bus_name, bus_data in buses.items():
			if bus_name not in bus_idx_map:
				continue
			i = bus_idx_map[bus_name]
			is_energized = bus_data.get("is_energized", bus_name in energized_set)

			v_pu_list = bus_data.get("v_mag_pu", [])
			if v_pu_list and is_energized:
				v_pu = float(v_pu_list[0])
				if 0.1 < v_pu < 2.0:
					z_matrix[i, j] = v_pu
					custom_data[i][j] = f"Bus: {bus_name}<br>V: {v_pu:.4f} pu<br>Step: {snap.get('step', j)}"

	# 步序列
	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	fig = go.Figure()

	fig.add_trace(go.Heatmap(
		z=z_matrix,
		x=steps,
		y=bus_names,
		colorscale=VOLTAGE_COLORSCALE,
		zmin=v_min,
		zmax=v_max,
		colorbar=dict(
			title="V (pu)",
			titleside="right",
		),
		hoverinfo="text",
		text=custom_data,
		showscale=True,
	))

	# 电压安全界限标注
	fig.add_annotation(
		text=f"Safe range: [{v_min + 0.05:.2f}, {v_max - 0.05:.2f}] pu",
		xref="paper", yref="paper",
		x=1.0, y=1.02,
		showarrow=False,
		font=dict(size=10, color="#a0a0a0"),
	)

	layout = get_plotly_layout(
		title="Bus Voltage Heatmap",
		height=max(400, n_buses * 18 + 100),
		env_name="dsr",
	)
	layout["xaxis_title"] = "Step"
	layout["yaxis_title"] = "Bus"

	fig.update_layout(**layout)

	return fig
