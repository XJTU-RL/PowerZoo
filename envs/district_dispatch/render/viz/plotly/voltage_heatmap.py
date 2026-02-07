"""
Bus x Time 电压热力图
行=34个母线，列=时间步，颜色=Vpu，越限区域虚线标注。
"""

from typing import Any, Dict, List

import plotly.graph_objects as go

from envs.district_dispatch.render.viz.theme import (
	VOLTAGE_COLORSCALE,
	get_plotly_layout,
)


def _extract_bus_voltage_matrix(
	snapshots: List[Dict[str, Any]],
) -> tuple[list[str], list[float], list[list[float]]]:
	"""从快照序列提取母线电压矩阵。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		tuple: (bus_names, timestamps, voltage_matrix)
			- bus_names: 母线名称列表 (行)
			- timestamps: 时间戳列表 (列)
			- voltage_matrix: [n_buses x n_steps] 电压标幺值矩阵
	"""
	if not snapshots:
		return [], [], []

	# 收集所有母线名称
	all_buses: set[str] = set()
	for snap in snapshots:
		all_buses.update(snap.get("buses", {}).keys())

	bus_names = sorted(all_buses)
	timestamps = [snap.get("timestamp_h", i * 0.25) for i, snap in enumerate(snapshots)]

	# 构建电压矩阵
	voltage_matrix: list[list[float]] = []
	for bus in bus_names:
		row: list[float] = []
		for snap in snapshots:
			bus_data = snap.get("buses", {}).get(bus, {})
			vpu_list = bus_data.get("vpu", [1.0])
			avg_vpu = sum(vpu_list) / len(vpu_list) if vpu_list else 1.0
			row.append(avg_vpu)
		voltage_matrix.append(row)

	return bus_names, timestamps, voltage_matrix


def create_voltage_heatmap(snapshots: List[Dict[str, Any]]) -> go.Figure:
	"""创建 Bus x Time 电压热力图。

	行 = 34 个母线, 列 = 时间步。
	颜色 = Vpu (0.9-1.1 范围)，越限区域用虚线标注 (0.95, 1.05)。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		go.Figure: Plotly 热力图
	"""
	bus_names, timestamps, voltage_matrix = _extract_bus_voltage_matrix(snapshots)

	if not bus_names:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout("Voltage Heatmap (No Data)"))
		return fig

	# 时间轴标签
	time_labels = [f"{t:.2f}h" for t in timestamps]

	fig = go.Figure()

	fig.add_trace(go.Heatmap(
		z=voltage_matrix,
		x=time_labels,
		y=bus_names,
		colorscale=VOLTAGE_COLORSCALE,
		zmin=0.90,
		zmax=1.10,
		colorbar=dict(
			title=dict(text="Vpu"),
			tickvals=[0.90, 0.95, 1.00, 1.05, 1.10],
			ticktext=["0.90", "0.95", "1.00", "1.05", "1.10"],
		),
		hovertemplate=(
			"Bus: %{y}<br>"
			"Time: %{x}<br>"
			"Vpu: %{z:.4f}<br>"
			"<extra></extra>"
		),
	))

	# 添加越限参考线说明 (通过 annotation)
	n_steps = len(timestamps)
	if n_steps > 0:
		# 用文字标注参考电压
		fig.add_annotation(
			x=time_labels[-1],
			y=bus_names[-1],
			text="Limits: 0.95 ~ 1.05 pu",
			showarrow=False,
			font=dict(size=9, color="rgba(255,255,255,0.6)"),
			xanchor="right",
			yanchor="top",
		)

	layout = get_plotly_layout(
		title="Bus Voltage Heatmap (Vpu)",
		height=max(400, len(bus_names) * 18 + 100),
	)
	layout.update(
		xaxis=dict(title="Time", tickangle=-45, side="bottom"),
		yaxis=dict(title="Bus", autorange="reversed"),
	)
	fig.update_layout(**layout)

	return fig
