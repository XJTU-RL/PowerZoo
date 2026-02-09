"""
SmartGrid Voltage Heatmap (Plotly)
电压热图

绘制 bus x step 的电压热图。SmartGrid 特别处理：
episode 长达 360 步，需要降采样选项（max_display_steps=100）。
时间轴标签使用 "Day of Year"。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import get_plotly_layout, VOLTAGE_COLORSCALE

logger = logging.getLogger(__name__)


def create_voltage_heatmap(
	snapshots: List[Dict[str, Any]],
	max_display_steps: int = 100,
	v_min: float = 0.90,
	v_max: float = 1.10,
	height: int = 500,
	bus_filter: Optional[List[str]] = None,
) -> go.Figure:
	"""创建电压热图

	对于 360 步 episode，自动降采样到 max_display_steps 步
	以保持热图可读性。

	Args:
		snapshots: 快照列表
		max_display_steps: 最大显示步数（超过则降采样）
		v_min: 电压色标下限
		v_max: 电压色标上限
		height: 图表高度
		bus_filter: 仅显示指定母线（None 则显示全部）

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		return _empty_figure("No data available", height)

	# 提取母线名称
	bus_names = _get_bus_names(snapshots, bus_filter)
	if not bus_names:
		return _empty_figure("No bus data found", height)

	n_steps = len(snapshots)
	n_buses = len(bus_names)

	# 构建电压矩阵 (bus x step)
	v_matrix = np.full((n_buses, n_steps), np.nan)
	step_indices = list(range(n_steps))

	for j, snap in enumerate(snapshots):
		buses = snap.get("buses", {})
		for i, bus_name in enumerate(bus_names):
			bus_info = buses.get(bus_name, {})
			v_pu = bus_info.get("v_mag_pu", [])
			if v_pu:
				v_matrix[i, j] = float(np.mean(v_pu))

	# 降采样处理
	if n_steps > max_display_steps:
		v_matrix, step_indices = _downsample(
			v_matrix, step_indices, max_display_steps
		)
		logger.info(
			f"Voltage heatmap downsampled: {n_steps} -> {len(step_indices)} steps"
		)

	# 时间轴标签 (Day of Year)
	x_labels = [f"Day {idx + 1}" for idx in step_indices]

	# 限制 y 轴标签数量
	y_labels = bus_names
	if n_buses > 50:
		# 太多母线时只显示部分标签
		y_labels = [name if i % (n_buses // 30) == 0 else "" for i, name in enumerate(bus_names)]

	fig = go.Figure(data=go.Heatmap(
		z=v_matrix,
		x=x_labels,
		y=bus_names,
		colorscale=VOLTAGE_COLORSCALE,
		zmin=v_min,
		zmax=v_max,
		colorbar={
			"title": {"text": "Voltage (p.u.)", "side": "right"},
			"tickformat": ".3f",
		},
		hovertemplate=(
			"Bus: %{y}<br>"
			"%{x}<br>"
			"Voltage: %{z:.4f} p.u.<extra></extra>"
		),
	))

	n_display = len(step_indices)
	subtitle = f" (downsampled to {n_display} steps)" if n_steps > max_display_steps else ""

	layout = get_plotly_layout(
		title=f"Bus Voltage Heatmap{subtitle}",
		height=height,
		env_name="smartgrid",
	)
	layout["xaxis"]["title"] = {"text": "Day of Year"}
	layout["yaxis"]["title"] = {"text": "Bus"}
	layout["yaxis"]["autorange"] = "reversed"
	fig.update_layout(**layout)

	# 添加电压安全区间参考线
	fig.add_annotation(
		text=f"Safe range: [{v_min:.2f}, {v_max:.2f}] p.u.",
		xref="paper", yref="paper",
		x=1.0, y=1.02,
		showarrow=False,
		font={"size": 10, "color": "#a0a0a0"},
	)

	return fig


def _get_bus_names(
	snapshots: List[Dict[str, Any]],
	bus_filter: Optional[List[str]],
) -> List[str]:
	"""提取所有母线名称"""
	all_buses: set = set()
	for snap in snapshots[:5]:  # 只扫前几个快照
		buses = snap.get("buses", {})
		all_buses.update(buses.keys())

	names = sorted(all_buses)

	if bus_filter:
		names = [n for n in names if n in bus_filter]

	return names


def _downsample(
	matrix: np.ndarray,
	step_indices: List[int],
	target_steps: int,
) -> tuple:
	"""均匀降采样

	Args:
		matrix: 原始矩阵 (n_buses, n_steps)
		step_indices: 原始步索引
		target_steps: 目标步数

	Returns:
		(降采样矩阵, 降采样步索引)
	"""
	n_steps = matrix.shape[1]
	indices = np.linspace(0, n_steps - 1, target_steps, dtype=int)
	indices = np.unique(indices)

	return matrix[:, indices], [step_indices[i] for i in indices]


def _empty_figure(message: str, height: int) -> go.Figure:
	"""创建空图表"""
	fig = go.Figure()
	fig.add_annotation(
		text=message,
		xref="paper", yref="paper",
		x=0.5, y=0.5,
		showarrow=False,
		font={"size": 16, "color": "#a0a0a0"},
	)
	layout = get_plotly_layout(title="Voltage Heatmap", height=height, env_name="smartgrid")
	fig.update_layout(**layout)
	return fig
