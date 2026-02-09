"""
SmartGrid Voltage Profile (Plotly)
电压剖面图

绘制所有母线在指定时间步的电压分布，
或选定母线的电压时间序列。X 轴使用 "Day of Year"。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import get_plotly_layout, COLORS

logger = logging.getLogger(__name__)


def create_voltage_profile_at_step(
	snapshot: Dict[str, Any],
	v_min: float = 0.95,
	v_max: float = 1.05,
	height: int = 450,
) -> go.Figure:
	"""创建指定步的所有母线电压分布图

	Args:
		snapshot: 单步快照
		v_min: 安全下限
		v_max: 安全上限
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	fig = go.Figure()
	step = snapshot.get("step", 0)
	buses = snapshot.get("buses", {})

	bus_names: List[str] = []
	v_means: List[float] = []

	for name in sorted(buses.keys()):
		info = buses[name]
		v_pu = info.get("v_mag_pu", [])
		if v_pu:
			bus_names.append(name)
			v_means.append(float(np.mean(v_pu)))

	# 颜色编码：安全=绿、警告=黄、违规=红
	bar_colors = []
	for v in v_means:
		if v < v_min or v > v_max:
			bar_colors.append(COLORS["danger"])
		elif v < v_min + 0.01 or v > v_max - 0.01:
			bar_colors.append(COLORS["warning"])
		else:
			bar_colors.append(COLORS["success"])

	fig.add_trace(go.Bar(
		x=bus_names,
		y=v_means,
		marker_color=bar_colors,
		hovertemplate="Bus: %{x}<br>Voltage: %{y:.4f} p.u.<extra></extra>",
	))

	# 安全区间
	fig.add_hline(y=v_min, line_dash="dash", line_color=COLORS["warning"],
				  annotation_text=f"V_min={v_min}")
	fig.add_hline(y=v_max, line_dash="dash", line_color=COLORS["warning"],
				  annotation_text=f"V_max={v_max}")
	fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["text_secondary"],
				  annotation_text="Nominal")

	layout = get_plotly_layout(
		title=f"Voltage Profile (Day {step + 1})",
		height=height,
		env_name="smartgrid",
	)
	layout["xaxis"]["title"] = {"text": "Bus"}
	layout["yaxis"]["title"] = {"text": "Voltage (p.u.)"}
	layout["showlegend"] = False
	fig.update_layout(**layout)

	return fig


def create_voltage_timeseries(
	snapshots: List[Dict[str, Any]],
	bus_names: Optional[List[str]] = None,
	max_buses: int = 10,
	v_min: float = 0.95,
	v_max: float = 1.05,
	height: int = 450,
) -> go.Figure:
	"""创建选定母线的电压时间序列

	Args:
		snapshots: 快照列表
		bus_names: 指定母线（None 则自动选择）
		max_buses: 最大显示母线数
		v_min: 安全下限
		v_max: 安全上限
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	fig = go.Figure()

	if not snapshots:
		layout = get_plotly_layout(title="Voltage Time Series", height=height, env_name="smartgrid")
		fig.update_layout(**layout)
		return fig

	# 自动选择母线
	if bus_names is None:
		bus_names = _select_representative_buses(snapshots, max_buses)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	x_labels = [f"Day {s + 1}" for s in steps]

	color_palette = [
		COLORS["primary"], COLORS["success"], COLORS["warning"],
		COLORS["danger"], COLORS["info"], COLORS["secondary"],
		"#06B6D4", "#84CC16", "#F97316", "#EC4899",
	]

	for idx, bus_name in enumerate(bus_names):
		v_series: List[float] = []
		for snap in snapshots:
			buses = snap.get("buses", {})
			bus_info = buses.get(bus_name, {})
			v_pu = bus_info.get("v_mag_pu", [])
			v_series.append(float(np.mean(v_pu)) if v_pu else np.nan)

		color = color_palette[idx % len(color_palette)]
		fig.add_trace(go.Scatter(
			x=x_labels,
			y=v_series,
			mode="lines",
			name=bus_name,
			line={"color": color, "width": 1.5},
			hovertemplate=f"{bus_name}<br>%{{x}}<br>V = %{{y:.4f}} p.u.<extra></extra>",
		))

	# 安全区间
	fig.add_hline(y=v_min, line_dash="dash", line_color=COLORS["warning"],
				  annotation_text=f"V_min={v_min}")
	fig.add_hline(y=v_max, line_dash="dash", line_color=COLORS["warning"],
				  annotation_text=f"V_max={v_max}")

	layout = get_plotly_layout(
		title="Bus Voltage Time Series",
		height=height,
		env_name="smartgrid",
	)
	layout["xaxis"]["title"] = {"text": "Day of Year"}
	layout["yaxis"]["title"] = {"text": "Voltage (p.u.)"}
	fig.update_layout(**layout)

	return fig


def _select_representative_buses(
	snapshots: List[Dict[str, Any]],
	max_buses: int,
) -> List[str]:
	"""自动选择最具代表性的母线（电压波动最大的）"""
	bus_v_range: Dict[str, float] = {}

	for snap in snapshots:
		buses = snap.get("buses", {})
		for name, info in buses.items():
			v_pu = info.get("v_mag_pu", [])
			if v_pu:
				v_mean = float(np.mean(v_pu))
				if name not in bus_v_range:
					bus_v_range[name] = 0.0
				bus_v_range[name] = max(bus_v_range[name], abs(v_mean - 1.0))

	sorted_buses = sorted(bus_v_range.items(), key=lambda x: x[1], reverse=True)
	return [name for name, _ in sorted_buses[:max_buses]]
