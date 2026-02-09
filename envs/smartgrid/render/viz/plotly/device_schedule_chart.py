"""
SmartGrid Device Schedule Chart (Plotly)
设备调度时间表

绘制 CRBP 各类设备（电容器、调压器、电池、PV）
在整个 episode 中的状态/动作时间序列。
360 步的 X 轴使用 "Day of Year" 标签。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import get_plotly_layout, COLORS, DEVICE_COLORS

logger = logging.getLogger(__name__)


def create_device_schedule_chart(
	snapshots: List[Dict[str, Any]],
	height: int = 800,
) -> go.Figure:
	"""创建设备调度综合图表

	4 行子图：电容器状态、调压器 tap、电池 SOC/功率、PV 出力

	Args:
		snapshots: 快照列表
		height: 图表高度

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		layout = get_plotly_layout(title="Device Schedule", height=height, env_name="smartgrid")
		fig.update_layout(**layout)
		return fig

	# 检测设备类型
	has_caps, has_regs, has_bats, has_pvs = _detect_devices(snapshots)
	n_rows = sum([has_caps, has_regs, has_bats, has_pvs])
	if n_rows == 0:
		n_rows = 1

	titles: List[str] = []
	if has_caps:
		titles.append("Capacitor Status")
	if has_regs:
		titles.append("Regulator Tap Position")
	if has_bats:
		titles.append("Battery SOC & Power")
	if has_pvs:
		titles.append("PV Output")

	fig = make_subplots(
		rows=n_rows, cols=1,
		shared_xaxes=True,
		vertical_spacing=0.06,
		subplot_titles=titles,
	)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	x_labels = [f"Day {s + 1}" for s in steps]
	row_idx = 1

	# 电容器
	if has_caps:
		_add_capacitor_traces(fig, snapshots, x_labels, row_idx)
		row_idx += 1

	# 调压器
	if has_regs:
		_add_regulator_traces(fig, snapshots, x_labels, row_idx)
		row_idx += 1

	# 电池
	if has_bats:
		_add_battery_traces(fig, snapshots, x_labels, row_idx)
		row_idx += 1

	# PV
	if has_pvs:
		_add_pv_traces(fig, snapshots, x_labels, row_idx)
		row_idx += 1

	layout = get_plotly_layout(
		title="Device Schedule (CRBP)",
		height=height,
		env_name="smartgrid",
	)
	layout.pop("xaxis", None)
	layout.pop("yaxis", None)
	fig.update_layout(**layout)

	# 最后一行的 x 轴标签
	fig.update_xaxes(title_text="Day of Year", row=n_rows, col=1)

	return fig


def _detect_devices(snapshots: List[Dict[str, Any]]) -> tuple:
	"""检测快照中包含哪些设备类型"""
	has_caps = has_regs = has_bats = has_pvs = False
	for snap in snapshots[:3]:
		devices = snap.get("devices", {})
		if devices.get("capacitors"):
			has_caps = True
		if devices.get("regulators"):
			has_regs = True
		if devices.get("batteries"):
			has_bats = True
		if devices.get("pvs"):
			has_pvs = True
	return has_caps, has_regs, has_bats, has_pvs


def _add_capacitor_traces(
	fig: go.Figure,
	snapshots: List[Dict[str, Any]],
	x_labels: List[str],
	row: int,
) -> None:
	"""添加电容器状态 traces"""
	cap_series: Dict[str, List[int]] = {}

	for snap in snapshots:
		devices = snap.get("devices", {})
		caps = devices.get("capacitors", {})
		for name, info in caps.items():
			if name not in cap_series:
				cap_series[name] = []
			cap_series[name].append(int(info.get("status", 0)))

	colors = [DEVICE_COLORS["capacitor_on"], DEVICE_COLORS["capacitor_off"],
			  COLORS["info"], COLORS["secondary"]]

	for idx, (name, series) in enumerate(cap_series.items()):
		color = colors[idx % len(colors)]
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(series)],
				y=series,
				mode="lines",
				name=name,
				line={"color": color, "width": 2, "shape": "hv"},
				hovertemplate=f"{name}<br>%{{x}}<br>Status: %{{y}}<extra></extra>",
			),
			row=row, col=1,
		)

	fig.update_yaxes(
		title_text="Status (0/1)", row=row, col=1,
		range=[-0.1, 1.1], dtick=1,
	)


def _add_regulator_traces(
	fig: go.Figure,
	snapshots: List[Dict[str, Any]],
	x_labels: List[str],
	row: int,
) -> None:
	"""添加调压器 tap traces"""
	reg_series: Dict[str, List[int]] = {}

	for snap in snapshots:
		devices = snap.get("devices", {})
		regs = devices.get("regulators", {})
		for name, info in regs.items():
			if name not in reg_series:
				reg_series[name] = []
			reg_series[name].append(int(info.get("tap", 16)))

	colors = [DEVICE_COLORS["regulator_tap"], COLORS["info"],
			  COLORS["warning"], COLORS["secondary"]]

	for idx, (name, series) in enumerate(reg_series.items()):
		color = colors[idx % len(colors)]
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(series)],
				y=series,
				mode="lines",
				name=name,
				line={"color": color, "width": 1.5},
				hovertemplate=f"{name}<br>%{{x}}<br>Tap: %{{y}}<extra></extra>",
			),
			row=row, col=1,
		)

	fig.update_yaxes(title_text="Tap Position", row=row, col=1)


def _add_battery_traces(
	fig: go.Figure,
	snapshots: List[Dict[str, Any]],
	x_labels: List[str],
	row: int,
) -> None:
	"""添加电池 SOC 和功率 traces"""
	bat_soc: Dict[str, List[float]] = {}
	bat_power: Dict[str, List[float]] = {}

	for snap in snapshots:
		devices = snap.get("devices", {})
		bats = devices.get("batteries", {})
		for name, info in bats.items():
			if name not in bat_soc:
				bat_soc[name] = []
				bat_power[name] = []
			bat_soc[name].append(float(info.get("soc", 0.0)))
			bat_power[name].append(float(info.get("power_kw", 0.0)))

	for idx, (name, series) in enumerate(bat_soc.items()):
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(series)],
				y=series,
				mode="lines",
				name=f"{name} SOC",
				line={"color": DEVICE_COLORS["battery_soc"], "width": 2},
				hovertemplate=f"{name} SOC<br>%{{x}}<br>SOC: %{{y:.1%}}<extra></extra>",
			),
			row=row, col=1,
		)

	for idx, (name, series) in enumerate(bat_power.items()):
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(series)],
				y=series,
				mode="lines",
				name=f"{name} Power",
				line={"color": DEVICE_COLORS["storage_discharge"], "width": 1.5, "dash": "dot"},
				yaxis="y2",
				hovertemplate=f"{name} Power<br>%{{x}}<br>P: %{{y:.1f}} kW<extra></extra>",
			),
			row=row, col=1,
		)

	fig.update_yaxes(title_text="SOC / Power (kW)", row=row, col=1)


def _add_pv_traces(
	fig: go.Figure,
	snapshots: List[Dict[str, Any]],
	x_labels: List[str],
	row: int,
) -> None:
	"""添加 PV 出力 traces"""
	pv_power: Dict[str, List[float]] = {}

	for snap in snapshots:
		devices = snap.get("devices", {})
		pvs = devices.get("pvs", {})
		for name, info in pvs.items():
			if name not in pv_power:
				pv_power[name] = []
			pv_power[name].append(float(info.get("power_ratio", 0.0)))

	for idx, (name, series) in enumerate(pv_power.items()):
		fig.add_trace(
			go.Scatter(
				x=x_labels[:len(series)],
				y=series,
				mode="lines",
				name=name,
				line={"color": DEVICE_COLORS["pv"], "width": 1.5},
				fill="tozeroy",
				fillcolor="rgba(245, 158, 11, 0.15)",
				hovertemplate=f"{name}<br>%{{x}}<br>Output: %{{y:.1%}}<extra></extra>",
			),
			row=row, col=1,
		)

	fig.update_yaxes(title_text="PV Output Ratio", row=row, col=1, range=[0, 1.1])
