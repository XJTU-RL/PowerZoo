"""
设备调度时间序列图 (多子图)
PV Output, Storage SOC+Power, EV Load, Regulator Taps, Capacitor States
"""

from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.district_dispatch.render.viz.theme import (
	COLORS,
	DEVICE_COLORS,
	get_plotly_layout,
)


def _extract_device_timeseries(
	snapshots: List[Dict[str, Any]],
) -> Tuple[List[float], Dict[str, Any]]:
	"""从快照序列提取设备调度时间序列。

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		tuple: (timestamps, device_series)
			- timestamps: 时间戳列表
			- device_series: 设备数据字典
	"""
	timestamps: List[float] = []
	series: Dict[str, List[float]] = {
		"pv_available": [],
		"pv_actual": [],
		"pv_curtailment": [],
		"storage_soc": [],
		"storage_power": [],
		"ev_demand": [],
		"ev_modulation": [],
	}
	reg_series: Dict[str, List[int]] = {}
	cap_series: Dict[str, List[Dict[str, Any]]] = {}

	for i, snap in enumerate(snapshots):
		timestamps.append(snap.get("timestamp_h", i * 0.25))
		devices = snap.get("devices", {})
		regulators = snap.get("regulators", {})

		# PV
		pv = devices.get("pv", {})
		available = pv.get("available_kw", 0.0)
		actual = pv.get("output_kw", 0.0)
		series["pv_available"].append(available)
		series["pv_actual"].append(actual)
		series["pv_curtailment"].append(max(0.0, available - actual))

		# Storage
		storage = devices.get("storage", {})
		series["storage_soc"].append(storage.get("soc", 0.5))
		series["storage_power"].append(storage.get("power_kw", 0.0))

		# EV
		ev = devices.get("ev", {})
		series["ev_demand"].append(ev.get("load_kw", 0.0))
		series["ev_modulation"].append(ev.get("modulation_ratio", 1.0))

		# Regulators
		for reg_name, reg_data in regulators.items():
			if reg_name not in reg_series:
				reg_series[reg_name] = []
			reg_series[reg_name].append(reg_data.get("tap_number", 0))

		# Capacitors
		circuit = snap.get("circuit", {})
		caps = circuit.get("capacitors", {})
		for cap_name, cap_data in caps.items():
			if cap_name not in cap_series:
				cap_series[cap_name] = []
			cap_series[cap_name].append(cap_data)

	return timestamps, {
		"scalar": series,
		"regulators": reg_series,
		"capacitors": cap_series,
	}


def create_device_schedule(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建设备调度时间序列图（多子图）。

	5 个子图:
	1. PV Output: Available vs Actual, curtailment 阴影区域
	2. Storage SOC + Power: 双轴 (SOC 线 + 充放电柱状)
	3. EV Load: 需求曲线 with modulation ratio
	4. Regulator Taps: 6 个调压器分接头位置 over time
	5. Capacitor States: kvar 注入 + 开关状态

	Args:
		snapshots: 按时间排序的快照列表

	Returns:
		go.Figure: Plotly 多子图
	"""
	timestamps, device_data = _extract_device_timeseries(snapshots)
	scalar = device_data["scalar"]
	reg_data = device_data["regulators"]
	cap_data = device_data["capacitors"]

	n_rows = 5
	fig = make_subplots(
		rows=n_rows, cols=1,
		shared_xaxes=True,
		vertical_spacing=0.05,
		subplot_titles=[
			"PV Output", "Storage SOC & Power",
			"EV Load", "Regulator Taps", "Capacitor States",
		],
		specs=[
			[{"secondary_y": False}],
			[{"secondary_y": True}],
			[{"secondary_y": True}],
			[{"secondary_y": False}],
			[{"secondary_y": False}],
		],
	)

	time_labels = [f"{t:.2f}" for t in timestamps]

	if not time_labels:
		fig.update_layout(**get_plotly_layout("Device Schedule (No Data)", height=900))
		return fig

	# === Row 1: PV Output ===
	fig.add_trace(go.Scatter(
		x=time_labels, y=scalar["pv_available"],
		mode="lines",
		name="PV Available",
		line=dict(color=DEVICE_COLORS["pv"], dash="dash"),
	), row=1, col=1)

	fig.add_trace(go.Scatter(
		x=time_labels, y=scalar["pv_actual"],
		mode="lines",
		name="PV Actual",
		line=dict(color=DEVICE_COLORS["pv"]),
		fill="tonexty",
		fillcolor="rgba(245,158,11,0.2)",
	), row=1, col=1)

	# Curtailment 阴影 (available 和 actual 之间)
	fig.add_trace(go.Scatter(
		x=time_labels, y=scalar["pv_curtailment"],
		mode="lines",
		name="Curtailment",
		line=dict(color="#EF4444", width=1, dash="dot"),
	), row=1, col=1)

	# === Row 2: Storage SOC + Power ===
	fig.add_trace(go.Scatter(
		x=time_labels,
		y=[s * 100 for s in scalar["storage_soc"]],
		mode="lines",
		name="SOC (%)",
		line=dict(color="#10B981", width=2),
	), row=2, col=1, secondary_y=False)

	# 充放电柱状图
	charge_colors = [
		DEVICE_COLORS["storage_charge"] if p >= 0 else DEVICE_COLORS["storage_discharge"]
		for p in scalar["storage_power"]
	]
	fig.add_trace(go.Bar(
		x=time_labels, y=scalar["storage_power"],
		name="Power (kW)",
		marker_color=charge_colors,
		opacity=0.7,
	), row=2, col=1, secondary_y=True)

	# === Row 3: EV Load ===
	fig.add_trace(go.Scatter(
		x=time_labels, y=scalar["ev_demand"],
		mode="lines",
		name="EV Demand",
		line=dict(color=DEVICE_COLORS["ev"], width=2),
	), row=3, col=1, secondary_y=False)

	fig.add_trace(go.Scatter(
		x=time_labels, y=scalar["ev_modulation"],
		mode="lines",
		name="Modulation Ratio",
		line=dict(color="#F97316", dash="dash"),
	), row=3, col=1, secondary_y=True)

	# === Row 4: Regulator Taps ===
	reg_colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#7C3AED", "#3B82F6"]
	for idx, (reg_name, taps) in enumerate(reg_data.items()):
		color = reg_colors[idx % len(reg_colors)]
		fig.add_trace(go.Scatter(
			x=time_labels, y=taps,
			mode="lines+markers",
			name=f"Reg {reg_name}",
			line=dict(color=color, width=1.5),
			marker=dict(size=4),
		), row=4, col=1)

	# === Row 5: Capacitor States ===
	for cap_name, cap_states in cap_data.items():
		kvar_values = [
			s.get("kvar", 0.0) if isinstance(s, dict) else 0.0
			for s in cap_states
		]
		fig.add_trace(go.Bar(
			x=time_labels, y=kvar_values,
			name=f"Cap {cap_name} (kvar)",
			opacity=0.7,
		), row=5, col=1)

	# Layout
	base_layout = get_plotly_layout(
		title="Device Schedule Timeline",
		height=1200,
	)
	base_layout["legend"] = dict(
		bgcolor="rgba(0,0,0,0.3)",
		orientation="h",
		yanchor="bottom",
		y=-0.05,
		xanchor="center",
		x=0.5,
		font=dict(size=9),
	)
	fig.update_layout(
		**base_layout,
		hovermode="x unified",
		showlegend=True,
	)

	# Y-axis labels
	fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
	fig.update_yaxes(title_text="SOC (%)", secondary_y=False, row=2, col=1)
	fig.update_yaxes(title_text="Power (kW)", secondary_y=True, row=2, col=1)
	fig.update_yaxes(title_text="Load (kW)", secondary_y=False, row=3, col=1)
	fig.update_yaxes(title_text="Ratio", secondary_y=True, row=3, col=1)
	fig.update_yaxes(title_text="Tap Position", row=4, col=1)
	fig.update_yaxes(title_text="kvar", row=5, col=1)
	fig.update_xaxes(title_text="Time (h)", row=n_rows, col=1)

	return fig
