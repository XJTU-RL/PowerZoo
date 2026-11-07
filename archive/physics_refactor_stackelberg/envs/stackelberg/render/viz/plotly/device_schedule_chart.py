# -*- coding: utf-8 -*-
"""
设备调度图

展示 PV 输出、ESS 充放电、负荷变化的 24 小时时间序列。
Stackelberg 特色：同时展示 UC 调度决策和 Consumer 响应。
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import DEVICE_COLORS, get_plotly_layout


def create_device_schedule(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建设备调度时间序列图

	3 行子图：
	1. PV 输出 + ESS 功率
	2. 负荷变化
	3. ESS SOC

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Device Schedule (No Data)", env_name="stackelberg"
		))
		return fig

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	# 提取数据
	pv_kw = []
	ess_kw = []
	ess_soc = []
	total_load = []
	total_loss = []

	for snap in snapshots:
		circuit = snap.get("circuit", {})
		devices = snap.get("devices", {})
		market = snap.get("market_data", {})

		pv_kw.append(circuit.get("total_pv_kw", 0.0))
		ess_kw.append(circuit.get("total_storage_kw", 0.0))
		total_load.append(circuit.get("total_load_kw", 0.0))
		total_loss.append(circuit.get("total_loss_kw", 0.0))

		# ESS SOC
		soc = snap.get("ess_soc", 0.5)
		if isinstance(soc, (int, float)):
			ess_soc.append(soc)
		else:
			# 从设备数据提取
			storage_list = devices.get("storage", [])
			if storage_list:
				ess_soc.append(storage_list[0].get("soc", 0.5))
			else:
				ess_soc.append(0.5)

	fig = make_subplots(
		rows=3, cols=1,
		shared_xaxes=True,
		vertical_spacing=0.08,
		subplot_titles=(
			"PV Output & ESS Power",
			"System Load & Losses",
			"ESS State of Charge",
		),
	)

	# Row 1: PV + ESS
	fig.add_trace(go.Scatter(
		x=steps, y=pv_kw,
		mode="lines+markers", name="PV Output",
		line=dict(color=DEVICE_COLORS.get("pv", "#F59E0B"), width=2),
		fill="tozeroy",
		fillcolor="rgba(245,158,11,0.1)",
	), row=1, col=1)

	fig.add_trace(go.Bar(
		x=steps, y=ess_kw,
		name="ESS Power",
		marker_color=[
			DEVICE_COLORS.get("ess_charge", "#3B82F6")
			if v > 0
			else DEVICE_COLORS.get("ess_discharge", "#EF4444")
			for v in ess_kw
		],
		opacity=0.7,
	), row=1, col=1)

	# Row 2: Load + Losses
	fig.add_trace(go.Scatter(
		x=steps, y=total_load,
		mode="lines+markers", name="Total Load",
		line=dict(color="#7C3AED", width=2),
	), row=2, col=1)

	fig.add_trace(go.Scatter(
		x=steps, y=total_loss,
		mode="lines", name="Total Loss",
		line=dict(color="#EF4444", width=1, dash="dash"),
		fill="tozeroy",
		fillcolor="rgba(239,68,68,0.1)",
	), row=2, col=1)

	# Row 3: ESS SOC
	fig.add_trace(go.Scatter(
		x=steps, y=[s * 100 for s in ess_soc],
		mode="lines+markers", name="ESS SOC",
		line=dict(color="#10B981", width=2),
		fill="tozeroy",
		fillcolor="rgba(16,185,129,0.1)",
	), row=3, col=1)

	# SOC 限值
	fig.add_hline(y=20, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=3, col=1)
	fig.add_hline(y=90, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=3, col=1)

	layout = get_plotly_layout(
		title="Device Schedule (24h)",
		height=700,
		env_name="stackelberg",
	)
	layout.update(
		hovermode="x unified",
	)
	fig.update_layout(**layout)

	fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
	fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
	fig.update_yaxes(title_text="SOC (%)", row=3, col=1)
	fig.update_xaxes(title_text="Step (Hour)", row=3, col=1)

	return fig
