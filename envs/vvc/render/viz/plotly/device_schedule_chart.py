# -*- coding: utf-8 -*-
"""
VVC 设备调度时序图 (Plotly)

VVC 独有图表：4 个子图展示各类设备在 episode 中的状态变化:
1. 电容器开关时序 (阶梯图)
2. 调压器分接头轨迹 (折线图)
3. 电池 SOC + 功率 (双 Y 轴)
4. PV 输出 + 削减率 (双 Y 轴)
"""

from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import COLORS, DEVICE_COLORS, get_plotly_layout


def create_device_schedule(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建 VVC 设备调度时序图。

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure (4 子图)
	"""
	if not snapshots:
		fig = go.Figure()
		layout = get_plotly_layout(
			title="Device Schedule (No Data)", height=400, env_name="vvc"
		)
		fig.update_layout(**layout)
		return fig

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	# 收集设备名称
	cap_names: List[str] = []
	reg_names: List[str] = []
	bat_names: List[str] = []
	pv_names: List[str] = []

	for snap in snapshots:
		devices = snap.get("devices", {})
		if devices.get("capacitors"):
			cap_names = sorted(devices["capacitors"].keys())
		if devices.get("regulators"):
			reg_names = sorted(devices["regulators"].keys())
		if devices.get("batteries"):
			bat_names = sorted(devices["batteries"].keys())
		if devices.get("pvsystems"):
			pv_names = sorted(devices["pvsystems"].keys())
		if cap_names or reg_names or bat_names or pv_names:
			break

	# 创建子图
	n_rows = sum([
		1 if cap_names else 0,
		1 if reg_names else 0,
		1 if bat_names else 0,
		1 if pv_names else 0,
	])
	n_rows = max(n_rows, 1)

	subplot_titles = []
	if cap_names:
		subplot_titles.append("Capacitor Switch States")
	if reg_names:
		subplot_titles.append("Regulator Tap Positions")
	if bat_names:
		subplot_titles.append("Battery SOC & Power")
	if pv_names:
		subplot_titles.append("PV Output & Curtailment")

	specs = [[{"secondary_y": True}] for _ in range(n_rows)]

	fig = make_subplots(
		rows=n_rows, cols=1,
		subplot_titles=subplot_titles,
		specs=specs,
		vertical_spacing=0.08,
		shared_xaxes=True,
	)

	row_idx = 1

	# --- 子图 1: 电容器开关 ---
	if cap_names:
		palette = ["#22D3EE", "#06B6D4", "#0891B2", "#0E7490"]
		for i, cap_name in enumerate(cap_names):
			states = []
			for snap in snapshots:
				devices = snap.get("devices", {})
				cap_data = devices.get("capacitors", {}).get(cap_name, {})
				is_on = 1 if cap_data.get("is_on", False) else 0
				states.append(is_on)

			color = palette[i % len(palette)]
			fig.add_trace(
				go.Scatter(
					x=steps, y=states,
					mode="lines",
					line=dict(color=color, width=2, shape="hv"),
					name=f"Cap {cap_name}",
					legendgroup="cap",
				),
				row=row_idx, col=1,
			)

		fig.update_yaxes(
			title_text="State (0=OFF, 1=ON)",
			range=[-0.1, 1.1],
			dtick=1,
			row=row_idx, col=1,
		)
		row_idx += 1

	# --- 子图 2: 调压器分接头 ---
	if reg_names:
		palette = ["#A78BFA", "#8B5CF6", "#7C3AED", "#6D28D9"]
		for i, reg_name in enumerate(reg_names):
			taps = []
			for snap in snapshots:
				devices = snap.get("devices", {})
				reg_data = devices.get("regulators", {}).get(reg_name, {})
				taps.append(reg_data.get("tap", 0))

			color = palette[i % len(palette)]
			fig.add_trace(
				go.Scatter(
					x=steps, y=taps,
					mode="lines+markers",
					line=dict(color=color, width=2),
					marker=dict(size=5),
					name=f"Reg {reg_name}",
					legendgroup="reg",
				),
				row=row_idx, col=1,
			)

		fig.update_yaxes(
			title_text="Tap Position",
			row=row_idx, col=1,
		)
		row_idx += 1

	# --- 子图 3: 电池 SOC + 功率 ---
	if bat_names:
		soc_colors = ["#06B6D4", "#0891B2", "#0E7490"]
		power_colors = ["#F59E0B", "#D97706", "#B45309"]

		for i, bat_name in enumerate(bat_names):
			socs = []
			powers = []
			for snap in snapshots:
				devices = snap.get("devices", {})
				bat_data = devices.get("batteries", {}).get(bat_name, {})
				socs.append(bat_data.get("soc", 0.0) * 100)
				powers.append(bat_data.get("kw", 0.0))

			# SOC (左 Y 轴)
			fig.add_trace(
				go.Scatter(
					x=steps, y=socs,
					mode="lines",
					line=dict(color=soc_colors[i % len(soc_colors)], width=2),
					name=f"SOC {bat_name}",
					legendgroup="bat_soc",
				),
				row=row_idx, col=1,
				secondary_y=False,
			)

			# 功率 (右 Y 轴)
			fig.add_trace(
				go.Scatter(
					x=steps, y=powers,
					mode="lines",
					line=dict(
						color=power_colors[i % len(power_colors)],
						width=2,
						dash="dash",
					),
					name=f"Power {bat_name}",
					legendgroup="bat_power",
				),
				row=row_idx, col=1,
				secondary_y=True,
			)

		fig.update_yaxes(
			title_text="SOC (%)",
			range=[0, 100],
			row=row_idx, col=1,
			secondary_y=False,
		)
		fig.update_yaxes(
			title_text="Power (kW)",
			row=row_idx, col=1,
			secondary_y=True,
		)
		row_idx += 1

	# --- 子图 4: PV 输出 + 削减 ---
	if pv_names:
		output_colors = ["#F59E0B", "#D97706", "#B45309"]
		curtail_colors = ["#EF4444", "#DC2626", "#B91C1C"]

		for i, pv_name in enumerate(pv_names):
			outputs = []
			curtails = []
			for snap in snapshots:
				devices = snap.get("devices", {})
				pv_data = devices.get("pvsystems", {}).get(pv_name, {})
				outputs.append(pv_data.get("kw", 0.0))
				curtails.append(pv_data.get("curtail_pct", 0.0))

			# 输出 (左 Y 轴)
			fig.add_trace(
				go.Scatter(
					x=steps, y=outputs,
					mode="lines",
					line=dict(color=output_colors[i % len(output_colors)], width=2),
					fill="tozeroy",
					fillcolor=f"rgba(245, 158, 11, 0.1)",
					name=f"PV Out {pv_name}",
					legendgroup="pv_out",
				),
				row=row_idx, col=1,
				secondary_y=False,
			)

			# 削减率 (右 Y 轴)
			fig.add_trace(
				go.Scatter(
					x=steps, y=curtails,
					mode="lines",
					line=dict(
						color=curtail_colors[i % len(curtail_colors)],
						width=2,
						dash="dot",
					),
					name=f"Curtail {pv_name}",
					legendgroup="pv_curtail",
				),
				row=row_idx, col=1,
				secondary_y=True,
			)

		fig.update_yaxes(
			title_text="Output (kW)",
			row=row_idx, col=1,
			secondary_y=False,
		)
		fig.update_yaxes(
			title_text="Curtailment (%)",
			range=[0, 100],
			row=row_idx, col=1,
			secondary_y=True,
		)

	# --- 全局布局 ---
	layout = get_plotly_layout(
		title="VVC Device Schedule",
		height=300 * n_rows,
		env_name="vvc",
	)
	layout.pop("xaxis", None)
	layout.pop("yaxis", None)

	fig.update_layout(**layout)
	fig.update_xaxes(title_text="Step", row=n_rows, col=1)

	return fig
