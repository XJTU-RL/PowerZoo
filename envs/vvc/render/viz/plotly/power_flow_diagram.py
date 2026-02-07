# -*- coding: utf-8 -*-
"""
VVC 功率流向图 (Plotly)

替代 Sankey 图 (VVC 无分区)，用水平条形图展示各母线的
功率注入/吸收情况，区分负荷、发电、PV、储能。
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from envs.render_common.viz.theme import COLORS, DEVICE_COLORS, get_plotly_layout


def create_power_flow_diagram(
	snapshot: Dict[str, Any],
) -> go.Figure:
	"""创建功率流向图。

	以条形图展示系统各组件的功率注入/吸收，
	正值表示向系统注入，负值表示从系统吸收。

	Args:
		snapshot: 单步快照字典

	Returns:
		Plotly Figure 对象
	"""
	fig = go.Figure()

	circuit = snapshot.get("circuit", {})
	devices = snapshot.get("devices", {})
	step = snapshot.get("step", 0)

	# 收集功率数据
	categories: List[str] = []
	values: List[float] = []
	bar_colors: List[str] = []
	hover_texts: List[str] = []

	# 总负荷 (吸收)
	total_load = circuit.get("total_load_kw", 0.0)
	if total_load != 0:
		categories.append("Total Load")
		values.append(-abs(total_load))
		bar_colors.append(DEVICE_COLORS["load"])
		hover_texts.append(f"Total Load: {total_load:.1f} kW")

	# 变电站 (注入/吸收)
	total_gen = circuit.get("total_gen_kw", 0.0)
	if total_gen != 0:
		categories.append("Substation")
		values.append(total_gen)
		bar_colors.append(COLORS["primary"])
		hover_texts.append(f"Substation: {total_gen:.1f} kW")

	# PV 系统 (注入)
	pv_systems = devices.get("pvsystems", {})
	for pv_name, pv_data in pv_systems.items():
		kw = pv_data.get("kw", 0.0)
		if kw > 0.01:
			categories.append(f"PV: {pv_name}")
			values.append(kw)
			bar_colors.append(DEVICE_COLORS["pv"])
			curtail = pv_data.get("curtail_pct", 0.0)
			hover_texts.append(
				f"PV {pv_name}: {kw:.1f} kW "
				f"(Curtail: {curtail:.1f}%)"
			)

	# 电池 (注入或吸收)
	batteries = devices.get("batteries", {})
	for bat_name, bat_data in batteries.items():
		kw = bat_data.get("kw", 0.0)
		if abs(kw) > 0.01:
			categories.append(f"Battery: {bat_name}")
			values.append(kw)  # 正=放电(注入), 负=充电(吸收)
			bar_colors.append(
				DEVICE_COLORS["storage_discharge"] if kw > 0
				else DEVICE_COLORS["storage_charge"]
			)
			soc = bat_data.get("soc", 0.0)
			hover_texts.append(
				f"Battery {bat_name}: {kw:.1f} kW "
				f"({'Discharge' if kw > 0 else 'Charge'}, "
				f"SOC: {soc:.1%})"
			)

	# 电容器 (无功补偿)
	capacitors = devices.get("capacitors", {})
	for cap_name, cap_data in capacitors.items():
		if cap_data.get("is_on", False):
			kvar = cap_data.get("kvar", 0.0)
			if kvar != 0:
				categories.append(f"Cap: {cap_name}")
				values.append(kvar * 0.1)  # 缩放用于可视化
				bar_colors.append(DEVICE_COLORS["capacitor_on"])
				hover_texts.append(
					f"Cap {cap_name}: {kvar:.0f} kvar (ON)"
				)

	# 线路损耗 (吸收)
	total_loss = circuit.get("total_loss_kw", 0.0)
	if total_loss > 0:
		categories.append("Line Losses")
		values.append(-total_loss)
		bar_colors.append(COLORS["danger"])
		hover_texts.append(f"Line Losses: {total_loss:.1f} kW")

	if not categories:
		layout = get_plotly_layout(
			title="Power Flow (No Data)", height=400, env_name="vvc"
		)
		fig.update_layout(**layout)
		return fig

	# 水平条形图
	fig.add_trace(go.Bar(
		y=categories,
		x=values,
		orientation="h",
		marker_color=bar_colors,
		hoverinfo="text",
		hovertext=hover_texts,
		text=[f"{v:+.1f}" for v in values],
		textposition="outside",
		textfont=dict(size=10, color=COLORS["text_secondary"]),
	))

	# 零线
	fig.add_vline(x=0, line_color=COLORS["text_secondary"], line_width=1)

	title = f"Power Flow Diagram (Step {step})"
	layout = get_plotly_layout(title=title, height=max(350, len(categories) * 40 + 100), env_name="vvc")
	layout.update({
		"xaxis_title": "Power (kW)",
		"yaxis": dict(autorange="reversed"),
		"showlegend": False,
	})
	fig.update_layout(**layout)

	return fig
