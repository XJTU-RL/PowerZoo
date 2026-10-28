"""
DSR Voltage Profile
电压分布图

显示所有带电母线的电压分布：
- 箱线图/散点图展示每步的电压分布
- 安全界限区域标注
- 电压违规母线高亮
"""

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import COLORS, get_plotly_layout


def create_voltage_profile(
	snapshots: List[Dict[str, Any]],
	v_min_limit: float = 0.95,
	v_max_limit: float = 1.05,
) -> go.Figure:
	"""创建电压分布图

	Args:
		snapshots: 快照列表
		v_min_limit: 电压下限
		v_max_limit: 电压上限

	Returns:
		Plotly Figure 对象
	"""
	fig = go.Figure()

	if not snapshots:
		fig.update_layout(**get_plotly_layout(title="Voltage Profile (No Data)", env_name="dsr"))
		return fig

	steps = []
	v_means = []
	v_mins = []
	v_maxs = []
	n_violations_list = []

	for snap in snapshots:
		step = snap.get("step", 0)
		buses = snap.get("buses", {})

		voltages: List[float] = []
		for bus_data in buses.values():
			if not bus_data.get("is_energized", False):
				continue
			for v in bus_data.get("v_mag_pu", []):
				if 0.1 < v < 2.0:
					voltages.append(v)

		if voltages:
			steps.append(step)
			v_arr = np.array(voltages)
			v_means.append(float(np.mean(v_arr)))
			v_mins.append(float(np.min(v_arr)))
			v_maxs.append(float(np.max(v_arr)))
			n_viol = int(np.sum((v_arr < v_min_limit) | (v_arr > v_max_limit)))
			n_violations_list.append(n_viol)

	if not steps:
		fig.update_layout(**get_plotly_layout(title="Voltage Profile (No Voltage Data)", env_name="dsr"))
		return fig

	# 安全范围区域
	fig.add_trace(go.Scatter(
		x=steps + steps[::-1],
		y=[v_max_limit] * len(steps) + [v_min_limit] * len(steps),
		fill="toself",
		fillcolor="rgba(16,185,129,0.1)",
		line=dict(width=0),
		name="Safe Range",
		hoverinfo="skip",
	))

	# 电压范围区域
	fig.add_trace(go.Scatter(
		x=steps + steps[::-1],
		y=v_maxs + v_mins[::-1],
		fill="toself",
		fillcolor="rgba(79,70,229,0.15)",
		line=dict(width=0),
		name="V Range",
		hoverinfo="skip",
	))

	# 平均电压
	fig.add_trace(go.Scatter(
		x=steps, y=v_means,
		mode="lines+markers",
		line=dict(color=COLORS["primary"], width=2),
		marker=dict(size=6),
		name="V Mean",
		hovertemplate="Step: %{x}<br>V Mean: %{y:.4f} pu<extra></extra>",
	))

	# 最小/最大电压
	fig.add_trace(go.Scatter(
		x=steps, y=v_mins,
		mode="lines",
		line=dict(color=COLORS["danger"], width=1, dash="dot"),
		name="V Min",
	))

	fig.add_trace(go.Scatter(
		x=steps, y=v_maxs,
		mode="lines",
		line=dict(color=COLORS["warning"], width=1, dash="dot"),
		name="V Max",
	))

	# 安全界限线
	fig.add_hline(y=v_min_limit, line_dash="dash", line_color=COLORS["danger"], opacity=0.5,
				  annotation_text=f"V_min={v_min_limit}", annotation_position="bottom right")
	fig.add_hline(y=v_max_limit, line_dash="dash", line_color=COLORS["danger"], opacity=0.5,
				  annotation_text=f"V_max={v_max_limit}", annotation_position="top right")

	# 违规标注
	for i, (step, n_viol) in enumerate(zip(steps, n_violations_list)):
		if n_viol > 0:
			fig.add_annotation(
				x=step, y=v_mins[i],
				text=f"{n_viol} viol.",
				showarrow=True,
				arrowhead=2,
				arrowsize=0.8,
				font=dict(size=9, color=COLORS["danger"]),
			)

	layout = get_plotly_layout(title="Voltage Profile", height=450, env_name="dsr")
	layout["xaxis_title"] = "Step"
	layout["yaxis_title"] = "Voltage (pu)"
	fig.update_layout(**layout)

	return fig
