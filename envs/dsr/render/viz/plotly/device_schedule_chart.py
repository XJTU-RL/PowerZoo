"""
DSR Device Schedule Chart
设备操作时序图

按智能体类型分组展示设备操作历史:
- Switch: 开关操作时序 (open/close)
- PV: 功率输出级别时序
- Load: 投切状态时序

使用热力图 + 标注呈现动作决策过程。
"""

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import (
	COLORS, DEVICE_COLORS, AGENT_ROLE_COLORS, get_plotly_layout,
)


def create_device_schedule_chart(
	snapshots: List[Dict[str, Any]],
	agent_types: Optional[List[str]] = None,
) -> go.Figure:
	"""创建设备操作时序图

	按 Switch / PV / Load 三类分 subplot 展示。

	Args:
		snapshots: 快照列表
		agent_types: 智能体类型列表

	Returns:
		Plotly Figure 对象
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Device Schedule (No Data)", env_name="dsr"))
		return fig

	# 收集动作数据
	switch_data, pv_data, load_data = _collect_device_data(snapshots, agent_types)

	has_switch = bool(switch_data["names"])
	has_pv = bool(pv_data["names"])
	has_load = bool(load_data["names"])

	n_rows = sum([has_switch, has_pv, has_load])
	if n_rows == 0:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(title="Device Schedule (No Devices)", env_name="dsr"))
		return fig

	subtitles = []
	if has_switch:
		subtitles.append("Switch Operations")
	if has_pv:
		subtitles.append("PV Power Levels")
	if has_load:
		subtitles.append("Load Status")

	fig = make_subplots(
		rows=n_rows, cols=1,
		subplot_titles=subtitles,
		vertical_spacing=0.08,
		shared_xaxes=True,
	)

	row = 1

	# Switch 子图
	if has_switch:
		_add_switch_subplot(fig, switch_data, row)
		row += 1

	# PV 子图
	if has_pv:
		_add_pv_subplot(fig, pv_data, row)
		row += 1

	# Load 子图
	if has_load:
		_add_load_subplot(fig, load_data, row)

	layout = get_plotly_layout(
		title="Device Operation Schedule",
		height=200 * n_rows + 100,
		env_name="dsr",
	)
	fig.update_layout(**layout)
	fig.update_xaxes(title_text="Step", row=n_rows, col=1)

	return fig


def _collect_device_data(
	snapshots: List[Dict[str, Any]],
	agent_types: Optional[List[str]],
) -> tuple:
	"""收集设备操作数据

	Args:
		snapshots: 快照列表
		agent_types: 智能体类型列表

	Returns:
		(switch_data, pv_data, load_data) 三元组
	"""
	switch_data = {"names": [], "steps": [], "states": {}}
	pv_data = {"names": [], "steps": [], "powers": {}}
	load_data = {"names": [], "steps": [], "states": {}}

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	# 从设备数据中收集
	for snap in snapshots:
		devices = snap.get("devices", {})

		# Switches
		for sw_name in devices.get("switches", {}):
			if sw_name not in switch_data["states"]:
				switch_data["names"].append(sw_name)
				switch_data["states"][sw_name] = []

		# PVs
		for pv_name in devices.get("pvs", {}):
			if pv_name not in pv_data["powers"]:
				pv_data["names"].append(pv_name)
				pv_data["powers"][pv_name] = []

		# Loads
		for ld_name in devices.get("loads", {}):
			if ld_name not in load_data["states"]:
				load_data["names"].append(ld_name)
				load_data["states"][ld_name] = []

	# 填充时序数据
	for snap in snapshots:
		devices = snap.get("devices", {})

		for sw_name in switch_data["names"]:
			sw = devices.get("switches", {}).get(sw_name, {})
			# 1 = closed, 0 = open
			switch_data["states"][sw_name].append(
				1.0 if sw.get("is_closed", True) else 0.0
			)

		for pv_name in pv_data["names"]:
			pv = devices.get("pvs", {}).get(pv_name, {})
			pv_data["powers"][pv_name].append(pv.get("power_ratio", 0.0))

		for ld_name in load_data["names"]:
			ld = devices.get("loads", {}).get(ld_name, {})
			load_data["states"][ld_name].append(
				1.0 if ld.get("is_connected", False) else 0.0
			)

	switch_data["steps"] = steps
	pv_data["steps"] = steps
	load_data["steps"] = steps

	return switch_data, pv_data, load_data


def _add_switch_subplot(fig: go.Figure, data: Dict, row: int) -> None:
	"""添加开关操作子图"""
	steps = data["steps"]
	for sw_name in data["names"]:
		states = data["states"][sw_name]
		colors = [DEVICE_COLORS["switch_closed"] if s > 0.5 else DEVICE_COLORS["switch_open"]
				  for s in states]
		fig.add_trace(go.Scatter(
			x=steps, y=states,
			mode="lines+markers",
			name=f"SW: {sw_name}",
			line=dict(width=2),
			marker=dict(size=8, color=colors),
			hovertemplate=f"{sw_name}<br>Step: %{{x}}<br>State: %{{customdata}}<extra></extra>",
			customdata=["Closed" if s > 0.5 else "Open" for s in states],
		), row=row, col=1)

	fig.update_yaxes(
		tickvals=[0, 1],
		ticktext=["Open", "Closed"],
		row=row, col=1,
	)


def _add_pv_subplot(fig: go.Figure, data: Dict, row: int) -> None:
	"""添加 PV 功率子图"""
	steps = data["steps"]
	for pv_name in data["names"]:
		powers = data["powers"][pv_name]
		fig.add_trace(go.Scatter(
			x=steps, y=[p * 100 for p in powers],
			mode="lines+markers",
			name=f"PV: {pv_name}",
			line=dict(color=DEVICE_COLORS["pv"], width=2),
			marker=dict(size=6),
			hovertemplate=f"{pv_name}<br>Step: %{{x}}<br>Power: %{{y:.0f}}%<extra></extra>",
		), row=row, col=1)

	fig.update_yaxes(title_text="Power (%)", row=row, col=1)


def _add_load_subplot(fig: go.Figure, data: Dict, row: int) -> None:
	"""添加负荷状态子图"""
	steps = data["steps"]
	for i, ld_name in enumerate(data["names"]):
		states = data["states"][ld_name]
		colors = [COLORS["success"] if s > 0.5 else COLORS["danger"] for s in states]
		fig.add_trace(go.Scatter(
			x=steps, y=states,
			mode="lines+markers",
			name=f"Load: {ld_name}",
			line=dict(width=1.5),
			marker=dict(size=6, color=colors),
			hovertemplate=f"{ld_name}<br>Step: %{{x}}<br>Status: %{{customdata}}<extra></extra>",
			customdata=["Connected" if s > 0.5 else "Disconnected" for s in states],
		), row=row, col=1)

	fig.update_yaxes(
		tickvals=[0, 1],
		ticktext=["Off", "On"],
		row=row, col=1,
	)
