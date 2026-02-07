# -*- coding: utf-8 -*-
"""
VVC 拓扑动画帧渲染器 (Matplotlib)

逐帧渲染拓扑图用于动画导出 (GIF/MP4)。
每帧包含节点电压着色和设备状态标注。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from envs.render_common.utils.color_scales import loading_to_color, voltage_to_color
from envs.render_common.viz.theme import COLORS, DEVICE_COLORS, apply_matplotlib_theme

logger = logging.getLogger(__name__)


def render_topology_frame(
	snapshot: Dict[str, Any],
	step: int,
	bus_coords: Optional[Dict[str, Tuple[float, float]]] = None,
	figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
	"""渲染单帧拓扑图。

	Args:
		snapshot: 快照字典
		step: 时间步编号
		bus_coords: 母线坐标字典
		figsize: 图像尺寸

	Returns:
		Matplotlib Figure 对象
	"""
	apply_matplotlib_theme()
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	buses = snapshot.get("buses", {})
	lines = snapshot.get("lines", {})
	devices = snapshot.get("devices", {})
	circuit = snapshot.get("circuit", {})

	if bus_coords is None:
		bus_coords = _auto_layout(buses)

	# --- 线路 ---
	for line_name, ld in lines.items():
		bus1 = ld.get("bus1", "")
		bus2 = ld.get("bus2", "")
		if bus1 not in bus_coords or bus2 not in bus_coords:
			continue

		x1, y1 = bus_coords[bus1]
		x2, y2 = bus_coords[bus2]
		loading = ld.get("loading_pct", 0.0)
		color = loading_to_color(loading)
		width = max(0.5, min(loading / 30.0, 3.0))

		ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)

	# --- 母线节点 ---
	for bus_name, (x, y) in bus_coords.items():
		bd = buses.get(bus_name, {})
		v_pu = bd.get("v_mag_pu", [1.0])
		avg_v = sum(v_pu) / len(v_pu) if v_pu else 1.0
		color = voltage_to_color(avg_v)

		ax.scatter(x, y, c=color, s=80, zorder=3, edgecolors="white", linewidths=0.5)
		ax.annotate(
			bus_name, (x, y),
			textcoords="offset points",
			xytext=(0, 8),
			fontsize=6,
			color=COLORS["text_secondary"],
			ha="center",
		)

	# --- 设备标记 ---
	_draw_device_markers(ax, devices, bus_coords)

	# --- 标题和信息 ---
	v_mean = circuit.get("v_mean_pu", 1.0)
	loss_kw = circuit.get("total_loss_kw", 0.0)
	converged = circuit.get("converged", False)

	title = f"VVC Topology - Step {step}"
	info = f"V_mean={v_mean:.4f}pu | Loss={loss_kw:.1f}kW | {'Converged' if converged else 'Not Converged'}"

	ax.set_title(title, fontsize=12, color=COLORS["primary"])
	ax.text(
		0.5, -0.05, info,
		transform=ax.transAxes,
		fontsize=9,
		color=COLORS["text_secondary"],
		ha="center",
	)

	ax.set_aspect("equal")
	ax.axis("off")
	fig.tight_layout()

	return fig


def _draw_device_markers(
	ax: plt.Axes,
	devices: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""在坐标图上绘制设备标记。"""
	# 电容器
	caps = devices.get("capacitors", {})
	for name, cd in caps.items():
		bus = cd.get("bus", "")
		if bus not in bus_coords:
			continue
		x, y = bus_coords[bus]
		is_on = cd.get("is_on", False)
		color = DEVICE_COLORS["capacitor_on"] if is_on else DEVICE_COLORS["capacitor_off"]
		ax.scatter(
			x + 0.2, y + 0.2,
			c=color, s=60, marker="s", zorder=4,
			edgecolors="white", linewidths=0.5,
		)

	# 调压器
	regs = devices.get("regulators", {})
	for name, rd in regs.items():
		bus = rd.get("bus", "")
		if bus not in bus_coords:
			continue
		x, y = bus_coords[bus]
		ax.scatter(
			x - 0.2, y + 0.2,
			c=DEVICE_COLORS["regulator_tap"], s=60, marker="^", zorder=4,
			edgecolors="white", linewidths=0.5,
		)

	# 电池
	bats = devices.get("batteries", {})
	for name, bd in bats.items():
		bus = bd.get("bus", "")
		if bus not in bus_coords:
			continue
		x, y = bus_coords[bus]
		kw = bd.get("kw", 0)
		color = DEVICE_COLORS["storage_discharge"] if kw > 0 else DEVICE_COLORS["storage_charge"]
		ax.scatter(
			x + 0.2, y - 0.2,
			c=color, s=60, marker="D", zorder=4,
			edgecolors="white", linewidths=0.5,
		)

	# PV
	pvs = devices.get("pvsystems", {})
	for name, pd in pvs.items():
		bus = pd.get("bus", "")
		if bus not in bus_coords:
			continue
		x, y = bus_coords[bus]
		ax.scatter(
			x - 0.2, y - 0.2,
			c=DEVICE_COLORS["pv"], s=60, marker="o", zorder=4,
			edgecolors="white", linewidths=0.5,
		)


def _auto_layout(
	buses: Dict[str, Any],
) -> Dict[str, Tuple[float, float]]:
	"""简易自动布局 (母线沿圆环排列)。"""
	bus_names = sorted(buses.keys())
	n = len(bus_names)
	coords: Dict[str, Tuple[float, float]] = {}
	for i, name in enumerate(bus_names):
		angle = 2 * np.pi * i / max(n, 1)
		coords[name] = (5.0 + 4.0 * np.cos(angle), 5.0 + 4.0 * np.sin(angle))
	return coords
