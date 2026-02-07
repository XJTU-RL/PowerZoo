"""
SmartGrid Topology Animator (Matplotlib)
拓扑图动画渲染器

为 GIF/MP4 动画导出提供逐帧拓扑图渲染函数。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from envs.render_common.viz.theme import apply_matplotlib_theme, COLORS, DEVICE_COLORS

logger = logging.getLogger(__name__)


def render_topology_frame(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
	v_min: float = 0.95,
	v_max: float = 1.05,
	figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
	"""渲染单帧拓扑图

	Args:
		snapshot: 快照字典
		bus_coords: 母线坐标
		v_min: 电压下限
		v_max: 电压上限
		figsize: 图表尺寸

	Returns:
		Matplotlib Figure
	"""
	apply_matplotlib_theme()

	fig, ax = plt.subplots(figsize=figsize)
	step = snapshot.get("step", 0)
	buses = snapshot.get("buses", {})
	lines = snapshot.get("lines", {})
	devices = snapshot.get("devices", {})

	# 绘制边
	for name, info in lines.items():
		bus1 = info.get("bus1", "")
		bus2 = info.get("bus2", "")
		if bus1 not in bus_coords or bus2 not in bus_coords:
			continue
		x0, y0 = bus_coords[bus1]
		x1, y1 = bus_coords[bus2]
		edge_type = info.get("type", "line")
		color = "#EF4444" if edge_type == "transformer" else "#6B7280"
		lw = 2.0 if edge_type == "transformer" else 1.0
		ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=1)

	# 绘制节点
	for bus_name, coord in bus_coords.items():
		bus_info = buses.get(bus_name, {})
		v_pu = bus_info.get("v_mag_pu", [])
		v_mean = float(np.mean(v_pu)) if v_pu else 1.0

		# 颜色映射
		normalized = (v_mean - (v_min - 0.05)) / ((v_max + 0.05) - (v_min - 0.05))
		normalized = max(0.0, min(1.0, normalized))

		if v_mean < v_min or v_mean > v_max:
			color = "#EF4444"
		elif abs(v_mean - 1.0) < 0.02:
			color = "#10B981"
		else:
			color = "#F59E0B"

		ax.scatter(
			coord[0], coord[1],
			c=color, s=40, zorder=3,
			edgecolors="white", linewidths=0.5,
		)

	# 绘制设备标记
	_draw_device_markers(ax, devices, bus_coords)

	ax.set_title(f"SmartGrid Topology - Day {step + 1}", fontsize=14, color="#e0e0e0")
	ax.set_aspect("equal")
	ax.axis("off")

	# 图例
	legend_elements = [
		mpatches.Patch(color="#10B981", label="Normal Voltage"),
		mpatches.Patch(color="#F59E0B", label="Near Limit"),
		mpatches.Patch(color="#EF4444", label="Violation"),
	]
	ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

	fig.tight_layout()
	return fig


def _draw_device_markers(
	ax: plt.Axes,
	devices: Dict[str, Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""在坐标轴上绘制设备标记"""
	marker_map = {
		"capacitors": ("D", DEVICE_COLORS["capacitor_on"], 60),
		"regulators": ("*", DEVICE_COLORS["regulator_tap"], 80),
		"batteries": ("s", DEVICE_COLORS["battery_soc"], 60),
		"pvs": ("^", DEVICE_COLORS["pv"], 60),
	}

	for dev_key, (marker, color, size) in marker_map.items():
		dev_dict = devices.get(dev_key, {})
		for name, dev in dev_dict.items():
			bus = dev.get("bus", dev.get("bus1", ""))
			if bus not in bus_coords:
				continue
			bx, by = bus_coords[bus]
			ax.scatter(
				bx, by, marker=marker, c=color,
				s=size, zorder=4, edgecolors="white", linewidths=0.5,
			)
