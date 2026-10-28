"""
DSR Topology Animator
拓扑动画器 -- 含故障高亮动画

使用 Matplotlib 绘制拓扑图帧，支持导出为 GIF/MP4。
每帧展示当前时间步的网络状态:
- 带电母线: 绿色/电压着色
- 断电母线: 灰色
- 故障线路: 红色虚线 + 闪烁效果
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def render_topology_frame(
	snapshot: Dict[str, Any],
	step: int,
	bus_coords: Dict[str, Tuple[float, float]],
	figsize: Tuple[float, float] = (10, 8),
) -> Any:
	"""渲染单帧拓扑图

	Args:
		snapshot: 快照字典
		step: 当前步编号
		bus_coords: 母线坐标字典
		figsize: 图片尺寸

	Returns:
		matplotlib.Figure 对象
	"""
	import matplotlib.pyplot as plt
	from envs.render_common.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	fig, ax = plt.subplots(1, 1, figsize=figsize)

	buses = snapshot.get("buses", {})
	lines = snapshot.get("lines", {})
	restoration = snapshot.get("restoration_data", {})
	energized_set = set(restoration.get("energized_buses", []))
	fault_lines = set(restoration.get("fault_lines", []))

	# 绘制线路
	for line_name, line_data in lines.items():
		from_bus = line_data.get("from_bus", "")
		to_bus = line_data.get("to_bus", "")
		if from_bus not in bus_coords or to_bus not in bus_coords:
			continue

		x0, y0 = bus_coords[from_bus]
		x1, y1 = bus_coords[to_bus]

		is_faulted = line_data.get("is_faulted", False) or line_name in fault_lines
		is_open = line_data.get("is_open", False)

		if is_faulted:
			ax.plot(
				[x0, x1], [y0, y1],
				color="#DC2626", linewidth=2.5, linestyle="--",
				alpha=0.8 + 0.2 * np.sin(step * np.pi),  # 闪烁
				zorder=2,
			)
			mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
			ax.text(
				mid_x, mid_y, "FAULT",
				fontsize=7, color="#DC2626", fontweight="bold",
				ha="center", va="center",
				bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
				zorder=5,
			)
		elif is_open:
			ax.plot(
				[x0, x1], [y0, y1],
				color="#EF4444", linewidth=1.5, linestyle=":",
				alpha=0.6, zorder=1,
			)
		else:
			ax.plot(
				[x0, x1], [y0, y1],
				color="white", linewidth=1.0, alpha=0.3, zorder=1,
			)

	# 绘制母线
	for bus_name, bus_data in buses.items():
		if bus_name not in bus_coords:
			continue
		x, y = bus_coords[bus_name]
		is_energized = bus_data.get("is_energized", bus_name in energized_set)

		if is_energized:
			color = "#10B981"
			marker = "o"
			size = 60
		else:
			color = "#6B7280"
			marker = "x"
			size = 40

		ax.scatter(x, y, c=color, marker=marker, s=size, zorder=3, edgecolors="white", linewidths=0.5)
		ax.text(
			x, y + 12, bus_name,
			fontsize=6, color="#a0a0a0", ha="center", va="bottom",
			zorder=4,
		)

	# 信息标注
	rest_pct = restoration.get("restoration_pct", 0.0)
	n_faults_count = len(fault_lines)
	info_text = f"Step: {step} | Restored: {rest_pct:.1f}% | Faults: {n_faults_count}"
	ax.set_title(info_text, fontsize=12, color="#e0e0e0", pad=10)

	ax.set_aspect("equal")
	ax.set_xticks([])
	ax.set_yticks([])

	fig.tight_layout()

	return fig


def create_topology_animation_frames(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
) -> List[Any]:
	"""创建拓扑动画帧序列

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标字典

	Returns:
		matplotlib.Figure 列表
	"""
	import matplotlib.pyplot as plt

	frames = []
	for snap in snapshots:
		step = snap.get("step", 0)
		fig = render_topology_frame(snap, step, bus_coords)
		frames.append(fig)

	return frames
