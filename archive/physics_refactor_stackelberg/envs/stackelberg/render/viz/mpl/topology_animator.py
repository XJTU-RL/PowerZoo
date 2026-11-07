# -*- coding: utf-8 -*-
"""
拓扑动画器

使用 Matplotlib FuncAnimation 生成电路拓扑随时间变化的动画，
节点颜色按电压变化，线路宽度按负载率变化。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from envs.render_common.utils.color_scales import voltage_to_color
from envs.stackelberg.render.assets.bus_coordinates import (
	load_bus_coordinates,
	load_topology_edges,
)

logger = logging.getLogger(__name__)


def create_topology_animation(
	snapshots: List[Dict[str, Any]],
	system_name: str = "13Bus",
	bus_coords: Optional[Dict[str, Tuple[float, float]]] = None,
	interval_ms: int = 500,
	output_path: Optional[str] = None,
) -> Optional[FuncAnimation]:
	"""创建拓扑动画

	Args:
		snapshots: 快照列表
		system_name: IEEE 系统名称
		bus_coords: 母线坐标
		interval_ms: 帧间隔 (毫秒)
		output_path: 输出文件路径 (.gif 或 .mp4)

	Returns:
		FuncAnimation 对象，或 None (如果保存到文件)
	"""
	if bus_coords is None:
		bus_coords = load_bus_coordinates(system_name)
	edges = load_topology_edges(system_name)

	fig, ax = plt.subplots(figsize=(12, 8), facecolor="#1a1a2e")
	ax.set_facecolor("#0f0f23")
	ax.set_aspect("equal")
	ax.set_title(
		f"{system_name} Topology Animation",
		color="#F59E0B", fontsize=14, fontweight="bold",
	)
	ax.axis("off")

	# 静态边
	for from_bus, to_bus in edges:
		if from_bus in bus_coords and to_bus in bus_coords:
			x0, y0 = bus_coords[from_bus]
			x1, y1 = bus_coords[to_bus]
			ax.plot(
				[x0, x1], [y0, y1],
				color="#3a3a5e", linewidth=1, zorder=1,
			)

	# 动态节点
	xs = [bus_coords[b][0] for b in bus_coords]
	ys = [bus_coords[b][1] for b in bus_coords]
	bus_names = list(bus_coords.keys())

	scatter = ax.scatter(
		xs, ys, s=80, c=["#10B981"] * len(xs),
		edgecolors="white", linewidth=0.5, zorder=3,
	)

	# 母线标签
	for name, (x, y) in bus_coords.items():
		ax.annotate(
			name, (x, y), textcoords="offset points",
			xytext=(0, 8), ha="center", fontsize=6, color="#a0a0a0",
		)

	step_text = ax.text(
		0.02, 0.98, "", transform=ax.transAxes,
		fontsize=12, color="#e0e0e0", verticalalignment="top",
	)

	def update(frame: int) -> list:
		snap = snapshots[frame]
		buses_data = snap.get("buses", {})
		step = snap.get("step", frame)

		colors = []
		for name in bus_names:
			bus_data = buses_data.get(name, {})
			v_pu = bus_data.get("v_mag_pu", [1.0])
			if isinstance(v_pu, list) and v_pu:
				avg_v = sum(v_pu) / len(v_pu)
			else:
				avg_v = float(v_pu) if isinstance(v_pu, (int, float)) else 1.0
			colors.append(voltage_to_color(avg_v))

		scatter.set_facecolors(colors)
		step_text.set_text(f"Step {step} / Hour {step % 24}")
		return [scatter, step_text]

	anim = FuncAnimation(
		fig, update, frames=len(snapshots),
		interval=interval_ms, blit=True,
	)

	if output_path:
		try:
			if output_path.endswith(".gif"):
				anim.save(output_path, writer="pillow", fps=1000 // interval_ms)
			else:
				anim.save(output_path, writer="ffmpeg", fps=1000 // interval_ms)
			logger.info(f"Animation saved to {output_path}")
			plt.close(fig)
			return None
		except Exception as exc:
			logger.warning(f"Failed to save animation: {exc}")

	return anim
