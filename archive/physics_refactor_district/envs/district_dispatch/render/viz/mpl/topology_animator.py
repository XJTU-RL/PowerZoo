# -*- coding: utf-8 -*-
"""
拓扑图帧渲染器

为 GIF/MP4 动画生成 IEEE 34-bus 网络拓扑图帧。
节点按电压着色，线路按负载率着色，Zone 背景区域高亮，
设备标注 PV/Storage/EV 文本。标题栏显示 step, time, total_reward。
所有图表文本使用英文标注。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from envs.district_dispatch.render.assets.bus_coordinates import (
	BUS_TO_ZONE,
	ZONE_BUSES,
)
from envs.district_dispatch.render.utils.color_scales import (
	loading_to_color,
	voltage_to_color,
)
from envs.district_dispatch.render.viz.theme import (
	COLORS,
	DEVICE_COLORS,
	ZONE_COLORS,
	ZONE_NAMES,
	apply_matplotlib_theme,
	get_matplotlib_rcparams,
)

logger = logging.getLogger(__name__)

# 默认线路拓扑连接 (from_bus, to_bus)
# IEEE 34-bus 干线 + 主要分支
_DEFAULT_EDGES: List[Tuple[str, str]] = [
	("800", "802"), ("802", "806"), ("806", "808"), ("808", "810"),
	("810", "812"), ("812", "814"), ("814", "850"), ("850", "816"),
	("816", "818"), ("818", "820"), ("820", "822"), ("816", "824"),
	("824", "826"), ("824", "828"), ("828", "830"), ("830", "854"),
	("854", "856"), ("854", "852"), ("852", "832"), ("832", "858"),
	("858", "864"), ("858", "834"), ("834", "842"), ("842", "844"),
	("844", "846"), ("846", "848"), ("834", "860"), ("860", "836"),
	("836", "840"), ("836", "862"), ("862", "838"), ("888", "890"),
	("832", "888"),
]


def _extract_bus_voltage(
	snapshot: Dict[str, Any], bus_name: str
) -> float:
	"""从快照中提取母线平均电压标幺值。

	优先使用 buses dict 中的 v_mean，
	其次使用 bus_data dict 中的 v_mean，
	均不存在则返回 1.0。

	Args:
		snapshot: 单步快照数据
		bus_name: 母线名称

	Returns:
		float: 电压标幺值
	"""
	# snapshot_assembler 格式
	buses = snapshot.get("buses", {})
	if bus_name in buses:
		return buses[bus_name].get("v_mean", 1.0)

	# episode_runner 格式
	bus_data = snapshot.get("bus_data", {})
	if bus_name in bus_data:
		return bus_data[bus_name].get("v_mean", 1.0)

	return 1.0


def _extract_line_loading(
	snapshot: Dict[str, Any], from_bus: str, to_bus: str
) -> float:
	"""从快照中提取线路负载率。

	按 from_bus-to_bus 或 to_bus-from_bus 命名搜索。

	Args:
		snapshot: 单步快照数据
		from_bus: 送端母线
		to_bus: 受端母线

	Returns:
		float: 负载率百分比 (0-100+)
	"""
	lines = snapshot.get("lines", {})
	if not lines:
		return 0.0

	for name, data in lines.items():
		fb = data.get("from_bus", "").replace(".", "").lower()
		tb = data.get("to_bus", "").replace(".", "").lower()
		f_low = from_bus.lower()
		t_low = to_bus.lower()

		if (fb == f_low and tb == t_low) or (fb == t_low and tb == f_low):
			return data.get("loading_pct", 0.0)

	return 0.0


def _get_device_labels(
	snapshot: Dict[str, Any], bus_name: str
) -> List[Tuple[str, str]]:
	"""获取某母线上的设备标签列表。

	返回 [(设备类型简称, 颜色), ...] 用于标注。

	Args:
		snapshot: 单步快照数据
		bus_name: 母线名称

	Returns:
		List[Tuple[str, str]]: [(label, hex_color), ...]
	"""
	devices = snapshot.get("devices", {})
	labels: List[Tuple[str, str]] = []

	bus_lower = bus_name.lower()

	# PV
	for pv_name, pv_data in devices.get("pv", {}).items():
		pv_bus = pv_data.get("bus", "").split(".")[0].lower()
		if pv_bus == bus_lower:
			kw = pv_data.get("kw_output", 0)
			labels.append((f"PV:{kw:.0f}kW", DEVICE_COLORS["pv"]))

	# Storage
	for st_name, st_data in devices.get("storage", {}).items():
		st_bus = st_data.get("bus", "").split(".")[0].lower()
		if st_bus == bus_lower:
			soc = st_data.get("soc", 0)
			kw = st_data.get("kw_output", st_data.get("kw", 0))
			if kw >= 0:
				labels.append((f"ESS:{soc:.0%}", DEVICE_COLORS["storage_discharge"]))
			else:
				labels.append((f"ESS:{soc:.0%}", DEVICE_COLORS["storage_charge"]))

	# EV
	for ev_name, ev_data in devices.get("ev", {}).items():
		ev_bus = ev_data.get("bus", "").split(".")[0].lower()
		if ev_bus == bus_lower:
			kw = ev_data.get("kw", ev_data.get("kw_demand", 0))
			labels.append((f"EV:{kw:.0f}kW", DEVICE_COLORS["ev"]))

	return labels


def create_topology_canvas(
	bus_coords: Dict[str, Tuple[float, float]],
	figsize: Tuple[float, float] = (14, 9),
	dpi: int = 150,
) -> Tuple[Figure, plt.Axes]:
	"""创建拓扑图画布。

	生成带暗色背景的 Matplotlib 画布，预设好坐标范围和边距。
	画布创建后可多次调用 render_frame 复用。

	Args:
		bus_coords: {母线名: (x, y)} 归一化坐标字典
		figsize: 图形尺寸 (英寸)
		dpi: 分辨率

	Returns:
		Tuple[Figure, Axes]: matplotlib 图形和坐标轴
	"""
	apply_matplotlib_theme()

	fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, 1.05)
	ax.set_aspect("equal")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

	fig.tight_layout(rect=[0, 0, 1, 0.93])
	return fig, ax


def _draw_zone_backgrounds(
	ax: plt.Axes,
	bus_coords: Dict[str, Tuple[float, float]],
	alpha: float = 0.08,
) -> None:
	"""绘制 Zone 背景区域。

	为每个 Zone 中的母线绘制一个半透明矩形背景。

	Args:
		ax: matplotlib 坐标轴
		bus_coords: 母线坐标
		alpha: 透明度
	"""
	for zone_id, bus_list in ZONE_BUSES.items():
		xs, ys = [], []
		for b in bus_list:
			if b in bus_coords:
				x, y = bus_coords[b]
				xs.append(x)
				ys.append(y)

		if not xs:
			continue

		pad = 0.03
		x_min, x_max = min(xs) - pad, max(xs) + pad
		y_min, y_max = min(ys) - pad, max(ys) + pad

		color = ZONE_COLORS[zone_id] if zone_id < len(ZONE_COLORS) else "#888888"
		rect = mpatches.FancyBboxPatch(
			(x_min, y_min),
			x_max - x_min,
			y_max - y_min,
			boxstyle="round,pad=0.01",
			facecolor=color,
			edgecolor=color,
			alpha=alpha,
			linewidth=1.0,
			linestyle="--",
		)
		ax.add_patch(rect)

		# Zone 标签
		label = ZONE_NAMES[zone_id] if zone_id < len(ZONE_NAMES) else f"Zone {zone_id}"
		ax.text(
			(x_min + x_max) / 2, y_max + 0.01,
			label,
			ha="center", va="bottom",
			fontsize=8, color=color, alpha=0.7,
			fontweight="bold",
		)


def render_frame(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
	fig: Figure,
	ax: plt.Axes,
	edges: Optional[List[Tuple[str, str]]] = None,
	node_size: float = 120.0,
	show_labels: bool = True,
	show_devices: bool = True,
) -> Figure:
	"""渲染单帧拓扑图。

	在给定画布上绘制 IEEE 34-bus 网络拓扑，
	节点按电压着色、线路按负载率着色。
	每次调用前会清空 ax 重新绘制。

	Args:
		snapshot: 单步快照数据
		bus_coords: {母线名: (x, y)} 归一化坐标字典
		fig: matplotlib Figure
		ax: matplotlib Axes
		edges: 线路连接列表 [(from, to), ...], None 则使用默认
		node_size: 节点散点大小
		show_labels: 是否显示母线名称
		show_devices: 是否显示设备标注

	Returns:
		Figure: 渲染后的 matplotlib Figure
	"""
	ax.clear()
	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(-0.05, 1.05)
	ax.set_aspect("equal")
	ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

	if edges is None:
		edges = _DEFAULT_EDGES

	# 1. Zone 背景
	_draw_zone_backgrounds(ax, bus_coords)

	# 2. 线路 (按负载率着色)
	for from_bus, to_bus in edges:
		if from_bus not in bus_coords or to_bus not in bus_coords:
			continue

		x1, y1 = bus_coords[from_bus]
		x2, y2 = bus_coords[to_bus]
		loading = _extract_line_loading(snapshot, from_bus, to_bus)
		color = loading_to_color(loading)

		ax.plot(
			[x1, x2], [y1, y2],
			color=color, linewidth=1.5, alpha=0.8, zorder=1,
		)

	# 3. 母线节点 (按电压着色)
	for bus_name, (x, y) in bus_coords.items():
		v_pu = _extract_bus_voltage(snapshot, bus_name)
		color = voltage_to_color(v_pu)

		ax.scatter(
			x, y, s=node_size, c=color, edgecolors="white",
			linewidths=0.5, zorder=3, alpha=0.9,
		)

		# 母线名称标签
		if show_labels:
			ax.text(
				x, y - 0.025, bus_name,
				ha="center", va="top", fontsize=6,
				color=COLORS["text_secondary"], zorder=4,
			)

		# 设备标注
		if show_devices:
			dev_labels = _get_device_labels(snapshot, bus_name)
			for i, (label, dev_color) in enumerate(dev_labels):
				offset_y = 0.02 + i * 0.025
				ax.text(
					x + 0.02, y + offset_y, label,
					ha="left", va="bottom", fontsize=5.5,
					color=dev_color, zorder=4,
					bbox=dict(
						facecolor=COLORS["bg_dark"], edgecolor=dev_color,
						alpha=0.7, boxstyle="round,pad=0.15", linewidth=0.5,
					),
				)

	# 4. 标题栏
	step = snapshot.get("step", 0)
	timestamp_h = snapshot.get("timestamp_h", step * 0.25)
	time_str = f"{int(timestamp_h)}:{int((timestamp_h % 1) * 60):02d}"
	total_reward = snapshot.get("cumulative_reward", snapshot.get("total_reward", 0.0))

	title = (
		f"Step {step}  |  Time {time_str}h  |  "
		f"Cumulative Reward: {total_reward:.2f}"
	)
	fig.suptitle(title, fontsize=12, color=COLORS["text_primary"], fontweight="bold")

	# 5. 图例
	legend_elements = [
		mpatches.Patch(facecolor=voltage_to_color(1.0), label="V ~ 1.0 pu (Normal)"),
		mpatches.Patch(facecolor=voltage_to_color(0.93), label="V ~ 0.93 pu (Low)"),
		mpatches.Patch(facecolor=voltage_to_color(1.07), label="V ~ 1.07 pu (High)"),
	]
	ax.legend(
		handles=legend_elements, loc="lower right",
		fontsize=7, framealpha=0.6,
	)

	return fig
