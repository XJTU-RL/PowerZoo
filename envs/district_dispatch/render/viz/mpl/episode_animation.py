# -*- coding: utf-8 -*-
"""
Episode 动画合成器

将一个 episode 的快照序列合成为 GIF 或 MP4 动画。
多面板布局：上方拓扑图 + 下方三个小图（电压/功率/SOC）。
使用 matplotlib.animation.FuncAnimation 驱动帧更新。
所有图表文本使用英文标注。
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from envs.district_dispatch.render.assets.bus_coordinates import (
	BUS_TO_ZONE,
	ZONE_BUSES,
)
from envs.district_dispatch.render.utils.color_scales import (
	soc_to_color,
	voltage_to_color,
)
from envs.district_dispatch.render.viz.mpl.topology_animator import render_frame
from envs.district_dispatch.render.viz.theme import (
	COLORS,
	ZONE_COLORS,
	ZONE_NAMES,
	apply_matplotlib_theme,
)

logger = logging.getLogger(__name__)


def _extract_voltage_series(
	snapshots: List[Dict[str, Any]],
) -> Dict[int, List[float]]:
	"""提取各 Zone 的平均电压时间序列。

	按 Zone 分组计算母线平均电压，返回 {zone_id: [v_mean_per_step]}。

	Args:
		snapshots: 快照列表

	Returns:
		Dict[int, List[float]]: {zone_id: [voltage_values]}
	"""
	zone_voltages: Dict[int, List[float]] = {0: [], 1: [], 2: []}

	for snap in snapshots:
		buses = snap.get("buses", snap.get("bus_data", {}))
		for zone_id, bus_list in ZONE_BUSES.items():
			vs = []
			for b in bus_list:
				if b in buses:
					vs.append(buses[b].get("v_mean", 1.0))
			zone_voltages[zone_id].append(np.mean(vs) if vs else 1.0)

	return zone_voltages


def _extract_power_series(
	snapshots: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
	"""提取功率时间序列。

	从 circuit 层提取 total_load, total_loss, total_pv, total_storage。

	Args:
		snapshots: 快照列表

	Returns:
		Dict[str, List[float]]: {指标名: [值序列]}
	"""
	series: Dict[str, List[float]] = {
		"load_kw": [],
		"loss_kw": [],
		"pv_kw": [],
		"storage_kw": [],
	}

	for snap in snapshots:
		circuit = snap.get("circuit", {})
		series["load_kw"].append(circuit.get("total_load_kw", 0.0))
		series["loss_kw"].append(circuit.get("total_loss_kw", 0.0))

		devices = snap.get("devices", {})
		pv_total = sum(
			d.get("kw_output", 0.0) for d in devices.get("pv", {}).values()
		)
		storage_total = sum(
			d.get("kw_output", d.get("kw", 0.0))
			for d in devices.get("storage", {}).values()
		)
		series["pv_kw"].append(pv_total)
		series["storage_kw"].append(storage_total)

	return series


def _extract_soc_series(
	snapshots: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
	"""提取储能 SOC 时间序列。

	Args:
		snapshots: 快照列表

	Returns:
		Dict[str, List[float]]: {storage_name: [soc_values]}
	"""
	# 先收集所有储能名
	all_names: List[str] = []
	for snap in snapshots:
		devices = snap.get("devices", {})
		for name in devices.get("storage", {}).keys():
			if name not in all_names:
				all_names.append(name)

	soc_series: Dict[str, List[float]] = {name: [] for name in all_names}

	for snap in snapshots:
		storage_data = snap.get("devices", {}).get("storage", {})
		for name in all_names:
			soc = storage_data.get(name, {}).get("soc", 0.5)
			soc_series[name].append(soc)

	return soc_series


def create_episode_animation(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	fps: int = 4,
	dpi: int = 150,
	progress_callback: Optional[Callable[[int, int], None]] = None,
) -> FuncAnimation:
	"""创建 episode 动画。

	多面板布局：上方大图为拓扑，下方三个小图分别为
	电压轨迹、功率曲线、SOC 曲线。

	Args:
		snapshots: 快照列表 (episode 全部步)
		bus_coords: {母线名: (x, y)} 归一化坐标
		fps: 帧率
		dpi: 分辨率
		progress_callback: 进度回调 fn(current_frame, total_frames)

	Returns:
		FuncAnimation: matplotlib 动画对象
	"""
	apply_matplotlib_theme()

	n_frames = len(snapshots)
	if n_frames == 0:
		raise ValueError("snapshots 为空，无法创建动画")

	# 预计算时间序列数据
	voltage_series = _extract_voltage_series(snapshots)
	power_series = _extract_power_series(snapshots)
	soc_series = _extract_soc_series(snapshots)
	steps = list(range(n_frames))

	# 创建多面板布局
	fig = plt.figure(figsize=(16, 12), dpi=dpi)
	gs = fig.add_gridspec(
		2, 3, height_ratios=[2, 1],
		hspace=0.25, wspace=0.3,
		left=0.05, right=0.95, top=0.92, bottom=0.05,
	)

	ax_topo = fig.add_subplot(gs[0, :])
	ax_volt = fig.add_subplot(gs[1, 0])
	ax_power = fig.add_subplot(gs[1, 1])
	ax_soc = fig.add_subplot(gs[1, 2])

	def _update(frame_idx: int) -> None:
		"""更新单帧。"""
		snapshot = snapshots[frame_idx]

		# 拓扑图
		render_frame(snapshot, bus_coords, fig, ax_topo)

		# 电压轨迹 (到当前帧)
		ax_volt.clear()
		ax_volt.set_title("Zone Avg Voltage (pu)", fontsize=9)
		ax_volt.set_xlabel("Step", fontsize=8)
		ax_volt.set_ylabel("V (pu)", fontsize=8)
		ax_volt.set_xlim(0, max(n_frames - 1, 1))
		ax_volt.set_ylim(0.90, 1.10)
		ax_volt.axhline(y=0.95, color=COLORS["warning"], linestyle="--", alpha=0.5, linewidth=0.8)
		ax_volt.axhline(y=1.05, color=COLORS["warning"], linestyle="--", alpha=0.5, linewidth=0.8)

		for zone_id in range(3):
			color = ZONE_COLORS[zone_id] if zone_id < len(ZONE_COLORS) else "#888"
			label = ZONE_NAMES[zone_id] if zone_id < len(ZONE_NAMES) else f"Zone {zone_id}"
			ax_volt.plot(
				steps[:frame_idx + 1],
				voltage_series[zone_id][:frame_idx + 1],
				color=color, label=label, linewidth=1.2,
			)
		ax_volt.legend(fontsize=6, loc="lower left")
		ax_volt.tick_params(labelsize=7)

		# 功率曲线
		ax_power.clear()
		ax_power.set_title("Power (kW)", fontsize=9)
		ax_power.set_xlabel("Step", fontsize=8)
		ax_power.set_ylabel("kW", fontsize=8)
		ax_power.set_xlim(0, max(n_frames - 1, 1))

		power_colors = {
			"load_kw": COLORS["text_secondary"],
			"loss_kw": COLORS["danger"],
			"pv_kw": COLORS["warning"],
			"storage_kw": COLORS["info"],
		}
		power_labels = {
			"load_kw": "Load",
			"loss_kw": "Loss",
			"pv_kw": "PV",
			"storage_kw": "Storage",
		}
		for key, vals in power_series.items():
			ax_power.plot(
				steps[:frame_idx + 1],
				vals[:frame_idx + 1],
				color=power_colors.get(key, "#888"),
				label=power_labels.get(key, key),
				linewidth=1.2,
			)
		ax_power.legend(fontsize=6, loc="upper left")
		ax_power.tick_params(labelsize=7)

		# SOC 曲线
		ax_soc.clear()
		ax_soc.set_title("Storage SOC", fontsize=9)
		ax_soc.set_xlabel("Step", fontsize=8)
		ax_soc.set_ylabel("SOC", fontsize=8)
		ax_soc.set_xlim(0, max(n_frames - 1, 1))
		ax_soc.set_ylim(0, 1)
		ax_soc.axhline(y=0.2, color=COLORS["danger"], linestyle="--", alpha=0.4, linewidth=0.8)
		ax_soc.axhline(y=0.8, color=COLORS["danger"], linestyle="--", alpha=0.4, linewidth=0.8)

		soc_cmap = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#7C3AED"]
		for i, (name, vals) in enumerate(soc_series.items()):
			color = soc_cmap[i % len(soc_cmap)]
			ax_soc.plot(
				steps[:frame_idx + 1],
				vals[:frame_idx + 1],
				color=color, label=name, linewidth=1.2,
			)
		if soc_series:
			ax_soc.legend(fontsize=6, loc="lower left")
		ax_soc.tick_params(labelsize=7)

		# 进度回调
		if progress_callback is not None:
			progress_callback(frame_idx + 1, n_frames)

	anim = FuncAnimation(
		fig, _update, frames=n_frames,
		interval=1000 // fps, repeat=False,
	)

	return anim


def save_gif(
	animation: FuncAnimation,
	path: str,
	fps: int = 4,
	dpi: int = 150,
) -> str:
	"""将动画保存为 GIF 文件。

	Args:
		animation: FuncAnimation 对象
		path: 输出 GIF 文件路径
		fps: 帧率
		dpi: 分辨率

	Returns:
		str: 保存的文件路径
	"""
	logger.info(f"Saving GIF to {path} (fps={fps}, dpi={dpi})")
	animation.save(path, writer="pillow", fps=fps, dpi=dpi)
	logger.info(f"GIF saved: {path}")
	return path


def save_mp4(
	animation: FuncAnimation,
	path: str,
	fps: int = 4,
	dpi: int = 150,
) -> str:
	"""将动画保存为 MP4 文件。

	需要系统安装 ffmpeg。

	Args:
		animation: FuncAnimation 对象
		path: 输出 MP4 文件路径
		fps: 帧率
		dpi: 分辨率

	Returns:
		str: 保存的文件路径
	"""
	logger.info(f"Saving MP4 to {path} (fps={fps}, dpi={dpi})")
	animation.save(path, writer="ffmpeg", fps=fps, dpi=dpi)
	logger.info(f"MP4 saved: {path}")
	return path
