"""
SmartGrid Episode Animation (Matplotlib)
Episode 动画导出

使用 Matplotlib animation 系统生成 GIF/MP4，
调用 topology_animator 逐帧渲染。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from envs.render_common.export.animation_exporter import export_episode_animation
from envs.smartgrid.render.viz.mpl.topology_animator import render_topology_frame

logger = logging.getLogger(__name__)


def export_smartgrid_animation(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
	fps: int = 4,
	dpi: int = 150,
	format: str = "gif",
	v_min: float = 0.95,
	v_max: float = 1.05,
	max_frames: int = 120,
) -> str:
	"""导出 SmartGrid episode 动画

	对于 360 步 episode，会降采样到 max_frames 帧。

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标
		output_path: 输出文件路径
		fps: 帧率
		dpi: 分辨率
		format: 输出格式 ("gif" 或 "mp4")
		v_min: 电压下限
		v_max: 电压上限
		max_frames: 最大帧数

	Returns:
		输出文件路径
	"""
	# 降采样
	if len(snapshots) > max_frames:
		import numpy as np
		indices = np.linspace(0, len(snapshots) - 1, max_frames, dtype=int)
		indices = np.unique(indices)
		selected = [snapshots[i] for i in indices]
		logger.info(f"Animation downsampled: {len(snapshots)} -> {len(selected)} frames")
	else:
		selected = snapshots

	def frame_renderer(snapshot: Dict[str, Any], step: int):
		return render_topology_frame(
			snapshot=snapshot,
			bus_coords=bus_coords,
			v_min=v_min,
			v_max=v_max,
		)

	return export_episode_animation(
		snapshots=selected,
		frame_renderer=frame_renderer,
		output_path=output_path,
		fps=fps,
		dpi=dpi,
		format=format,
	)
