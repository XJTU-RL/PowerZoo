"""
DSR Episode Animation
Episode 动画导出器

使用 topology_animator 的帧渲染函数，配合
render_common.export.animation_exporter 导出 GIF/MP4。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def export_episode_gif(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
	fps: int = 2,
	dpi: int = 150,
) -> str:
	"""将 episode 导出为 GIF 动画

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标字典
		output_path: 输出路径
		fps: 帧率 (DSR 默认 2fps，因步数少)
		dpi: 分辨率

	Returns:
		输出文件路径
	"""
	from envs.render_common.export.animation_exporter import export_episode_animation
	from envs.dsr.render.viz.mpl.topology_animator import render_topology_frame

	def frame_renderer(snapshot: Dict[str, Any], step: int):
		"""帧渲染回调"""
		return render_topology_frame(
			snapshot, step, bus_coords,
			figsize=(10, 8),
		)

	return export_episode_animation(
		snapshots=snapshots,
		frame_renderer=frame_renderer,
		output_path=output_path,
		fps=fps,
		dpi=dpi,
		format="gif",
	)


def export_episode_mp4(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
	fps: int = 2,
	dpi: int = 150,
) -> str:
	"""将 episode 导出为 MP4 视频

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标字典
		output_path: 输出路径
		fps: 帧率
		dpi: 分辨率

	Returns:
		输出文件路径
	"""
	from envs.render_common.export.animation_exporter import export_episode_animation
	from envs.dsr.render.viz.mpl.topology_animator import render_topology_frame

	def frame_renderer(snapshot: Dict[str, Any], step: int):
		"""帧渲染回调"""
		return render_topology_frame(
			snapshot, step, bus_coords,
			figsize=(10, 8),
		)

	return export_episode_animation(
		snapshots=snapshots,
		frame_renderer=frame_renderer,
		output_path=output_path,
		fps=fps,
		dpi=dpi,
		format="mp4",
	)
