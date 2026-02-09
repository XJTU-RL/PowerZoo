# -*- coding: utf-8 -*-
"""
VVC Episode 动画导出

使用 animation_exporter 将快照序列导出为 GIF/MP4，
帧渲染器调用 topology_animator.render_topology_frame。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from envs.render_common.export.animation_exporter import export_episode_animation
from envs.vvc.render.viz.mpl.topology_animator import render_topology_frame

logger = logging.getLogger(__name__)


def export_vvc_animation(
	snapshots: List[Dict[str, Any]],
	output_path: str,
	bus_coords: Optional[Dict[str, Tuple[float, float]]] = None,
	fps: int = 2,
	dpi: int = 150,
	format: str = "gif",
) -> str:
	"""将 VVC episode 导出为动画文件。

	Args:
		snapshots: 快照列表
		output_path: 输出文件路径
		bus_coords: 母线坐标
		fps: 帧率
		dpi: 分辨率
		format: "gif" 或 "mp4"

	Returns:
		输出文件路径
	"""
	if not snapshots:
		logger.warning("No snapshots to animate")
		return ""

	def frame_renderer(snapshot: Dict[str, Any], step: int):
		"""帧渲染函数"""
		return render_topology_frame(
			snapshot=snapshot,
			step=snapshot.get("step", step),
			bus_coords=bus_coords,
		)

	return export_episode_animation(
		snapshots=snapshots,
		frame_renderer=frame_renderer,
		output_path=output_path,
		fps=fps,
		dpi=dpi,
		format=format,
	)
