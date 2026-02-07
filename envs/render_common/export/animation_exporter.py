"""
Animation Exporter (Common)
动画导出器 -- 将 episode 回放导出为 GIF/MP4

使用 Matplotlib 动画系统逐帧渲染拓扑图 + 电压热图，
导出为 GIF (Pillow writer) 或 MP4 (ffmpeg writer)。
从 District Dispatch 提取，完全可复用。
"""

import io
import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def export_episode_animation(
	snapshots: List[Dict[str, Any]],
	frame_renderer: Callable[[Dict[str, Any], int], Any],
	output_path: str,
	fps: int = 2,
	dpi: int = 150,
	format: str = "gif",
) -> str:
	"""将 episode 快照序列导出为动画

	Args:
		snapshots: 快照列表
		frame_renderer: 帧渲染函数 fn(snapshot, step) -> matplotlib.Figure
		output_path: 输出文件路径
		fps: 帧率
		dpi: 分辨率
		format: 输出格式 ("gif" 或 "mp4")

	Returns:
		输出文件路径
	"""
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation, PillowWriter

	if not snapshots:
		logger.warning("No snapshots to animate")
		return ""

	# 渲染第一帧确定尺寸
	fig = frame_renderer(snapshots[0], 0)

	def update(frame_idx: int):
		"""动画更新函数"""
		fig.clear()
		frame_renderer(snapshots[frame_idx], frame_idx)

	anim = FuncAnimation(
		fig,
		update,
		frames=len(snapshots),
		interval=1000 // fps,
		repeat=False,
	)

	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

	if format == "gif":
		writer = PillowWriter(fps=fps)
		anim.save(output_path, writer=writer, dpi=dpi)
	elif format == "mp4":
		try:
			from matplotlib.animation import FFMpegWriter
			writer = FFMpegWriter(fps=fps, codec="libx264")
			anim.save(output_path, writer=writer, dpi=dpi)
		except Exception as exc:
			logger.warning(f"MP4 export failed (ffmpeg required): {exc}")
			# fallback to GIF
			gif_path = output_path.replace(".mp4", ".gif")
			writer = PillowWriter(fps=fps)
			anim.save(gif_path, writer=writer, dpi=dpi)
			output_path = gif_path
	else:
		raise ValueError(f"Unsupported format: {format}")

	plt.close(fig)

	logger.info(
		f"Animation exported: {output_path} "
		f"({len(snapshots)} frames, {fps} fps, {format})"
	)
	return output_path


def export_frames_as_images(
	snapshots: List[Dict[str, Any]],
	frame_renderer: Callable[[Dict[str, Any], int], Any],
	output_dir: str,
	dpi: int = 150,
	format: str = "png",
) -> List[str]:
	"""将 episode 快照序列导出为单帧图片

	Args:
		snapshots: 快照列表
		frame_renderer: 帧渲染函数
		output_dir: 输出目录
		dpi: 分辨率
		format: 图片格式

	Returns:
		输出文件路径列表
	"""
	import matplotlib.pyplot as plt

	os.makedirs(output_dir, exist_ok=True)
	paths: List[str] = []

	for idx, snap in enumerate(snapshots):
		fig = frame_renderer(snap, idx)
		filename = f"frame_{idx:04d}.{format}"
		filepath = os.path.join(output_dir, filename)
		fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
		plt.close(fig)
		paths.append(filepath)

	logger.info(f"Exported {len(paths)} frames to {output_dir}")
	return paths


def render_frame_to_bytes(
	frame_renderer: Callable[[Dict[str, Any], int], Any],
	snapshot: Dict[str, Any],
	step: int,
	dpi: int = 100,
	format: str = "png",
) -> bytes:
	"""将单帧渲染为 bytes (用于 Gradio 实时显示)

	Args:
		frame_renderer: 帧渲染函数
		snapshot: 快照字典
		step: 步编号
		dpi: 分辨率
		format: 图片格式

	Returns:
		图片 bytes
	"""
	import matplotlib.pyplot as plt

	fig = frame_renderer(snapshot, step)
	buf = io.BytesIO()
	fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	buf.seek(0)
	return buf.getvalue()
