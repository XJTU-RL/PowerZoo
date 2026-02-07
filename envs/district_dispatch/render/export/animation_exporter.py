"""
Animation Exporter
动画导出器 -- 将 episode 快照序列导出为 GIF/MP4

包装 viz/mpl/ 模块的动画生成功能，提供 GIF 和 MP4 两种输出格式。
MP4 导出依赖系统安装的 ffmpeg。
"""

import io
import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
	"""检查系统是否安装了 ffmpeg

	Returns:
		True 如果 ffmpeg 可用，否则 False
	"""
	return shutil.which("ffmpeg") is not None


def export_gif(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
	fps: int = 4,
	dpi: int = 150,
	progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> str:
	"""将快照序列导出为 GIF 动画

	逐帧渲染后使用 Pillow 合成 GIF 动画。

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标字典 {bus_name: (x, y)}
		output_path: 输出 .gif 文件路径
		fps: 帧率 (1-10)，默认 4
		dpi: 分辨率 (72-300)，默认 150
		progress_cb: 进度回调函数 (current, total, status_msg)

	Returns:
		输出文件的绝对路径

	Raises:
		ImportError: matplotlib 或 Pillow 未安装
		ValueError: 快照列表为空
	"""
	if not snapshots:
		raise ValueError("snapshots list is empty")

	fps = max(1, min(10, fps))
	dpi = max(72, min(300, dpi))

	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from PIL import Image

	from envs.district_dispatch.render.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	total = len(snapshots)
	frames: List[Image.Image] = []
	tmpdir = tempfile.mkdtemp(prefix="powerzoo_gif_")

	try:
		for idx, snap in enumerate(snapshots):
			if progress_cb is not None:
				progress_cb(idx, total, f"Rendering frame {idx + 1}/{total}")

			fig = _render_frame(snap, bus_coords, dpi)
			frame_path = os.path.join(tmpdir, f"frame_{idx:04d}.png")
			fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
			plt.close(fig)

			img = Image.open(frame_path).convert("RGBA")
			frames.append(img)

		if progress_cb is not None:
			progress_cb(total, total, "Compositing GIF...")

		# 合成 GIF
		duration_ms = int(1000 / fps)
		os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
		frames[0].save(
			output_path,
			save_all=True,
			append_images=frames[1:],
			duration=duration_ms,
			loop=0,
			optimize=True,
		)

		if progress_cb is not None:
			progress_cb(total, total, "GIF export complete")

	finally:
		# 清理临时文件
		shutil.rmtree(tmpdir, ignore_errors=True)

	return os.path.abspath(output_path)


def export_mp4(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
	fps: int = 4,
	dpi: int = 150,
	progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> str:
	"""将快照序列导出为 MP4 视频

	使用 matplotlib 的 FFMpegWriter 后端生成 MP4 视频。

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标字典 {bus_name: (x, y)}
		output_path: 输出 .mp4 文件路径
		fps: 帧率 (1-10)，默认 4
		dpi: 分辨率 (72-300)，默认 150
		progress_cb: 进度回调函数 (current, total, status_msg)

	Returns:
		输出文件的绝对路径

	Raises:
		RuntimeError: ffmpeg 未安装
		ValueError: 快照列表为空
	"""
	if not snapshots:
		raise ValueError("snapshots list is empty")

	if not check_ffmpeg():
		raise RuntimeError(
			"ffmpeg is not installed. "
			"Please install ffmpeg to export MP4 videos."
		)

	fps = max(1, min(10, fps))
	dpi = max(72, min(300, dpi))

	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.animation import FFMpegWriter

	from envs.district_dispatch.render.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	total = len(snapshots)
	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

	# 创建 writer
	writer = FFMpegWriter(fps=fps, metadata={"title": "PowerZoo Episode"})

	# 用首帧初始化 figure
	fig = _render_frame(snapshots[0], bus_coords, dpi)

	with writer.saving(fig, output_path, dpi=dpi):
		for idx, snap in enumerate(snapshots):
			if progress_cb is not None:
				progress_cb(idx, total, f"Encoding frame {idx + 1}/{total}")

			fig.clear()
			_draw_on_figure(fig, snap, bus_coords)
			writer.grab_frame()

	plt.close(fig)

	if progress_cb is not None:
		progress_cb(total, total, "MP4 export complete")

	return os.path.abspath(output_path)


# ======================================================================
# 内部渲染函数
# ======================================================================

def _render_frame(
	snap: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
	dpi: int,
) -> Any:
	"""渲染单帧到新 figure

	Args:
		snap: 单个快照字典
		bus_coords: 母线坐标
		dpi: 分辨率

	Returns:
		matplotlib.figure.Figure 对象
	"""
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(12, 8), dpi=dpi)
	_draw_on_figure(fig, snap, bus_coords)
	return fig


def _draw_on_figure(
	fig: Any,
	snap: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""在给定 figure 上绘制单帧内容

	绘制包含 4 个子图的仪表盘:
	- 左上: 网络拓扑 (母线电压着色)
	- 右上: 功率平衡柱状图
	- 左下: 设备出力时序 (当前步标记)
	- 右下: 系统指标摘要文本

	Args:
		fig: matplotlib Figure 对象
		snap: 单个快照字典
		bus_coords: 母线坐标
	"""
	step = snap.get("step", 0)
	timestamp = snap.get("timestamp_h", step * 0.25)
	circuit = snap.get("circuit", {})

	fig.suptitle(
		f"Step {step}  |  t = {timestamp:.2f} h",
		fontsize=14,
		fontweight="bold",
	)

	# -- 左上: 简化拓扑 (母线电压散点) --
	ax1 = fig.add_subplot(2, 2, 1)
	_draw_topology(ax1, snap, bus_coords)

	# -- 右上: 功率平衡 --
	ax2 = fig.add_subplot(2, 2, 2)
	_draw_power_balance(ax2, circuit)

	# -- 左下: 设备状态 --
	ax3 = fig.add_subplot(2, 2, 3)
	_draw_device_status(ax3, snap)

	# -- 右下: 系统指标 --
	ax4 = fig.add_subplot(2, 2, 4)
	_draw_system_metrics(ax4, snap)

	fig.tight_layout(rect=[0, 0, 1, 0.95])


def _draw_topology(
	ax: Any,
	snap: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> None:
	"""绘制母线电压散点图

	Args:
		ax: matplotlib Axes 对象
		snap: 快照字典
		bus_coords: 母线坐标
	"""
	buses = snap.get("buses", {})
	xs, ys, colors = [], [], []

	for bus_name, coord in bus_coords.items():
		xs.append(coord[0])
		ys.append(coord[1])
		bus_data = buses.get(bus_name, {})
		v_mean = bus_data.get("v_mean", 1.0)
		colors.append(v_mean)

	if xs:
		sc = ax.scatter(
			xs, ys, c=colors, cmap="RdYlGn",
			vmin=0.90, vmax=1.10, s=30, edgecolors="white", linewidths=0.5,
		)
		ax.get_figure().colorbar(sc, ax=ax, label="Voltage (pu)", shrink=0.8)

	ax.set_title("Bus Voltage Map")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")


def _draw_power_balance(ax: Any, circuit: Dict[str, Any]) -> None:
	"""绘制功率平衡柱状图

	Args:
		ax: matplotlib Axes 对象
		circuit: 系统级数据字典
	"""
	categories = ["Generation", "Load", "PV", "Storage", "Loss"]
	values = [
		circuit.get("total_gen_kw", 0),
		circuit.get("total_load_kw", 0),
		circuit.get("total_pv_kw", 0),
		circuit.get("total_storage_kw", 0),
		circuit.get("total_loss_kw", 0),
	]
	bar_colors = ["#10B981", "#6B7280", "#F59E0B", "#3B82F6", "#EF4444"]

	ax.barh(categories, values, color=bar_colors)
	ax.set_title("Power Balance (kW)")
	ax.set_xlabel("kW")


def _draw_device_status(ax: Any, snap: Dict[str, Any]) -> None:
	"""绘制设备状态摘要

	Args:
		ax: matplotlib Axes 对象
		snap: 快照字典
	"""
	devices = snap.get("devices", {})
	pvs = devices.get("pv", {})
	storages = devices.get("storage", {})
	evs = devices.get("ev", {})

	labels = []
	values = []

	# PV 出力
	for name, data in pvs.items():
		labels.append(f"PV:{name[:8]}")
		values.append(data.get("kw_output", 0))

	# 储能功率
	for name, data in storages.items():
		labels.append(f"ESS:{name[:8]}")
		values.append(data.get("kw_output", 0))

	# EV 负荷
	for name, data in evs.items():
		labels.append(f"EV:{name[:8]}")
		values.append(data.get("kw", 0))

	if labels:
		bar_colors = []
		for val in values:
			if val > 0:
				bar_colors.append("#10B981")
			elif val < 0:
				bar_colors.append("#EF4444")
			else:
				bar_colors.append("#6B7280")
		ax.barh(labels, values, color=bar_colors)
	ax.set_title("Device Output (kW)")
	ax.set_xlabel("kW")


def _draw_system_metrics(ax: Any, snap: Dict[str, Any]) -> None:
	"""绘制系统指标摘要文本

	Args:
		ax: matplotlib Axes 对象
		snap: 快照字典
	"""
	circuit = snap.get("circuit", {})
	rewards = snap.get("rewards")

	lines = [
		f"V_mean: {circuit.get('v_mean_pu', 'N/A'):.4f} pu"
		if isinstance(circuit.get("v_mean_pu"), (int, float))
		else "V_mean: N/A",

		f"V_min:  {circuit.get('v_min_pu', 'N/A'):.4f} pu"
		if isinstance(circuit.get("v_min_pu"), (int, float))
		else "V_min:  N/A",

		f"V_max:  {circuit.get('v_max_pu', 'N/A'):.4f} pu"
		if isinstance(circuit.get("v_max_pu"), (int, float))
		else "V_max:  N/A",

		f"Loss:   {circuit.get('total_loss_kw', 0):.2f} kW",
		f"Converged: {circuit.get('converged', 'N/A')}",
	]

	if isinstance(rewards, list) and rewards:
		total_r = sum(r for r in rewards if isinstance(r, (int, float)))
		lines.append(f"Reward: {total_r:.4f}")

	ax.axis("off")
	text = "\n".join(lines)
	ax.text(
		0.1, 0.9, text,
		transform=ax.transAxes,
		fontsize=11,
		verticalalignment="top",
		fontfamily="monospace",
	)
	ax.set_title("System Metrics")
