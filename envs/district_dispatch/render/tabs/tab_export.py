# -*- coding: utf-8 -*-
"""
Tab 8: Export Center
数据导出中心标签页 -- CSV/动画/HTML报告/JSON 等多格式导出

功能:
- CSV 导出: 选择数据类别 → zip 下载
- 动画导出: GIF/MP4 + FPS/DPI 配置
- 报告导出: HTML Report / Interactive Topology / Episode NPZ / Summary JSON
"""

import json
import logging
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from envs.district_dispatch.render.engine.episode_runner import EpisodeData
from envs.district_dispatch.render.export import (
	EXPORT_CATEGORIES,
	check_ffmpeg,
	export_gif,
	export_mp4,
	export_selected_csv,
	export_summary_json,
	export_topology_html,
	generate_report,
)

logger = logging.getLogger(__name__)

# 导出类别显示名称
_CATEGORY_LABELS: Dict[str, str] = {
	"bus_voltages": "Bus Voltages",
	"line_data": "Line Data",
	"transformer_data": "Transformer",
	"pv_data": "PV",
	"storage_data": "Storage",
	"ev_data": "EV",
	"regulator_data": "Regulator",
	"system_totals": "System Totals",
	"agent_actions": "Agent Actions",
	"agent_rewards": "Agent Rewards",
}

# 显示名 → 类别名反向映射
_LABEL_TO_CATEGORY: Dict[str, str] = {v: k for k, v in _CATEGORY_LABELS.items()}

# 可选的类别列表 (显示名)
_CATEGORY_CHOICES: List[str] = [
	_CATEGORY_LABELS.get(c, c) for c in EXPORT_CATEGORIES
]


def _get_snapshots(snapshots: Any) -> Optional[List[Dict[str, Any]]]:
	"""从 State 值中安全获取快照列表。"""
	if snapshots is None:
		return None
	if isinstance(snapshots, list) and len(snapshots) > 0:
		return snapshots
	return None


def _export_csv(
	snapshots: Any,
	selected_labels: List[str],
) -> Tuple[Optional[str], str]:
	"""导出选定类别的 CSV zip 文件。

	Returns:
		(file_path_or_none, status_msg)
	"""
	snaps = _get_snapshots(snapshots)
	if snaps is None:
		return None, "No snapshot data available"

	if not selected_labels:
		return None, "No categories selected"

	# 将显示名转换回类别名
	categories = []
	for label in selected_labels:
		cat = _LABEL_TO_CATEGORY.get(label, label)
		if cat in EXPORT_CATEGORIES:
			categories.append(cat)

	if not categories:
		return None, "No valid categories selected"

	try:
		csv_bytes = export_selected_csv(snaps, categories)

		# 写入临时文件供下载
		tmp_dir = tempfile.mkdtemp(prefix="powerzoo_export_")
		out_path = os.path.join(tmp_dir, "episode_data.zip")
		with open(out_path, "wb") as f:
			f.write(csv_bytes)

		msg = f"Exported {len(categories)} categories ({len(csv_bytes)} bytes)"
		return out_path, msg
	except Exception as e:
		logger.error(f"CSV export failed: {e}\n{traceback.format_exc()}")
		return None, f"Export failed: {e}"


def _export_animation(
	snapshots: Any,
	bus_coords: Any,
	fmt: str,
	fps: int,
	dpi: int,
) -> Tuple[Optional[str], str]:
	"""导出 GIF/MP4 动画。

	Returns:
		(file_path_or_none, status_msg)
	"""
	snaps = _get_snapshots(snapshots)
	if snaps is None:
		return None, "No snapshot data available"

	if not bus_coords:
		return None, "No bus coordinates available"

	fps = max(1, min(10, int(fps)))
	dpi = max(72, min(300, int(dpi)))

	tmp_dir = tempfile.mkdtemp(prefix="powerzoo_anim_")
	fmt_lower = fmt.lower().strip()

	try:
		if fmt_lower == "mp4":
			if not check_ffmpeg():
				return None, "ffmpeg not installed. Cannot export MP4."
			out_path = os.path.join(tmp_dir, "episode.mp4")
			result_path = export_mp4(
				snaps, bus_coords, out_path,
				fps=fps, dpi=dpi,
			)
		else:
			out_path = os.path.join(tmp_dir, "episode.gif")
			result_path = export_gif(
				snaps, bus_coords, out_path,
				fps=fps, dpi=dpi,
			)

		msg = f"Animation exported: {os.path.basename(result_path)}"
		return result_path, msg

	except Exception as e:
		logger.error(f"Animation export failed: {e}\n{traceback.format_exc()}")
		return None, f"Animation export failed: {e}"


def _export_html_report(
	snapshots: Any,
	bus_coords: Any,
) -> Tuple[Optional[str], str]:
	"""生成 HTML 分析报告。

	Returns:
		(file_path_or_none, status_msg)
	"""
	snaps = _get_snapshots(snapshots)
	if snaps is None:
		return None, "No snapshot data available"

	if not bus_coords:
		return None, "No bus coordinates available"

	try:
		tmp_dir = tempfile.mkdtemp(prefix="powerzoo_report_")
		out_path = os.path.join(tmp_dir, "episode_report.html")
		result_path = generate_report(snaps, bus_coords, out_path)
		return result_path, "HTML report generated"
	except Exception as e:
		logger.error(f"Report export failed: {e}\n{traceback.format_exc()}")
		return None, f"Report export failed: {e}"


def _export_topology(
	snapshots: Any,
	bus_coords: Any,
) -> Tuple[Optional[str], str]:
	"""导出交互式拓扑 HTML。

	Returns:
		(file_path_or_none, status_msg)
	"""
	snaps = _get_snapshots(snapshots)
	if snaps is None:
		return None, "No snapshot data available"

	if not bus_coords:
		return None, "No bus coordinates available"

	try:
		# 使用最后一个快照
		snapshot = snaps[-1]
		tmp_dir = tempfile.mkdtemp(prefix="powerzoo_topo_")
		out_path = os.path.join(tmp_dir, "topology.html")
		result_path = export_topology_html(snapshot, bus_coords, out_path)
		return result_path, "Interactive topology exported"
	except Exception as e:
		logger.error(f"Topology export failed: {e}\n{traceback.format_exc()}")
		return None, f"Topology export failed: {e}"


def _export_episode_npz(
	snapshots: Any,
) -> Tuple[Optional[str], str]:
	"""将 episode 数据导出为 .npz 文件。

	Returns:
		(file_path_or_none, status_msg)
	"""
	snaps = _get_snapshots(snapshots)
	if snaps is None:
		return None, "No snapshot data available"

	try:
		tmp_dir = tempfile.mkdtemp(prefix="powerzoo_npz_")
		out_path = os.path.join(tmp_dir, "episode_data.npz")

		# 提取可数值化的数据
		steps = []
		timestamps = []
		all_rewards = []

		for snap in snaps:
			steps.append(snap.get("step", 0))
			timestamps.append(snap.get("timestamp_h", 0.0))
			rewards = snap.get("rewards")
			if isinstance(rewards, list):
				all_rewards.append(rewards)

		save_dict = {
			"steps": np.array(steps),
			"timestamps": np.array(timestamps),
		}
		if all_rewards:
			save_dict["rewards"] = np.array(all_rewards, dtype=object)

		np.savez_compressed(out_path, **save_dict)
		return out_path, f"Episode data exported to NPZ ({len(snaps)} steps)"

	except Exception as e:
		logger.error(f"NPZ export failed: {e}\n{traceback.format_exc()}")
		return None, f"NPZ export failed: {e}"


def _export_json_summary(
	snapshots: Any,
) -> Tuple[Optional[str], str]:
	"""导出 JSON 汇总。

	Returns:
		(file_path_or_none, status_msg)
	"""
	snaps = _get_snapshots(snapshots)
	if snaps is None:
		return None, "No snapshot data available"

	try:
		json_str = export_summary_json(snaps)
		tmp_dir = tempfile.mkdtemp(prefix="powerzoo_json_")
		out_path = os.path.join(tmp_dir, "summary.json")
		with open(out_path, "w", encoding="utf-8") as f:
			f.write(json_str)
		return out_path, "Summary JSON exported"
	except Exception as e:
		logger.error(f"JSON export failed: {e}\n{traceback.format_exc()}")
		return None, f"JSON export failed: {e}"


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Export Center 标签页。

	Args:
		shared_states: 共享状态字典，包含:
			- "snapshots": gr.State
			- "bus_coords": gr.State
			- "model_loaded": gr.State
			- "inference_engine": gr.State
			- "episode_runner": gr.State

	Returns:
		该 Tab 内关键组件引用
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Export Center"):
		gr.Markdown("## Export Center")
		gr.Markdown(
			"Export episode data in various formats: CSV, animation, "
			"HTML report, topology, and more."
		)

		# -- 进度/状态 --
		export_status = gr.Textbox(
			label="Export Status",
			interactive=False,
			lines=1,
		)

		# ===== CSV 导出区 =====
		gr.Markdown("### CSV Data Export")
		with gr.Row():
			csv_categories = gr.CheckboxGroup(
				choices=_CATEGORY_CHOICES,
				value=_CATEGORY_CHOICES,
				label="Select Data Categories",
			)
		with gr.Row():
			btn_csv = gr.Button("Export CSV (zip)", variant="primary")
			csv_file = gr.File(label="Download CSV", interactive=False)

		# ===== 动画导出区 =====
		gr.Markdown("### Animation Export")
		with gr.Row():
			anim_format = gr.Dropdown(
				choices=["GIF", "MP4"],
				value="GIF",
				label="Format",
				scale=1,
			)
			anim_fps = gr.Slider(
				minimum=1,
				maximum=10,
				value=4,
				step=1,
				label="FPS",
				scale=1,
			)
			anim_dpi = gr.Slider(
				minimum=72,
				maximum=300,
				value=150,
				step=10,
				label="DPI",
				scale=1,
			)
		with gr.Row():
			btn_anim = gr.Button("Generate Animation", variant="secondary")
			anim_file = gr.File(label="Download Animation", interactive=False)

		# ===== 报告导出区 =====
		gr.Markdown("### Report & Data Export")
		with gr.Row():
			btn_report = gr.Button("Generate HTML Report")
			report_file = gr.File(label="Download Report", interactive=False)
		with gr.Row():
			btn_topology = gr.Button("Export Interactive Topology")
			topology_file = gr.File(label="Download Topology", interactive=False)
		with gr.Row():
			btn_npz = gr.Button("Export Episode Data (.npz)")
			npz_file = gr.File(label="Download NPZ", interactive=False)
		with gr.Row():
			btn_json = gr.Button("Export Summary JSON")
			json_file = gr.File(label="Download JSON", interactive=False)

		components.update({
			"export_status": export_status,
			"csv_categories": csv_categories,
			"btn_csv": btn_csv,
			"csv_file": csv_file,
			"anim_format": anim_format,
			"anim_fps": anim_fps,
			"anim_dpi": anim_dpi,
			"btn_anim": btn_anim,
			"anim_file": anim_file,
			"btn_report": btn_report,
			"report_file": report_file,
			"btn_topology": btn_topology,
			"topology_file": topology_file,
			"btn_npz": btn_npz,
			"npz_file": npz_file,
			"btn_json": btn_json,
			"json_file": json_file,
		})

		# -- 事件: CSV 导出 --
		btn_csv.click(
			fn=_export_csv,
			inputs=[
				shared_states["snapshots"],
				csv_categories,
			],
			outputs=[csv_file, export_status],
		)

		# -- 事件: 动画导出 --
		btn_anim.click(
			fn=_export_animation,
			inputs=[
				shared_states["snapshots"],
				shared_states["bus_coords"],
				anim_format,
				anim_fps,
				anim_dpi,
			],
			outputs=[anim_file, export_status],
		)

		# -- 事件: HTML 报告 --
		btn_report.click(
			fn=_export_html_report,
			inputs=[
				shared_states["snapshots"],
				shared_states["bus_coords"],
			],
			outputs=[report_file, export_status],
		)

		# -- 事件: 拓扑导出 --
		btn_topology.click(
			fn=_export_topology,
			inputs=[
				shared_states["snapshots"],
				shared_states["bus_coords"],
			],
			outputs=[topology_file, export_status],
		)

		# -- 事件: NPZ 导出 --
		btn_npz.click(
			fn=_export_episode_npz,
			inputs=[shared_states["snapshots"]],
			outputs=[npz_file, export_status],
		)

		# -- 事件: JSON 导出 --
		btn_json.click(
			fn=_export_json_summary,
			inputs=[shared_states["snapshots"]],
			outputs=[json_file, export_status],
		)

	return components
