# -*- coding: utf-8 -*-
"""
Tab 8: Export - 数据导出

支持 CSV、JSON、HTML 报告、GIF/MP4 动画导出。
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from envs.render_common.engine.episode_reader import EpisodeData, EpisodeReader
from envs.render_common.export.json_exporter import export_episode_json, export_summary_json
from envs.vvc.render.export.csv_exporter import VVCCsvExporter
from envs.vvc.render.export.topology_html_exporter import VVCReportGenerator
from envs.vvc.render.viz.mpl.episode_animation import export_vvc_animation

logger = logging.getLogger(__name__)


def _load_for_export(
	file_path: str,
) -> Tuple[Optional[EpisodeData], List[Dict[str, Any]], str]:
	"""加载 episode 用于导出。"""
	if not file_path:
		return None, [], "No file path"

	try:
		ep = EpisodeReader.load_episode(file_path)
		return ep, ep.snapshots, f"Loaded: {ep.episode_length} steps"
	except Exception as exc:
		logger.error(f"Load failed: {exc}", exc_info=True)
		return None, [], f"Load failed: {exc}"


def _export_csv(
	snapshots: List[Dict[str, Any]],
	output_dir: str,
	categories: List[str],
) -> Tuple[str, str]:
	"""导出 CSV 文件。"""
	if not snapshots:
		return "", "No data to export"

	if not output_dir:
		output_dir = tempfile.mkdtemp(prefix="vvc_csv_")

	os.makedirs(output_dir, exist_ok=True)

	try:
		exporter = VVCCsvExporter()

		if categories:
			files = exporter.export_selected(snapshots, output_dir, categories)
		else:
			files = exporter.export_all(snapshots, output_dir)

		file_list = "\n".join(f"  - {f}" for f in files)
		return output_dir, f"Exported {len(files)} CSV files to {output_dir}:\n{file_list}"

	except Exception as exc:
		logger.error(f"CSV export failed: {exc}", exc_info=True)
		return "", f"CSV export failed: {exc}"


def _export_json(
	episode: Optional[EpisodeData],
	snapshots: List[Dict[str, Any]],
	output_dir: str,
	include_snapshots: bool,
) -> Tuple[str, str]:
	"""导出 JSON 文件。"""
	if not snapshots:
		return "", "No data to export"

	if not output_dir:
		output_dir = tempfile.mkdtemp(prefix="vvc_json_")

	os.makedirs(output_dir, exist_ok=True)

	try:
		files: List[str] = []

		# 摘要 JSON
		summary_path = os.path.join(output_dir, "summary.json")
		metadata = {}
		if episode is not None:
			metadata = {
				"total_reward": episode.total_reward,
				"episode_length": episode.episode_length,
				"config_summary": episode.config_summary,
			}
		export_summary_json(snapshots, summary_path, metadata=metadata)
		files.append(summary_path)

		# 完整 episode JSON
		if include_snapshots:
			episode_path = os.path.join(output_dir, "episode.json")
			export_episode_json(snapshots, episode_path, metadata=metadata)
			files.append(episode_path)

		file_list = "\n".join(f"  - {f}" for f in files)
		return output_dir, f"Exported JSON to {output_dir}:\n{file_list}"

	except Exception as exc:
		logger.error(f"JSON export failed: {exc}", exc_info=True)
		return "", f"JSON export failed: {exc}"


def _export_html(
	episode: Optional[EpisodeData],
	snapshots: List[Dict[str, Any]],
	output_path: str,
) -> Tuple[str, str]:
	"""导出 HTML 报告。"""
	if not snapshots:
		return "", "No data to export"

	if not output_path:
		fd, output_path = tempfile.mkstemp(suffix=".html", prefix="vvc_report_")
		os.close(fd)

	try:
		generator = VVCReportGenerator()
		metadata = {}
		if episode is not None:
			metadata = {
				"total_reward": episode.total_reward,
				"episode_length": episode.episode_length,
			}

		result_path = generator.generate_report(
			snapshots=snapshots,
			output_path=output_path,
			metadata=metadata,
		)
		return result_path, f"HTML report exported: {result_path}"

	except Exception as exc:
		logger.error(f"HTML export failed: {exc}", exc_info=True)
		return "", f"HTML export failed: {exc}"


def _export_animation(
	snapshots: List[Dict[str, Any]],
	output_path: str,
	bus_coords: Dict,
	fps: int,
	fmt: str,
) -> Tuple[str, str]:
	"""导出动画文件。"""
	if not snapshots:
		return "", "No data to export"

	if not output_path:
		suffix = f".{fmt}"
		fd, output_path = tempfile.mkstemp(suffix=suffix, prefix="vvc_anim_")
		os.close(fd)

	try:
		result_path = export_vvc_animation(
			snapshots=snapshots,
			output_path=output_path,
			bus_coords=bus_coords if bus_coords else None,
			fps=fps,
			format=fmt,
		)
		if result_path:
			return result_path, f"Animation exported: {result_path}"
		return "", "Animation export returned empty"

	except Exception as exc:
		logger.error(f"Animation export failed: {exc}", exc_info=True)
		return "", f"Animation export failed: {exc}"


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Export Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	state_bus_coords = shared_states["bus_coords"]

	# 本地状态
	state_episode = gr.State(None)
	state_snapshots = gr.State([])

	# === 数据源 ===
	with gr.Group():
		gr.Markdown("### Data Source")
		with gr.Row():
			file_input = gr.Textbox(
				label="Episode File Path",
				placeholder="e.g., recorded_episodes/vvc_episode_001.npz",
				scale=4,
			)
			load_btn = gr.Button("Load", variant="primary", scale=1)

	# === CSV 导出 ===
	with gr.Accordion("CSV Export", open=True):
		with gr.Row():
			csv_dir_input = gr.Textbox(
				label="Output Directory",
				placeholder="Leave empty for temp directory",
				scale=3,
			)
			csv_categories = gr.CheckboxGroup(
				choices=VVCCsvExporter.EXPORT_CATEGORIES,
				value=VVCCsvExporter.EXPORT_CATEGORIES,
				label="Categories",
				scale=3,
			)
			csv_btn = gr.Button("Export CSV", variant="secondary", scale=1)

	# === JSON 导出 ===
	with gr.Accordion("JSON Export", open=False):
		with gr.Row():
			json_dir_input = gr.Textbox(
				label="Output Directory",
				placeholder="Leave empty for temp directory",
				scale=3,
			)
			json_full_cb = gr.Checkbox(
				label="Include Full Snapshots",
				value=False,
				scale=2,
			)
			json_btn = gr.Button("Export JSON", variant="secondary", scale=1)

	# === HTML 报告 ===
	with gr.Accordion("HTML Report", open=False):
		with gr.Row():
			html_path_input = gr.Textbox(
				label="Output Path",
				placeholder="Leave empty for temp file",
				scale=4,
			)
			html_btn = gr.Button("Export HTML", variant="secondary", scale=1)

	# === 动画导出 ===
	with gr.Accordion("Animation Export", open=False):
		with gr.Row():
			anim_path_input = gr.Textbox(
				label="Output Path",
				placeholder="Leave empty for temp file",
				scale=2,
			)
			anim_format = gr.Dropdown(
				choices=["gif", "mp4"],
				value="gif",
				label="Format",
				scale=1,
			)
			anim_fps = gr.Number(label="FPS", value=2, precision=0, scale=1)
			anim_btn = gr.Button("Export Animation", variant="secondary", scale=1)

	status_box = gr.Textbox(label="Status", interactive=False, lines=3)

	# --- 事件绑定 ---
	def on_load(file_path):
		ep, snaps, status = _load_for_export(file_path)
		return ep, snaps, status

	def on_csv(snapshots, output_dir, categories):
		_, status = _export_csv(snapshots, output_dir, categories)
		return status

	def on_json(episode, snapshots, output_dir, include_full):
		_, status = _export_json(episode, snapshots, output_dir, include_full)
		return status

	def on_html(episode, snapshots, output_path):
		_, status = _export_html(episode, snapshots, output_path)
		return status

	def on_anim(snapshots, output_path, bus_coords, fps, fmt):
		_, status = _export_animation(
			snapshots, output_path, bus_coords, int(fps), fmt,
		)
		return status

	load_btn.click(
		fn=on_load,
		inputs=[file_input],
		outputs=[state_episode, state_snapshots, status_box],
	)
	csv_btn.click(
		fn=on_csv,
		inputs=[state_snapshots, csv_dir_input, csv_categories],
		outputs=[status_box],
	)
	json_btn.click(
		fn=on_json,
		inputs=[state_episode, state_snapshots, json_dir_input, json_full_cb],
		outputs=[status_box],
	)
	html_btn.click(
		fn=on_html,
		inputs=[state_episode, state_snapshots, html_path_input],
		outputs=[status_box],
	)
	anim_btn.click(
		fn=on_anim,
		inputs=[state_snapshots, anim_path_input, state_bus_coords, anim_fps, anim_format],
		outputs=[status_box],
	)

	return {
		"status_box": status_box,
	}
