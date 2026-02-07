# -*- coding: utf-8 -*-
"""
SmartGrid Tab: Export
数据导出标签页

提供 CSV/JSON/HTML 报告/GIF 动画等多种导出功能。
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import gradio as gr

from envs.render_common.engine.episode_recorder import EpisodeRecorder
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.export.json_exporter import export_episode_json, export_summary_json
from envs.smartgrid.render.export.csv_exporter import SmartGridCsvExporter
from envs.smartgrid.render.export.report_generator import SmartGridReportGenerator
from envs.smartgrid.render.viz.mpl.episode_animation import export_smartgrid_animation
from envs.smartgrid.render.assets.bus_coordinates import get_bus_coordinates

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Export 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Export"):

		gr.Markdown("### Data Export & Report Generation")

		with gr.Row():
			source_dropdown = gr.Dropdown(
				label="Data Source",
				choices=["Live Inference", "Loaded Recording"],
				value="Live Inference",
			)

		gr.Markdown("#### CSV Export")

		with gr.Row():
			csv_categories = gr.CheckboxGroup(
				label="Export Categories",
				choices=SmartGridCsvExporter.EXPORT_CATEGORIES,
				value=SmartGridCsvExporter.EXPORT_CATEGORIES,
			)

		with gr.Row():
			export_csv_btn = gr.Button("Export CSV (ZIP)", variant="secondary")
			csv_download = gr.File(label="Download CSV")

		gr.Markdown("#### JSON Export")

		with gr.Row():
			export_json_btn = gr.Button("Export Episode JSON", variant="secondary")
			export_summary_btn = gr.Button("Export Summary JSON", variant="secondary")
			json_download = gr.File(label="Download JSON")

		gr.Markdown("#### HTML Report")

		with gr.Row():
			export_report_btn = gr.Button("Generate HTML Report", variant="primary")
			report_download = gr.File(label="Download Report")

		gr.Markdown("#### Animation Export")

		with gr.Row():
			anim_format = gr.Dropdown(
				label="Format",
				choices=["gif", "mp4"],
				value="gif",
			)
			anim_fps = gr.Number(label="FPS", value=4, precision=0)
			export_anim_btn = gr.Button("Export Animation", variant="secondary")
			anim_download = gr.File(label="Download Animation")

		gr.Markdown("#### Episode Recording")

		with gr.Row():
			save_dir_input = gr.Textbox(
				label="Save Directory",
				value="recorded_episodes",
			)
			save_rec_btn = gr.Button("Save Episode (.npz)", variant="secondary")
			save_status = gr.Textbox(
				label="Save Status",
				interactive=False,
				value="",
			)

	# --- 回调 ---

	def _get_snapshots(source: str) -> List[Dict[str, Any]]:
		"""获取快照列表"""
		if source == "Live Inference":
			return shared_states.get("live_snapshots", [])
		else:
			ep = shared_states.get("loaded_episode")
			return ep.snapshots if ep else []

	def _export_csv(source: str, categories: List[str]):
		"""导出 CSV"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		exporter = SmartGridCsvExporter()
		tmp_path = os.path.join(tempfile.gettempdir(), "smartgrid_export.zip")
		exporter.export_selected(snapshots, categories, tmp_path)
		return tmp_path

	def _export_json(source: str):
		"""导出 Episode JSON"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		tmp_path = os.path.join(tempfile.gettempdir(), "smartgrid_episode.json")
		export_episode_json(snapshots, tmp_path)
		return tmp_path

	def _export_summary(source: str):
		"""导出摘要 JSON"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		tmp_path = os.path.join(tempfile.gettempdir(), "smartgrid_summary.json")
		export_summary_json(snapshots, tmp_path)
		return tmp_path

	def _export_report(source: str):
		"""生成 HTML 报告"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		generator = SmartGridReportGenerator()
		tmp_path = os.path.join(tempfile.gettempdir(), "smartgrid_report.html")

		metadata = {
			"system_name": shared_states.get("live_system", "Unknown"),
			"algorithm": "Unknown",
		}

		generator.generate_report(snapshots, metadata, tmp_path)
		return tmp_path

	def _export_animation(source: str, fmt: str, fps: int):
		"""导出动画"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		system_name = shared_states.get("live_system", "34Bus_PV")
		bus_coords = get_bus_coordinates(system_name)

		ext = "gif" if fmt == "gif" else "mp4"
		tmp_path = os.path.join(tempfile.gettempdir(), f"smartgrid_animation.{ext}")

		export_smartgrid_animation(
			snapshots=snapshots,
			bus_coords=bus_coords,
			output_path=tmp_path,
			fps=int(fps),
			format=fmt,
		)
		return tmp_path

	def _save_recording(source: str, save_dir: str):
		"""保存 episode 录制"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return "Error: No snapshots available"

		episode_data = EpisodeData(
			snapshots=snapshots,
			total_reward=snapshots[-1].get("cumulative_reward", 0.0) if snapshots else 0.0,
			episode_length=len(snapshots),
			config_summary={
				"env_name": "smartgrid",
				"system_name": shared_states.get("live_system", "Unknown"),
			},
			metadata={
				"algorithm": "unknown",
				"seed": 0,
			},
		)

		recorder = EpisodeRecorder(save_dir)
		filepath = recorder.save_episode(episode_data)
		return f"Saved: {filepath}"

	# --- 绑定 ---

	export_csv_btn.click(
		fn=_export_csv,
		inputs=[source_dropdown, csv_categories],
		outputs=[csv_download],
	)

	export_json_btn.click(
		fn=_export_json,
		inputs=[source_dropdown],
		outputs=[json_download],
	)

	export_summary_btn.click(
		fn=_export_summary,
		inputs=[source_dropdown],
		outputs=[json_download],
	)

	export_report_btn.click(
		fn=_export_report,
		inputs=[source_dropdown],
		outputs=[report_download],
	)

	export_anim_btn.click(
		fn=_export_animation,
		inputs=[source_dropdown, anim_format, anim_fps],
		outputs=[anim_download],
	)

	save_rec_btn.click(
		fn=_save_recording,
		inputs=[source_dropdown, save_dir_input],
		outputs=[save_status],
	)

	components["csv_download"] = csv_download
	components["report_download"] = report_download

	return components
