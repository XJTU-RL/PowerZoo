# -*- coding: utf-8 -*-
"""
DSR Tab: Data Export
数据导出标签页

提供 CSV/JSON/HTML 报告/GIF 动画等多种导出功能。
DSR 特有导出类别: restoration_states, action_masks, fault_data。
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import gradio as gr

from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.export.json_exporter import export_episode_json, export_summary_json
from envs.dsr.render.export.csv_exporter import DSRCsvExporter
from envs.dsr.render.export.report_generator import DSRReportGenerator
from envs.dsr.render.viz.mpl.episode_animation import export_episode_gif, export_episode_mp4
from envs.dsr.render.assets.bus_coordinates import get_bus_coordinates

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Data Export 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Data Export"):

		gr.Markdown("### Data Export & Report Generation")

		with gr.Row():
			source_dropdown = gr.Dropdown(
				label="Data Source",
				choices=["Live Inference", "Loaded Recording"],
				value="Live Inference",
			)

		gr.Markdown("#### CSV Export")
		gr.Markdown(
			"*DSR-specific categories: "
			"`restoration_states`, `action_masks`, `fault_data`*"
		)

		with gr.Row():
			csv_categories = gr.CheckboxGroup(
				label="Export Categories",
				choices=DSRCsvExporter.EXPORT_CATEGORIES,
				value=DSRCsvExporter.EXPORT_CATEGORIES,
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
		gr.Markdown("*DSR animations use 2 FPS (short episodes, 10-20 frames)*")

		with gr.Row():
			anim_format = gr.Dropdown(
				label="Format",
				choices=["gif", "mp4"],
				value="gif",
			)
			anim_fps = gr.Number(label="FPS", value=2, precision=0)
			export_anim_btn = gr.Button("Export Animation", variant="secondary")
			anim_download = gr.File(label="Download Animation")

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

		exporter = DSRCsvExporter()
		tmp_path = os.path.join(tempfile.gettempdir(), "dsr_export.zip")
		exporter.export_selected(snapshots, categories, tmp_path)
		return tmp_path

	def _export_json(source: str):
		"""导出 Episode JSON"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		tmp_path = os.path.join(tempfile.gettempdir(), "dsr_episode.json")
		export_episode_json(snapshots, tmp_path)
		return tmp_path

	def _export_summary(source: str):
		"""导出摘要 JSON"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		tmp_path = os.path.join(tempfile.gettempdir(), "dsr_summary.json")
		export_summary_json(snapshots, tmp_path)
		return tmp_path

	def _export_report(source: str):
		"""生成 HTML 报告"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		generator = DSRReportGenerator()
		tmp_path = os.path.join(tempfile.gettempdir(), "dsr_report.html")

		metadata = {
			"system_name": shared_states.get("live_system", "Unknown"),
			"algorithm": "Unknown",
		}

		# 提取总奖励
		if snapshots:
			metadata["total_reward"] = snapshots[-1].get("cumulative_reward", 0.0)

		generator.generate_report(snapshots, metadata, tmp_path)
		return tmp_path

	def _export_animation(source: str, fmt: str, fps: int):
		"""导出动画"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return None

		system_name = shared_states.get("live_system", "13Bus")
		bus_coords = get_bus_coordinates(system_name)

		ext = "gif" if fmt == "gif" else "mp4"
		tmp_path = os.path.join(tempfile.gettempdir(), f"dsr_animation.{ext}")

		if fmt == "gif":
			export_episode_gif(
				snapshots=snapshots,
				bus_coords=bus_coords,
				output_path=tmp_path,
				fps=int(fps),
			)
		else:
			export_episode_mp4(
				snapshots=snapshots,
				bus_coords=bus_coords,
				output_path=tmp_path,
				fps=int(fps),
			)
		return tmp_path

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

	components["csv_download"] = csv_download
	components["report_download"] = report_download

	return components
