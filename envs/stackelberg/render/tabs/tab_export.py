# -*- coding: utf-8 -*-
"""
Tab 8: Export

数据导出面板，支持 CSV、HTML 报告、动画 (GIF/MP4)、
拓扑图、NPZ、JSON 等多种格式。
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Export 标签页

	Args:
		shared_states: 共享状态字典
	"""
	gr.Markdown("### Data Export")

	with gr.Row():
		output_dir = gr.Textbox(
			label="Output Directory",
			value=os.path.join(tempfile.gettempdir(), "stackelberg_export"),
		)

	with gr.Row():
		with gr.Column():
			gr.Markdown("#### CSV Export")
			csv_categories = gr.CheckboxGroup(
				label="Categories",
				choices=[
					"bus_voltages", "system_totals",
					"agent_actions", "agent_rewards",
					"market_data", "leader_actions", "follower_actions",
				],
				value=["market_data", "leader_actions", "follower_actions"],
			)
			csv_btn = gr.Button("Export CSV (ZIP)", variant="primary")
			csv_file = gr.File(label="Download CSV ZIP")

		with gr.Column():
			gr.Markdown("#### HTML Report")
			report_btn = gr.Button("Generate HTML Report", variant="primary")
			report_file = gr.File(label="Download Report")

	with gr.Row():
		with gr.Column():
			gr.Markdown("#### Animation")
			anim_format = gr.Radio(
				label="Format", choices=["GIF", "MP4"], value="GIF",
			)
			anim_btn = gr.Button("Generate Animation", variant="secondary")
			anim_file = gr.File(label="Download Animation")

		with gr.Column():
			gr.Markdown("#### Raw Data")
			raw_format = gr.Radio(
				label="Format", choices=["JSON", "NPZ"], value="JSON",
			)
			raw_btn = gr.Button("Export Raw Data", variant="secondary")
			raw_file = gr.File(label="Download Raw Data")

	export_status = gr.Markdown("")

	# ------------------------------------------------------------------
	# 回调
	# ------------------------------------------------------------------

	def _export_csv(snapshots, categories, out_dir):
		"""导出 CSV"""
		if not snapshots or not isinstance(snapshots, list):
			return None, "No data to export."

		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, "stackelberg_data.zip")

		try:
			from envs.stackelberg.render.export.csv_exporter import StackelbergCsvExporter
			exporter = StackelbergCsvExporter()
			exporter.export_selected(snapshots, list(categories), out_path)
			return out_path, f"CSV exported: {out_path}"
		except Exception as exc:
			logger.error(f"CSV export failed: {exc}")
			return None, f"Export failed: {exc}"

	def _export_report(snapshots, out_dir):
		"""生成 HTML 报告"""
		if not snapshots or not isinstance(snapshots, list):
			return None, "No data to export."

		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, "stackelberg_report.html")

		try:
			from envs.stackelberg.render.export.report_generator import StackelbergReportGenerator
			generator = StackelbergReportGenerator()
			generator.generate_report(snapshots, output_path=out_path)
			return out_path, f"Report generated: {out_path}"
		except Exception as exc:
			logger.error(f"Report generation failed: {exc}")
			return None, f"Report failed: {exc}"

	def _export_animation(snapshots, fmt, out_dir):
		"""生成动画"""
		if not snapshots or not isinstance(snapshots, list):
			return None, "No data to export."

		os.makedirs(out_dir, exist_ok=True)
		ext = ".gif" if fmt == "GIF" else ".mp4"
		out_path = os.path.join(out_dir, f"stackelberg_animation{ext}")

		try:
			from envs.stackelberg.render.viz.mpl.episode_animation import create_episode_animation
			create_episode_animation(snapshots, output_path=out_path)
			if os.path.isfile(out_path):
				return out_path, f"Animation saved: {out_path}"
			return None, "Animation generation failed (no output file)"
		except Exception as exc:
			logger.error(f"Animation failed: {exc}")
			return None, f"Animation failed: {exc}"

	def _export_raw(snapshots, fmt, out_dir):
		"""导出原始数据"""
		if not snapshots or not isinstance(snapshots, list):
			return None, "No data to export."

		os.makedirs(out_dir, exist_ok=True)

		if fmt == "JSON":
			out_path = os.path.join(out_dir, "stackelberg_snapshots.json")
			try:
				# 将 numpy 数组转为列表
				def _convert(obj):
					if isinstance(obj, np.ndarray):
						return obj.tolist()
					if isinstance(obj, (np.float32, np.float64)):
						return float(obj)
					if isinstance(obj, (np.int32, np.int64)):
						return int(obj)
					raise TypeError(f"Not serializable: {type(obj)}")

				with open(out_path, "w") as f:
					json.dump(snapshots, f, default=_convert, indent=2)
				return out_path, f"JSON exported: {out_path}"
			except Exception as exc:
				return None, f"JSON export failed: {exc}"
		else:
			out_path = os.path.join(out_dir, "stackelberg_snapshots.npz")
			try:
				np.savez_compressed(out_path, snapshots=np.array(snapshots, dtype=object))
				return out_path, f"NPZ exported: {out_path}"
			except Exception as exc:
				return None, f"NPZ export failed: {exc}"

	# 绑定事件
	csv_btn.click(
		_export_csv,
		inputs=[shared_states["snapshots"], csv_categories, output_dir],
		outputs=[csv_file, export_status],
	)
	report_btn.click(
		_export_report,
		inputs=[shared_states["snapshots"], output_dir],
		outputs=[report_file, export_status],
	)
	anim_btn.click(
		_export_animation,
		inputs=[shared_states["snapshots"], anim_format, output_dir],
		outputs=[anim_file, export_status],
	)
	raw_btn.click(
		_export_raw,
		inputs=[shared_states["snapshots"], raw_format, output_dir],
		outputs=[raw_file, export_status],
	)
