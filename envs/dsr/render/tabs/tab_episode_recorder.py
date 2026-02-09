# -*- coding: utf-8 -*-
"""
DSR Tab: Episode Recorder
Episode 录制管理标签页

录制、保存和加载 episode 数据，
支持 .npz 格式的 episode 文件管理。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr

from envs.render_common.engine.episode_reader import EpisodeData, EpisodeReader
from envs.render_common.engine.episode_recorder import EpisodeRecorder

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Episode Recorder 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Episode Recorder"):

		gr.Markdown("### Episode Recording & Playback Management")

		gr.Markdown("#### Save Current Episode")

		with gr.Row():
			source_dropdown = gr.Dropdown(
				label="Data Source",
				choices=["Live Inference", "Loaded Recording"],
				value="Live Inference",
			)
			save_dir_input = gr.Textbox(
				label="Save Directory",
				value="recorded_episodes",
			)

		with gr.Row():
			save_btn = gr.Button("Save Episode (.npz)", variant="primary")
			save_status = gr.Textbox(
				label="Save Status",
				interactive=False,
				value="",
			)

		gr.Markdown("#### Load Recorded Episode")

		with gr.Row():
			recording_dir_input = gr.Textbox(
				label="Recording Directory",
				value="recorded_episodes",
				placeholder="Path to recorded episode files",
			)
			scan_btn = gr.Button("Scan Recordings", variant="secondary")

		recording_dropdown = gr.Dropdown(
			label="Recorded Episode",
			choices=[],
			interactive=True,
		)

		with gr.Row():
			load_btn = gr.Button("Load Recording", variant="secondary")
			rec_status = gr.Textbox(
				label="Recording Status",
				interactive=False,
				value="No recording loaded",
			)

		rec_info_json = gr.JSON(label="Recording Info")

	# --- 回调 ---

	def _get_snapshots(source: str) -> List[Dict[str, Any]]:
		"""获取快照列表"""
		if source == "Live Inference":
			return shared_states.get("live_snapshots", [])
		else:
			ep = shared_states.get("loaded_episode")
			return ep.snapshots if ep else []

	def _save_episode(source: str, save_dir: str):
		"""保存当前 episode"""
		snapshots = _get_snapshots(source)
		if not snapshots:
			return "Error: No snapshots available"

		# 提取恢复信息
		last_rest = snapshots[-1].get("restoration_data", {}) if snapshots else {}

		episode_data = EpisodeData(
			snapshots=snapshots,
			total_reward=snapshots[-1].get("cumulative_reward", 0.0) if snapshots else 0.0,
			episode_length=len(snapshots),
			config_summary={
				"env_name": "dsr",
				"system_name": shared_states.get("live_system", "Unknown"),
				"final_restoration_pct": last_rest.get("restoration_pct", 0.0),
			},
			metadata={
				"algorithm": "unknown",
				"seed": 0,
				"n_faults": len(last_rest.get("fault_lines", [])),
			},
		)

		recorder = EpisodeRecorder(save_dir)
		filepath = recorder.save_episode(episode_data)
		return f"Saved: {filepath}"

	def _scan_recordings(rec_dir: str):
		"""扫描已录制的 episode"""
		recordings = EpisodeRecorder.list_recordings(rec_dir)
		shared_states["recordings"] = recordings

		if not recordings:
			return gr.update(choices=[], value=None), "No recordings found"

		choices = [r["filename"] for r in recordings]
		return gr.update(choices=choices, value=choices[0]), f"Found {len(recordings)} recordings"

	def _load_recording(filename: str, rec_dir: str):
		"""加载已录制的 episode"""
		recordings = shared_states.get("recordings", [])
		selected = next((r for r in recordings if r["filename"] == filename), None)
		if selected is None:
			return "Error: Recording not found", None

		try:
			episode_data = EpisodeReader.load_episode(selected["path"])
			shared_states["loaded_episode"] = episode_data

			info = {
				"episode_length": episode_data.episode_length,
				"total_reward": episode_data.total_reward,
				"config": episode_data.config_summary,
				"metadata": episode_data.metadata,
			}

			return (
				f"Loaded: {filename} "
				f"({episode_data.episode_length} steps, "
				f"reward={episode_data.total_reward:.4f})",
				info,
			)
		except Exception as e:
			logger.error(f"Recording loading failed: {e}")
			return f"Error: {e}", None

	# --- 绑定 ---

	save_btn.click(
		fn=_save_episode,
		inputs=[source_dropdown, save_dir_input],
		outputs=[save_status],
	)

	scan_btn.click(
		fn=_scan_recordings,
		inputs=[recording_dir_input],
		outputs=[recording_dropdown, rec_status],
	)

	load_btn.click(
		fn=_load_recording,
		inputs=[recording_dropdown, recording_dir_input],
		outputs=[rec_status, rec_info_json],
	)

	components["save_status"] = save_status
	components["rec_status"] = rec_status
	components["rec_info_json"] = rec_info_json

	return components
