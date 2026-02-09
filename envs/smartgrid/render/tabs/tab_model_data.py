# -*- coding: utf-8 -*-
"""
SmartGrid Tab: Model & Data
模型与数据加载标签页

提供模型检查点扫描、加载和 episode 录制文件管理功能。
"""

import logging
import os
from typing import Any, Dict, Optional

import gradio as gr

from envs.render_common.engine.checkpoint_loader import CheckpointLoader
from envs.render_common.engine.episode_reader import EpisodeReader
from envs.render_common.engine.episode_recorder import EpisodeRecorder
from envs.render_common.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Model & Data 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Model & Data"):

		gr.Markdown("### Model Checkpoint")

		with gr.Row():
			results_dir_input = gr.Textbox(
				label="Results Directory",
				value="results",
				placeholder="Path to training results directory",
			)
			scan_btn = gr.Button("Scan", variant="secondary")

		run_dropdown = gr.Dropdown(
			label="Training Run",
			choices=[],
			interactive=True,
		)
		checkpoint_dropdown = gr.Dropdown(
			label="Checkpoint",
			choices=[],
			interactive=True,
		)

		with gr.Row():
			load_model_btn = gr.Button("Load Model", variant="primary")
			model_status = gr.Textbox(
				label="Status",
				interactive=False,
				value="No model loaded",
			)

		gr.Markdown("### Episode Recording")

		with gr.Row():
			recording_dir_input = gr.Textbox(
				label="Recording Directory",
				value="recorded_episodes",
				placeholder="Path to recorded episode files",
			)
			scan_rec_btn = gr.Button("Scan Recordings", variant="secondary")

		recording_dropdown = gr.Dropdown(
			label="Recorded Episode",
			choices=[],
			interactive=True,
		)

		with gr.Row():
			load_rec_btn = gr.Button("Load Recording", variant="secondary")
			rec_status = gr.Textbox(
				label="Recording Status",
				interactive=False,
				value="No recording loaded",
			)

		model_info_json = gr.JSON(label="Model Info", visible=False)

	# --- 回调 ---

	def _scan_runs(results_dir: str):
		"""扫描训练运行目录"""
		runs = CheckpointLoader.scan_training_runs(results_dir)
		shared_states["training_runs"] = runs

		if not runs:
			return gr.update(choices=[], value=None), "No runs found"

		choices = [r["name"] for r in runs]
		return gr.update(choices=choices, value=choices[0]), f"Found {len(runs)} runs"

	def _on_run_selected(run_name: str, results_dir: str):
		"""选择训练运行后更新检查点下拉框"""
		runs = shared_states.get("training_runs", [])
		selected = next((r for r in runs if r["name"] == run_name), None)
		if selected is None:
			return gr.update(choices=[], value=None)

		checkpoints = selected.get("checkpoints", [])
		if not checkpoints:
			return gr.update(choices=[], value=None)

		choices = [c["name"] for c in checkpoints]
		return gr.update(choices=choices, value=choices[0])

	def _load_model(run_name: str, ckpt_name: str, results_dir: str):
		"""加载选定的模型检查点"""
		runs = shared_states.get("training_runs", [])
		selected_run = next((r for r in runs if r["name"] == run_name), None)
		if selected_run is None:
			return "Error: Run not found", None

		checkpoints = selected_run.get("checkpoints", [])
		selected_ckpt = next((c for c in checkpoints if c["name"] == ckpt_name), None)
		if selected_ckpt is None:
			return "Error: Checkpoint not found", None

		ckpt_path = selected_ckpt["path"]
		n_agents = selected_ckpt["n_agents"]

		# SmartGrid 默认: 同构 agent，obs/action 维度从配置获取
		# 这里先用占位值，实际应从 config 或 training_state 中解析
		obs_dims = [64] * n_agents
		action_dims = [10] * n_agents

		try:
			engine = InferenceEngine(
				checkpoint_dir=ckpt_path,
				n_agents=n_agents,
				obs_dims=obs_dims,
				action_dims=action_dims,
				device="cpu",
			)
			engine.load_actors()

			shared_states["inference_engine"] = engine
			info = engine.get_model_info()

			return f"Loaded: {ckpt_name} ({n_agents} agents)", info

		except Exception as e:
			logger.error(f"Model loading failed: {e}")
			return f"Error: {e}", None

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
			return "Error: Recording not found"

		try:
			episode_data = EpisodeReader.load_episode(selected["path"])
			shared_states["loaded_episode"] = episode_data
			return (
				f"Loaded: {filename} "
				f"({episode_data.episode_length} steps, "
				f"reward={episode_data.total_reward:.4f})"
			)
		except Exception as e:
			logger.error(f"Recording loading failed: {e}")
			return f"Error: {e}"

	# --- 绑定 ---

	scan_btn.click(
		fn=_scan_runs,
		inputs=[results_dir_input],
		outputs=[run_dropdown, model_status],
	)

	run_dropdown.change(
		fn=_on_run_selected,
		inputs=[run_dropdown, results_dir_input],
		outputs=[checkpoint_dropdown],
	)

	load_model_btn.click(
		fn=_load_model,
		inputs=[run_dropdown, checkpoint_dropdown, results_dir_input],
		outputs=[model_status, model_info_json],
	)

	scan_rec_btn.click(
		fn=_scan_recordings,
		inputs=[recording_dir_input],
		outputs=[recording_dropdown, rec_status],
	)

	load_rec_btn.click(
		fn=_load_recording,
		inputs=[recording_dropdown, recording_dir_input],
		outputs=[rec_status],
	)

	components["results_dir_input"] = results_dir_input
	components["run_dropdown"] = run_dropdown
	components["checkpoint_dropdown"] = checkpoint_dropdown
	components["model_status"] = model_status
	components["model_info_json"] = model_info_json

	return components
