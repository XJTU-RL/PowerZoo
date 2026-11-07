# -*- coding: utf-8 -*-
"""
Tab 1: Model & Data

加载模型检查点和 episode 数据文件。
支持扫描结果目录、加载权重、列出已有 episode。
"""

import glob
import json
import logging
import os
from typing import Any, Dict

import gradio as gr

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Model & Data 标签页

	Args:
		shared_states: 共享状态字典，包含 gr.State 组件
	"""
	gr.Markdown("### Model & Data Management")

	with gr.Row():
		with gr.Column(scale=2):
			gr.Markdown("#### Load Model Checkpoint")
			results_dir = gr.Textbox(
				label="Results Directory",
				value="./results",
				placeholder="/path/to/results/stackelberg/",
			)
			scan_btn = gr.Button("Scan Checkpoints", variant="secondary")
			checkpoint_dropdown = gr.Dropdown(
				label="Select Checkpoint", choices=[], interactive=True,
			)
			load_btn = gr.Button("Load Model", variant="primary")
			load_status = gr.Markdown("No model loaded")

		with gr.Column(scale=2):
			gr.Markdown("#### Episode Data Files")
			episode_dir = gr.Textbox(
				label="Episode Directory",
				value="./results",
				placeholder="/path/to/episode/files/",
			)
			scan_ep_btn = gr.Button("Scan Episodes", variant="secondary")
			episode_dropdown = gr.Dropdown(
				label="Select Episode File", choices=[], interactive=True,
			)
			load_ep_btn = gr.Button("Load Episode", variant="primary")
			ep_status = gr.Markdown("No episode loaded")

	with gr.Row():
		gr.Markdown("#### Environment Configuration")
		with gr.Column():
			system_name = gr.Dropdown(
				label="IEEE System",
				choices=["13Bus", "34Bus", "123Bus"],
				value="13Bus",
			)
			n_consumers = gr.Slider(
				label="Number of Consumers",
				minimum=1, maximum=10, step=1, value=2,
			)
			episode_length = gr.Number(
				label="Episode Length",
				value=24,
			)

	config_summary = gr.JSON(label="Current Configuration", value={})

	# ------------------------------------------------------------------
	# 回调函数
	# ------------------------------------------------------------------

	def _scan_checkpoints(path: str):
		"""扫描检查点目录"""
		if not os.path.isdir(path):
			return gr.update(choices=[], value=None)

		patterns = ["**/*.pt", "**/*.pth", "**/*.ckpt"]
		found = []
		for pattern in patterns:
			found.extend(glob.glob(os.path.join(path, pattern), recursive=True))

		found.sort(key=os.path.getmtime, reverse=True)
		choices = [os.path.relpath(f, path) for f in found[:50]]

		if not choices:
			return gr.update(choices=["No checkpoints found"], value=None)

		return gr.update(choices=choices, value=choices[0])

	def _load_model(ckpt_path: str, base_dir: str):
		"""加载模型检查点"""
		if not ckpt_path or ckpt_path == "No checkpoints found":
			return "No valid checkpoint selected"

		full_path = os.path.join(base_dir, ckpt_path)
		if not os.path.isfile(full_path):
			return f"File not found: {full_path}"

		try:
			from envs.render_common.engine.inference_engine import InferenceEngine
			engine = InferenceEngine.from_checkpoint(full_path)
			return f"Model loaded: {ckpt_path}"
		except Exception as exc:
			logger.error(f"Failed to load model: {exc}")
			return f"Load failed: {exc}"

	def _scan_episodes(path: str):
		"""扫描 episode 文件"""
		if not os.path.isdir(path):
			return gr.update(choices=[], value=None)

		patterns = ["**/*.json", "**/*.npz"]
		found = []
		for pattern in patterns:
			found.extend(glob.glob(os.path.join(path, pattern), recursive=True))

		# 过滤可能的 episode 文件
		episode_files = [
			f for f in found
			if "episode" in os.path.basename(f).lower()
			or "snapshot" in os.path.basename(f).lower()
		]
		episode_files.sort(key=os.path.getmtime, reverse=True)
		choices = [os.path.relpath(f, path) for f in episode_files[:50]]

		if not choices:
			return gr.update(choices=["No episode files found"], value=None)

		return gr.update(choices=choices, value=choices[0])

	def _load_episode(ep_path: str, base_dir: str, snapshots_state):
		"""加载 episode 文件"""
		if not ep_path or ep_path == "No episode files found":
			return "No valid episode file selected", snapshots_state

		full_path = os.path.join(base_dir, ep_path)
		if not os.path.isfile(full_path):
			return f"File not found: {full_path}", snapshots_state

		try:
			if full_path.endswith(".json"):
				with open(full_path, "r") as f:
					data = json.load(f)
				if isinstance(data, list):
					snapshots = data
				elif isinstance(data, dict):
					snapshots = data.get("snapshots", [data])
				else:
					snapshots = [data]
			else:
				import numpy as np
				npz = np.load(full_path, allow_pickle=True)
				snapshots = list(npz.get("snapshots", []))

			return (
				f"Episode loaded: {len(snapshots)} snapshots from {ep_path}",
				snapshots,
			)
		except Exception as exc:
			logger.error(f"Failed to load episode: {exc}")
			return f"Load failed: {exc}", snapshots_state

	def _update_config(sys_name, n_cons, ep_len):
		"""更新配置摘要"""
		return {
			"env_name": "stackelberg",
			"system_name": sys_name,
			"n_consumers": int(n_cons),
			"n_agents": int(n_cons) + 1,
			"episode_length": int(ep_len),
			"uc_action_dim": 5,
			"consumer_action_dim": 3,
		}

	# 绑定事件
	scan_btn.click(_scan_checkpoints, inputs=[results_dir], outputs=[checkpoint_dropdown])
	load_btn.click(_load_model, inputs=[checkpoint_dropdown, results_dir], outputs=[load_status])
	scan_ep_btn.click(_scan_episodes, inputs=[episode_dir], outputs=[episode_dropdown])
	load_ep_btn.click(
		_load_episode,
		inputs=[episode_dropdown, episode_dir, shared_states["snapshots"]],
		outputs=[ep_status, shared_states["snapshots"]],
	)

	for component in [system_name, n_consumers, episode_length]:
		component.change(
			_update_config,
			inputs=[system_name, n_consumers, episode_length],
			outputs=[config_summary],
		)
