# -*- coding: utf-8 -*-
"""
Tab 1: Model & Data - 模型加载和数据管理

提供 checkpoint 目录扫描、模型加载、模型详情展示、
已录制 episode 列表浏览、母线坐标加载等功能。
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from envs.render_common.engine.checkpoint_loader import CheckpointLoader
from envs.render_common.engine.episode_recorder import EpisodeRecorder
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.vvc.render.assets.bus_coordinates import get_available_systems, load_bus_coordinates

logger = logging.getLogger(__name__)


def _scan_checkpoints(results_dir: str) -> Tuple[List[List[str]], str]:
	"""扫描训练运行目录中的检查点。"""
	if not results_dir or not os.path.isdir(results_dir):
		return [], f"Directory not found: {results_dir}"

	runs = CheckpointLoader.scan_training_runs(results_dir)
	if not runs:
		return [], f"No training runs found in: {results_dir}"

	rows: List[List[str]] = []
	for run in runs:
		n_ckpts = len(run["checkpoints"])
		best_ep = "-"
		best_reward = "-"
		if run["checkpoints"]:
			for ckpt in run["checkpoints"]:
				ts = ckpt.get("training_state", {})
				if ts.get("best_reward") is not None:
					br = ts["best_reward"]
					if br != float("-inf"):
						best_reward = f"{br:.2f}"
						best_ep = str(ckpt["episode"])
						break
			if best_ep == "-" and run["checkpoints"]:
				best_ep = str(run["checkpoints"][0]["episode"])

		rows.append([
			run["name"], run["algorithm"],
			str(n_ckpts), best_ep, best_reward, run["path"],
		])

	status = f"Found {len(runs)} run(s) with {sum(len(r['checkpoints']) for r in runs)} checkpoint(s)"
	return rows, status


def _load_model(
	run_path: str,
	model_loaded: bool,
	inference_engine: Optional[InferenceEngine],
) -> Tuple[Optional[InferenceEngine], bool, str, str]:
	"""加载指定 run 的最佳模型。"""
	if not run_path or not os.path.isdir(run_path):
		return inference_engine, model_loaded, "", f"Invalid run path: {run_path}"

	models_dir = os.path.join(run_path, "models")
	if not os.path.isdir(models_dir):
		models_dir = run_path

	best = CheckpointLoader.get_best_checkpoint(models_dir)
	if best is None:
		return inference_engine, model_loaded, "", "No valid checkpoint found"

	n_agents = best["n_agents"]
	checkpoint_dir = best["path"]

	# VVC 默认维度 (将从实际环境中获取)
	obs_dims = [64] * n_agents
	action_dims = [1] * n_agents

	try:
		engine = InferenceEngine(
			checkpoint_dir=checkpoint_dir,
			n_agents=n_agents,
			obs_dims=obs_dims,
			action_dims=action_dims,
		)
		engine.load_actors()

		info = engine.get_model_info()
		md_lines = [
			"### Model Info",
			f"- **Checkpoint**: `{best['name']}`",
			f"- **Agents**: {info['n_agents']}",
			f"- **Device**: {info['device']}",
			f"- **RNN**: {'Yes' if info['use_rnn'] else 'No'}",
			"",
			"| Agent | Obs Dim | Act Dim | Params |",
			"|-------|---------|---------|--------|",
		]
		for ai in info.get("agents", []):
			md_lines.append(
				f"| {ai['agent_id']} | {ai['obs_dim']} "
				f"| {ai['action_dim']} | {ai['trainable_params']:,} |"
			)

		model_md = "\n".join(md_lines)
		status = f"Model loaded: {best['name']} ({n_agents} agents)"
		return engine, True, model_md, status

	except Exception as exc:
		logger.error(f"Model loading failed: {exc}", exc_info=True)
		return inference_engine, model_loaded, "", f"Load failed: {exc}"


def _scan_episodes(episodes_dir: str) -> Tuple[List[List[str]], str]:
	"""扫描已录制的 episode 文件。"""
	if not episodes_dir or not os.path.isdir(episodes_dir):
		return [], f"Directory not found: {episodes_dir}"

	recordings = EpisodeRecorder.list_recordings(episodes_dir)
	if not recordings:
		return [], "No recorded episodes found"

	rows: List[List[str]] = []
	for rec in recordings:
		rows.append([
			rec["filename"],
			str(rec["n_steps"]),
			f"{rec['total_reward']:.2f}" if rec["total_reward"] == rec["total_reward"] else "N/A",
			rec.get("algorithm", "unknown"),
			str(rec.get("seed", "-")),
			rec.get("timestamp", "-"),
			rec["path"],
		])

	return rows, f"Found {len(recordings)} recorded episode(s)"


def _load_bus_coords(system_name: str) -> Tuple[Dict, str]:
	"""加载指定系统的母线坐标。"""
	try:
		coords = load_bus_coordinates(system_name)
		return coords, f"Loaded {len(coords)} bus coordinates for {system_name}"
	except Exception as exc:
		logger.warning(f"Bus coordinate loading failed: {exc}")
		return {}, f"Failed: {exc}"


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Model & Data Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	state_model_loaded = shared_states["model_loaded"]
	state_inference_engine = shared_states["inference_engine"]
	state_bus_coords = shared_states["bus_coords"]

	# === Checkpoint 扫描 ===
	with gr.Group():
		gr.Markdown("### Checkpoint Scanner")
		with gr.Row():
			ckpt_dir_input = gr.Textbox(
				label="Results Directory",
				placeholder="e.g., results/",
				scale=4,
			)
			scan_btn = gr.Button("Scan", variant="primary", scale=1)
			load_btn = gr.Button("Load Model", variant="secondary", scale=1)

	with gr.Row():
		with gr.Column(scale=3):
			gr.Markdown("### Training Runs")
			ckpt_table = gr.Dataframe(
				headers=["Run Name", "Algorithm", "Checkpoints", "Best Episode", "Best Reward", "Path"],
				datatype=["str"] * 6,
				interactive=False,
				wrap=True,
			)
		with gr.Column(scale=2):
			gr.Markdown("### Model Details")
			model_info_md = gr.Markdown("*No model loaded*")

	# === Episode 列表 ===
	with gr.Group():
		gr.Markdown("### Recorded Episodes")
		with gr.Row():
			episodes_dir_input = gr.Textbox(
				label="Episodes Directory",
				placeholder="e.g., recorded_episodes/",
				scale=3,
			)
			scan_ep_btn = gr.Button("Scan Episodes", scale=1)

		with gr.Row():
			system_dropdown = gr.Dropdown(
				choices=get_available_systems(),
				value="13Bus",
				label="Bus System",
				scale=2,
			)
			load_coords_btn = gr.Button("Load Bus Coords", scale=1)

		episodes_table = gr.Dataframe(
			headers=["Filename", "Steps", "Total Reward", "Algorithm", "Seed", "Timestamp", "Path"],
			datatype=["str"] * 7,
			interactive=False,
			wrap=True,
		)

	status_box = gr.Textbox(label="Status", interactive=False, lines=1)
	selected_run_path = gr.State("")

	# --- 事件绑定 ---
	def on_scan(results_dir):
		rows, status = _scan_checkpoints(results_dir)
		return rows, status

	def on_select_run(evt: gr.SelectData, table_data):
		if evt.index is not None and table_data:
			row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
			if row_idx < len(table_data):
				return table_data[row_idx][-1]
		return ""

	def on_load_model(run_path, model_loaded_val, engine_val):
		new_engine, loaded, model_md, status = _load_model(
			run_path, model_loaded_val, engine_val
		)
		return new_engine, loaded, model_md, status

	def on_scan_episodes(episodes_dir):
		rows, status = _scan_episodes(episodes_dir)
		return rows, status

	def on_load_coords(system_name):
		coords, status = _load_bus_coords(system_name)
		return coords, status

	scan_btn.click(fn=on_scan, inputs=[ckpt_dir_input], outputs=[ckpt_table, status_box])
	ckpt_table.select(fn=on_select_run, inputs=[ckpt_table], outputs=[selected_run_path])
	load_btn.click(
		fn=on_load_model,
		inputs=[selected_run_path, state_model_loaded, state_inference_engine],
		outputs=[state_inference_engine, state_model_loaded, model_info_md, status_box],
	)
	scan_ep_btn.click(fn=on_scan_episodes, inputs=[episodes_dir_input], outputs=[episodes_table, status_box])
	load_coords_btn.click(fn=on_load_coords, inputs=[system_dropdown], outputs=[state_bus_coords, status_box])

	return {
		"ckpt_dir_input": ckpt_dir_input,
		"ckpt_table": ckpt_table,
		"model_info_md": model_info_md,
		"episodes_table": episodes_table,
		"status_box": status_box,
		"load_btn": load_btn,
	}
