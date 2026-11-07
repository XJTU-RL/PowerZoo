# -*- coding: utf-8 -*-
"""
Tab 1: Model & Data - 模型加载和数据管理

提供 checkpoint 目录扫描、模型加载、模型详情展示、
已录制 episode 列表浏览等功能。
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from envs.district_dispatch.render.assets.bus_coordinates import load_bus_coordinates
from envs.district_dispatch.render.engine.checkpoint_loader import CheckpointLoader
from envs.district_dispatch.render.engine.episode_recorder import EpisodeRecorder
from envs.district_dispatch.render.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------

def _scan_checkpoints(results_dir: str) -> Tuple[List[List[str]], str]:
	"""扫描训练运行目录中的检查点。

	Args:
		results_dir: 训练结果根目录

	Returns:
		(dataframe_rows, status_message)
		dataframe_rows: [[run_name, algorithm, n_checkpoints, best_episode], ...]
	"""
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
			# 找最佳 checkpoint
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
			run["name"],
			run["algorithm"],
			str(n_ckpts),
			best_ep,
			best_reward,
			run["path"],
		])

	status = f"Found {len(runs)} training run(s) with {sum(len(r['checkpoints']) for r in runs)} checkpoint(s)"
	return rows, status


def _load_model(
	run_path: str,
	model_loaded: bool,
	inference_engine: Optional[InferenceEngine],
) -> Tuple[Optional[InferenceEngine], bool, str, str]:
	"""加载指定 run 的最佳模型。

	Args:
		run_path: 训练运行目录绝对路径
		model_loaded: 当前模型加载状态
		inference_engine: 当前推理引擎实例

	Returns:
		(new_engine, loaded_flag, model_info_markdown, status_message)
	"""
	if not run_path or not os.path.isdir(run_path):
		return inference_engine, model_loaded, "", f"Invalid run path: {run_path}"

	# 查找 models/ 子目录
	models_dir = os.path.join(run_path, "models")
	if not os.path.isdir(models_dir):
		models_dir = run_path

	best = CheckpointLoader.get_best_checkpoint(models_dir)
	if best is None:
		return inference_engine, model_loaded, "", "No valid checkpoint found"

	n_agents = best["n_agents"]
	checkpoint_dir = best["path"]

	# NOTE: 使用默认的 obs_dim 和 action_dim，实际项目应从 config 中读取
	# 3-zone district dispatch 每个 agent 的维度相同
	# 这里先用占位值，后续可通过 config 动态获取
	obs_dims = [64] * n_agents
	action_dims = [5] * n_agents

	try:
		engine = InferenceEngine(
			checkpoint_dir=checkpoint_dir,
			n_agents=n_agents,
			obs_dims=obs_dims,
			action_dims=action_dims,
		)
		engine.load_actors()

		# 构建模型详情 Markdown
		info = engine.get_model_info()
		md_lines = [
			f"### Model Info",
			f"- **Checkpoint**: `{best['name']}`",
			f"- **Agents**: {info['n_agents']}",
			f"- **Device**: {info['device']}",
			f"- **RNN**: {'Yes' if info['use_rnn'] else 'No'}",
			f"- **Hidden Size**: {info['hidden_size']}",
			"",
			"| Agent | Obs Dim | Act Dim | Params |",
			"|-------|---------|---------|--------|",
		]
		for agent_info in info.get("agents", []):
			md_lines.append(
				f"| {agent_info['agent_id']} "
				f"| {agent_info['obs_dim']} "
				f"| {agent_info['action_dim']} "
				f"| {agent_info['trainable_params']:,} |"
			)

		model_md = "\n".join(md_lines)
		status = f"Model loaded: {best['name']} ({n_agents} agents)"
		return engine, True, model_md, status

	except Exception as exc:
		logger.error(f"Model loading failed: {exc}", exc_info=True)
		return inference_engine, model_loaded, "", f"Load failed: {exc}"


def _scan_episodes(episodes_dir: str) -> Tuple[List[List[str]], str]:
	"""扫描已录制的 episode 文件。

	Args:
		episodes_dir: 录制文件目录

	Returns:
		(dataframe_rows, status_message)
	"""
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

	status = f"Found {len(recordings)} recorded episode(s)"
	return rows, status


def _load_bus_coords() -> Tuple[Dict, str]:
	"""加载 IEEE 34-bus 母线坐标。

	Returns:
		(bus_coords_dict, status_message)
	"""
	try:
		coords = load_bus_coordinates()
		return coords, f"Loaded {len(coords)} bus coordinates"
	except Exception as exc:
		logger.warning(f"Bus coordinate loading failed: {exc}")
		return {}, f"Failed to load bus coordinates: {exc}"


# ------------------------------------------------------------------
# Tab 创建入口
# ------------------------------------------------------------------

def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Model & Data Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典，包含:
			- snapshots: 快照列表
			- bus_coords: 母线坐标
			- model_loaded: 模型加载标志
			- inference_engine: 推理引擎实例
			- episode_runner: EpisodeRunner 实例

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	state_model_loaded = shared_states["model_loaded"]
	state_inference_engine = shared_states["inference_engine"]
	state_bus_coords = shared_states["bus_coords"]

	# === 上方: Checkpoint 扫描 ===
	with gr.Group():
		gr.Markdown("### Checkpoint Scanner")
		with gr.Row():
			ckpt_dir_input = gr.Textbox(
				label="Results Directory",
				placeholder="e.g., results/ or /path/to/training/results",
				scale=4,
			)
			scan_btn = gr.Button("Scan", variant="primary", scale=1)
			load_btn = gr.Button("Load Model", variant="secondary", scale=1)

	# === 中部: 左右分栏 ===
	with gr.Row():
		# 左: Checkpoint 列表
		with gr.Column(scale=3):
			gr.Markdown("### Training Runs")
			ckpt_table = gr.Dataframe(
				headers=["Run Name", "Algorithm", "Checkpoints", "Best Episode", "Best Reward", "Path"],
				datatype=["str", "str", "str", "str", "str", "str"],
				interactive=False,
				wrap=True,
			)
		# 右: 模型详情
		with gr.Column(scale=2):
			gr.Markdown("### Model Details")
			model_info_md = gr.Markdown("*No model loaded*")

	# === 下方: 录制 Episode 列表 ===
	with gr.Group():
		gr.Markdown("### Recorded Episodes")
		with gr.Row():
			episodes_dir_input = gr.Textbox(
				label="Episodes Directory",
				placeholder="e.g., recorded_episodes/",
				scale=4,
			)
			scan_ep_btn = gr.Button("Scan Episodes", scale=1)
			load_coords_btn = gr.Button("Load Bus Coords", scale=1)

		episodes_table = gr.Dataframe(
			headers=["Filename", "Steps", "Total Reward", "Algorithm", "Seed", "Timestamp", "Path"],
			datatype=["str", "str", "str", "str", "str", "str", "str"],
			interactive=False,
			wrap=True,
		)

	# === 状态栏 ===
	status_box = gr.Textbox(
		label="Status",
		interactive=False,
		lines=1,
	)

	# === 隐藏: 选中的 run_path ===
	selected_run_path = gr.State("")

	# ------------------------------------------------------------------
	# 事件绑定
	# ------------------------------------------------------------------

	def on_scan(results_dir):
		"""扫描按钮回调"""
		rows, status = _scan_checkpoints(results_dir)
		return rows, status

	def on_select_run(evt: gr.SelectData, table_data):
		"""表格行选择回调，提取 run path"""
		if evt.index is not None and table_data:
			row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
			if row_idx < len(table_data):
				# Path 在最后一列
				path = table_data[row_idx][-1]
				return path
		return ""

	def on_load_model(run_path, model_loaded_val, engine_val):
		"""加载模型按钮回调"""
		new_engine, loaded, model_md, status = _load_model(
			run_path, model_loaded_val, engine_val
		)
		return new_engine, loaded, model_md, status

	def on_scan_episodes(episodes_dir):
		"""扫描 episode 按钮回调"""
		rows, status = _scan_episodes(episodes_dir)
		return rows, status

	def on_load_coords():
		"""加载母线坐标按钮回调"""
		coords, status = _load_bus_coords()
		return coords, status

	# 绑定: 扫描 checkpoints
	scan_btn.click(
		fn=on_scan,
		inputs=[ckpt_dir_input],
		outputs=[ckpt_table, status_box],
	)

	# 绑定: 选择 run
	ckpt_table.select(
		fn=on_select_run,
		inputs=[ckpt_table],
		outputs=[selected_run_path],
	)

	# 绑定: 加载模型
	load_btn.click(
		fn=on_load_model,
		inputs=[selected_run_path, state_model_loaded, state_inference_engine],
		outputs=[state_inference_engine, state_model_loaded, model_info_md, status_box],
	)

	# 绑定: 扫描 episodes
	scan_ep_btn.click(
		fn=on_scan_episodes,
		inputs=[episodes_dir_input],
		outputs=[episodes_table, status_box],
	)

	# 绑定: 加载母线坐标
	load_coords_btn.click(
		fn=on_load_coords,
		inputs=[],
		outputs=[state_bus_coords, status_box],
	)

	return {
		"ckpt_dir_input": ckpt_dir_input,
		"ckpt_table": ckpt_table,
		"model_info_md": model_info_md,
		"episodes_table": episodes_table,
		"status_box": status_box,
		"load_btn": load_btn,
	}
