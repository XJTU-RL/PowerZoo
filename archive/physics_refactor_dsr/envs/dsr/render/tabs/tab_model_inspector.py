# -*- coding: utf-8 -*-
"""
DSR Tab: Model Inspector
模型检查点管理标签页

提供模型检查点扫描、加载和模型信息查看功能。
DSR 模型需要支持异构 agent 的 obs/action 维度配置。
"""

import logging
import os
from typing import Any, Dict, Optional

import gradio as gr

from envs.render_common.engine.checkpoint_loader import CheckpointLoader
from envs.render_common.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Model Inspector 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Model Inspector"):

		gr.Markdown("### Model Checkpoint Inspector")
		gr.Markdown(
			"*DSR uses heterogeneous agents: "
			"Switch, PV, and Load agents may have different obs/action dims.*"
		)

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

		model_info_json = gr.JSON(label="Model Info")

		gr.Markdown("#### Agent Architecture")

		agent_info_table = gr.Dataframe(
			label="Agent Info",
			headers=["Agent ID", "Type", "Obs Dim", "Action Dim"],
			interactive=False,
		)

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
			return "Error: Run not found", None, []

		checkpoints = selected_run.get("checkpoints", [])
		selected_ckpt = next((c for c in checkpoints if c["name"] == ckpt_name), None)
		if selected_ckpt is None:
			return "Error: Checkpoint not found", None, []

		ckpt_path = selected_ckpt["path"]
		n_agents = selected_ckpt["n_agents"]

		# DSR 异构 agent: obs/action 维度可能不同
		# 从 checkpoint 元数据中获取，如果没有则使用默认值
		obs_dims = selected_ckpt.get("obs_dims", [32] * n_agents)
		action_dims = selected_ckpt.get("action_dims", [5] * n_agents)
		agent_types = selected_ckpt.get("agent_types", ["unknown"] * n_agents)

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

			# 构建 agent 信息表
			agent_table = []
			for i in range(n_agents):
				agent_table.append([
					str(i),
					agent_types[i] if i < len(agent_types) else "unknown",
					str(obs_dims[i]) if i < len(obs_dims) else "?",
					str(action_dims[i]) if i < len(action_dims) else "?",
				])

			return (
				f"Loaded: {ckpt_name} ({n_agents} agents)",
				info,
				agent_table,
			)

		except Exception as e:
			logger.error(f"Model loading failed: {e}")
			return f"Error: {e}", None, []

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
		outputs=[model_status, model_info_json, agent_info_table],
	)

	components["results_dir_input"] = results_dir_input
	components["run_dropdown"] = run_dropdown
	components["checkpoint_dropdown"] = checkpoint_dropdown
	components["model_status"] = model_status
	components["model_info_json"] = model_info_json

	return components
