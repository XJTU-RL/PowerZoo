# -*- coding: utf-8 -*-
"""
Tab 7: Training - 训练监控

解析训练日志，展示奖励曲线、损失曲线、
训练指标时间线。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.utils.training_log_parser import (
	get_episode_metrics,
	get_reward_curves,
	parse_progress_file,
	scan_training_results,
)
from envs.render_common.viz.theme import COLORS, get_plotly_layout

logger = logging.getLogger(__name__)


def _scan_runs(results_dir: str) -> Tuple[List[List[str]], str]:
	"""扫描训练结果目录。"""
	if not results_dir:
		return [], "No directory"

	try:
		runs = scan_training_results(results_dir)
		if not runs:
			return [], f"No runs found in {results_dir}"

		rows: List[List[str]] = []
		for run in runs:
			rows.append([
				run.get("name", ""),
				run.get("algorithm", ""),
				str(run.get("n_episodes", 0)),
				run.get("status", ""),
				run.get("path", ""),
			])

		return rows, f"Found {len(runs)} training run(s)"
	except Exception as exc:
		logger.error(f"Scan failed: {exc}", exc_info=True)
		return [], f"Scan failed: {exc}"


def _load_training_data(
	run_path: str,
) -> Tuple[Optional[Dict], str]:
	"""加载训练日志数据。"""
	if not run_path:
		return None, "No run path"

	try:
		progress = parse_progress_file(run_path)
		if progress is None:
			return None, "No progress file found"
		return progress, f"Loaded training data from {run_path}"
	except Exception as exc:
		logger.error(f"Load failed: {exc}", exc_info=True)
		return None, f"Load failed: {exc}"


def _create_reward_curves(
	run_path: str,
) -> go.Figure:
	"""创建奖励曲线。"""
	layout = get_plotly_layout(title="Training Reward Curves")

	try:
		curves = get_reward_curves(run_path)
		if not curves:
			fig = go.Figure(layout=layout)
			fig.add_annotation(text="No reward data", showarrow=False)
			return fig

		fig = go.Figure(layout=layout)

		# 平均奖励
		if "mean_reward" in curves:
			data = curves["mean_reward"]
			fig.add_trace(go.Scatter(
				x=data.get("steps", []),
				y=data.get("values", []),
				mode="lines",
				name="Mean Reward",
				line=dict(color=COLORS["primary"], width=2),
			))

		# 最大/最小奖励
		if "max_reward" in curves:
			data = curves["max_reward"]
			fig.add_trace(go.Scatter(
				x=data.get("steps", []),
				y=data.get("values", []),
				mode="lines",
				name="Max Reward",
				line=dict(color=COLORS["success"], width=1, dash="dash"),
			))

		if "min_reward" in curves:
			data = curves["min_reward"]
			fig.add_trace(go.Scatter(
				x=data.get("steps", []),
				y=data.get("values", []),
				mode="lines",
				name="Min Reward",
				line=dict(color=COLORS["danger"], width=1, dash="dash"),
			))

		fig.update_layout(
			xaxis_title="Training Step",
			yaxis_title="Reward",
		)
		return fig

	except Exception as exc:
		logger.warning(f"Reward curve failed: {exc}")
		fig = go.Figure(layout=layout)
		fig.add_annotation(text=f"Error: {exc}", showarrow=False)
		return fig


def _create_metrics_chart(
	run_path: str,
) -> go.Figure:
	"""创建训练指标图。"""
	layout = get_plotly_layout(title="Training Metrics")

	try:
		metrics = get_episode_metrics(run_path)
		if not metrics:
			fig = go.Figure(layout=layout)
			fig.add_annotation(text="No metrics data", showarrow=False)
			return fig

		fig = make_subplots(
			rows=2, cols=1,
			subplot_titles=["Episode Length", "Value Loss"],
			shared_xaxes=True,
			vertical_spacing=0.12,
		)
		fig.update_layout(layout)

		# Episode length
		if "episode_length" in metrics:
			data = metrics["episode_length"]
			fig.add_trace(
				go.Scatter(
					x=data.get("steps", []),
					y=data.get("values", []),
					mode="lines",
					name="Ep Length",
					line=dict(color="#06B6D4", width=1.5),
				),
				row=1, col=1,
			)

		# Value loss
		if "value_loss" in metrics:
			data = metrics["value_loss"]
			fig.add_trace(
				go.Scatter(
					x=data.get("steps", []),
					y=data.get("values", []),
					mode="lines",
					name="Value Loss",
					line=dict(color="#F59E0B", width=1.5),
				),
				row=2, col=1,
			)

		fig.update_xaxes(title_text="Training Step", row=2, col=1)

		return fig

	except Exception as exc:
		logger.warning(f"Metrics chart failed: {exc}")
		fig = go.Figure(layout=layout)
		fig.add_annotation(text=f"Error: {exc}", showarrow=False)
		return fig


def _build_progress_table(
	progress: Optional[Dict],
) -> List[List[str]]:
	"""构建训练进度表。"""
	if progress is None:
		return []

	rows: List[List[str]] = []
	episodes = progress.get("episodes", [])

	for ep in episodes[-50:]:  # 最近 50 条
		rows.append([
			str(ep.get("episode", 0)),
			str(ep.get("step", 0)),
			f"{ep.get('reward', 0):.4f}",
			f"{ep.get('episode_length', 0)}",
			f"{ep.get('value_loss', 0):.6f}" if ep.get("value_loss") else "-",
			f"{ep.get('policy_loss', 0):.6f}" if ep.get("policy_loss") else "-",
		])

	return rows


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Training Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	# 本地状态
	state_progress = gr.State(None)
	state_run_path = gr.State("")

	# === 运行扫描 ===
	with gr.Group():
		gr.Markdown("### Training Runs")
		with gr.Row():
			results_dir_input = gr.Textbox(
				label="Results Directory",
				placeholder="e.g., results/",
				scale=4,
			)
			scan_btn = gr.Button("Scan", variant="primary", scale=1)
			load_btn = gr.Button("Load Selected", variant="secondary", scale=1)

	runs_table = gr.Dataframe(
		headers=["Name", "Algorithm", "Episodes", "Status", "Path"],
		datatype=["str"] * 5,
		interactive=False,
		wrap=True,
	)

	# === 训练曲线 ===
	with gr.Row():
		with gr.Column():
			gr.Markdown("### Reward Curves")
			reward_plot = gr.Plot(label="Reward")
		with gr.Column():
			gr.Markdown("### Training Metrics")
			metrics_plot = gr.Plot(label="Metrics")

	# === 训练进度表 ===
	with gr.Accordion("Training Progress (Recent 50)", open=False):
		progress_table = gr.Dataframe(
			headers=["Episode", "Step", "Reward", "Ep Length", "Value Loss", "Policy Loss"],
			datatype=["str"] * 6,
			interactive=False,
			wrap=True,
		)

	status_box = gr.Textbox(label="Status", interactive=False, lines=1)

	# --- 事件绑定 ---
	def on_scan(results_dir):
		rows, status = _scan_runs(results_dir)
		return rows, status

	def on_select_run(evt: gr.SelectData, table_data):
		if evt.index is not None and table_data:
			row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
			if row_idx < len(table_data):
				return table_data[row_idx][-1]
		return ""

	def on_load(run_path):
		if not run_path:
			return None, "", [], gr.Plot(), gr.Plot(), "No run selected"

		progress, status = _load_training_data(run_path)
		table_rows = _build_progress_table(progress)
		reward_fig = _create_reward_curves(run_path)
		metrics_fig = _create_metrics_chart(run_path)

		return progress, run_path, table_rows, reward_fig, metrics_fig, status

	scan_btn.click(
		fn=on_scan,
		inputs=[results_dir_input],
		outputs=[runs_table, status_box],
	)
	runs_table.select(
		fn=on_select_run,
		inputs=[runs_table],
		outputs=[state_run_path],
	)
	load_btn.click(
		fn=on_load,
		inputs=[state_run_path],
		outputs=[
			state_progress, state_run_path,
			progress_table, reward_plot, metrics_plot, status_box,
		],
	)

	return {
		"reward_plot": reward_plot,
		"metrics_plot": metrics_plot,
		"status_box": status_box,
	}
