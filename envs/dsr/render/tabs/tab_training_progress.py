# -*- coding: utf-8 -*-
"""
DSR Tab: Training Progress
训练进度标签页

解析训练日志和 progress.txt，展示奖励曲线、
训练指标趋势等信息。
"""

import logging
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.utils.training_log_parser import (
	scan_training_results,
	parse_progress_file,
	get_reward_curves,
	get_episode_metrics,
	parse_training_log,
)
from envs.render_common.viz.theme import get_plotly_layout, COLORS

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Training Progress 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Training"):

		gr.Markdown("### Training Progress Viewer")

		with gr.Row():
			results_dir_input = gr.Textbox(
				label="Results Directory",
				value="results",
				placeholder="Path to training results",
			)
			scan_btn = gr.Button("Scan", variant="secondary")

		run_dropdown = gr.Dropdown(
			label="Training Run",
			choices=[],
			interactive=True,
		)

		with gr.Row():
			load_btn = gr.Button("Load Progress", variant="primary")
			run_info = gr.Textbox(
				label="Run Info",
				interactive=False,
				value="No run loaded",
			)

		reward_curve_plot = gr.Plot(label="Episode Reward Curve")

		with gr.Row():
			metrics_plot = gr.Plot(label="Training Metrics")
			config_json = gr.JSON(label="Training Configuration")

	# --- 回调 ---

	def _scan_runs(results_dir: str):
		"""扫描训练结果"""
		runs = scan_training_results(results_dir)
		shared_states["training_progress_runs"] = runs

		if not runs:
			return gr.update(choices=[], value=None), "No runs found"

		choices = [r["name"] for r in runs]
		return gr.update(choices=choices, value=choices[0]), f"Found {len(runs)} runs"

	def _load_progress(run_name: str, results_dir: str):
		"""加载训练进度"""
		runs = shared_states.get("training_progress_runs", [])
		selected = next((r for r in runs if r["name"] == run_name), None)
		if selected is None:
			return "Run not found", None, None, None

		run_path = selected["path"]

		# 奖励曲线
		reward_df = get_reward_curves(run_path)
		reward_fig = _build_reward_curve(reward_df)

		# 训练指标
		metrics = get_episode_metrics(run_path)
		metrics_fig = _build_metrics_plot(metrics)

		# 配置信息
		log_path = os.path.join(run_path, "training.log")
		parsed = parse_training_log(log_path)
		config_info = parsed.get("config", {})
		config_info["final_reward"] = parsed.get("final_reward")
		config_info["total_episodes"] = parsed.get("total_episodes")
		config_info["training_time"] = parsed.get("training_time")

		info_text = (
			f"Algorithm: {selected.get('algorithm', 'unknown')}, "
			f"Env: {selected.get('env', 'unknown')}, "
			f"Episodes: {selected.get('episodes', 'N/A')}"
		)

		return info_text, reward_fig, metrics_fig, config_info

	# --- 绑定 ---

	scan_btn.click(
		fn=_scan_runs,
		inputs=[results_dir_input],
		outputs=[run_dropdown, run_info],
	)

	load_btn.click(
		fn=_load_progress,
		inputs=[run_dropdown, results_dir_input],
		outputs=[run_info, reward_curve_plot, metrics_plot, config_json],
	)

	components["reward_curve_plot"] = reward_curve_plot
	components["metrics_plot"] = metrics_plot

	return components


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _build_reward_curve(reward_df) -> Optional[go.Figure]:
	"""构建奖励曲线图"""
	if reward_df is None or reward_df.empty:
		return None

	layout = get_plotly_layout(
		title="Episode Reward Curve",
		height=450,
		env_name="dsr",
	)
	fig = go.Figure(layout=layout)

	colors = [COLORS["primary"], COLORS["success"], COLORS["warning"]]

	for idx, col in enumerate(reward_df.columns):
		if "reward" in col.lower():
			color = colors[idx % len(colors)]
			fig.add_trace(go.Scatter(
				x=list(range(len(reward_df))),
				y=reward_df[col].tolist(),
				mode="lines",
				name=col,
				line={"color": color, "width": 1.5},
			))

	fig.update_xaxes(title_text="Episode")
	fig.update_yaxes(title_text="Reward")

	return fig


def _build_metrics_plot(metrics: Dict[str, List[float]]) -> Optional[go.Figure]:
	"""构建训练指标图"""
	if not metrics:
		return None

	# 选择关键指标
	key_metrics = []
	for name in metrics:
		lower = name.lower()
		if any(k in lower for k in ["loss", "entropy", "kl", "ratio", "value"]):
			key_metrics.append(name)

	if not key_metrics:
		key_metrics = list(metrics.keys())[:4]

	n_plots = min(len(key_metrics), 4)
	if n_plots == 0:
		return None

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=key_metrics[:4],
	)

	colors = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["info"]]

	for idx, name in enumerate(key_metrics[:4]):
		row = idx // 2 + 1
		col = idx % 2 + 1
		values = metrics[name]
		fig.add_trace(
			go.Scatter(
				x=list(range(len(values))),
				y=values,
				mode="lines",
				name=name,
				line={"color": colors[idx], "width": 1},
				showlegend=False,
			),
			row=row, col=col,
		)

	layout = get_plotly_layout(
		title="Training Metrics", height=500, env_name="dsr"
	)
	layout.pop("xaxis", None)
	layout.pop("yaxis", None)
	fig.update_layout(**layout)

	return fig
