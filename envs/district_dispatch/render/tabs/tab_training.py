# -*- coding: utf-8 -*-
"""
Tab 7: Training Progress
训练进度查看标签页 -- 扫描训练结果目录，展示训练曲线和汇总指标

功能:
- 扫描 results/ 目录发现所有训练运行
- 以 Dataframe 展示运行列表
- 选择运行后展示 2x2 训练曲线 (Plotly)
- Markdown 训练摘要
"""

import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.district_dispatch.render.utils.training_log_parser import (
	get_episode_metrics,
	get_reward_curves,
	parse_progress_file,
	parse_training_log,
	scan_training_results,
)

logger = logging.getLogger(__name__)

# 默认结果目录
_DEFAULT_RESULTS_DIR = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"..", "..", "..", "..", "results",
)

# 平滑窗口大小
_SMOOTH_WINDOW = 10


def _smooth(values: List[float], window: int = _SMOOTH_WINDOW) -> List[float]:
	"""简单移动平均平滑。"""
	if len(values) < window:
		return values
	smoothed = []
	for i in range(len(values)):
		start = max(0, i - window + 1)
		smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
	return smoothed


def _scan_results(dir_path: str) -> Tuple[pd.DataFrame, str, List[Dict[str, Any]]]:
	"""扫描训练结果目录。

	Returns:
		(dataframe_for_display, status_msg, raw_runs_list)
	"""
	dir_path = dir_path.strip()
	if not dir_path:
		dir_path = os.path.abspath(_DEFAULT_RESULTS_DIR)

	if not os.path.isdir(dir_path):
		return pd.DataFrame(), f"Directory not found: {dir_path}", []

	runs = scan_training_results(dir_path)
	if not runs:
		return pd.DataFrame(), f"No training runs found in {dir_path}", []

	# 构建展示用 DataFrame
	rows = []
	for r in runs:
		rows.append({
			"Run Name": r["name"],
			"Algorithm": r["algorithm"],
			"Env": r["env"],
			"Episodes": r.get("episodes") or "N/A",
			"Has Progress": "Yes" if r["has_progress"] else "No",
			"Timestamp": r.get("timestamp", ""),
		})

	df = pd.DataFrame(rows)
	msg = f"Found {len(runs)} training run(s) in {dir_path}"
	return df, msg, runs


def _build_training_curves(run_path: str) -> Tuple[Optional[go.Figure], str]:
	"""为指定运行生成 2x2 训练曲线和摘要。

	Returns:
		(plotly_figure, summary_markdown)
	"""
	if not run_path or not os.path.isdir(run_path):
		return None, "*Select a training run to view curves.*"

	# 解析 progress 数据
	progress_path = os.path.join(run_path, "progress.txt")
	df = parse_progress_file(progress_path)

	if df.empty:
		return None, f"No progress.txt found or empty in {run_path}"

	# 解析 training.log
	log_info = parse_training_log(os.path.join(run_path, "training.log"))

	# 获取 episode metrics
	metrics = get_episode_metrics(run_path)

	# 构建 2x2 子图
	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"Episode Reward",
			"Eval Reward",
			"Actor/Policy Loss",
			"Critic/Value Loss",
		),
		vertical_spacing=0.12,
		horizontal_spacing=0.10,
	)

	# x 轴: episode index
	_dark = dict(
		template="plotly_dark",
		paper_bgcolor="#0f0f23",
		plot_bgcolor="#0f0f23",
	)

	# 1. Episode Reward (原始 + 平滑)
	reward_col = _find_col(df, ["average_episode_rewards", "episode_reward", "reward"])
	if reward_col:
		vals = df[reward_col].dropna().tolist()
		x = list(range(len(vals)))
		fig.add_trace(
			go.Scatter(x=x, y=vals, mode="lines", name="Raw Reward",
					   line=dict(color="rgba(99,102,241,0.3)", width=1)),
			row=1, col=1,
		)
		smoothed = _smooth(vals)
		fig.add_trace(
			go.Scatter(x=x, y=smoothed, mode="lines", name="Smoothed Reward",
					   line=dict(color="#6366F1", width=2)),
			row=1, col=1,
		)

	# 2. Eval Reward
	eval_col = _find_col(df, ["eval_average_episode_rewards", "eval_reward"])
	if eval_col:
		vals = df[eval_col].dropna().tolist()
		x = list(range(len(vals)))
		fig.add_trace(
			go.Scatter(x=x, y=vals, mode="lines+markers", name="Eval Reward",
					   line=dict(color="#10B981", width=2),
					   marker=dict(size=4)),
			row=1, col=2,
		)

	# 3. Actor/Policy Loss
	actor_col = _find_col(df, ["policy_loss", "actor_loss", "pg_loss"])
	if actor_col:
		vals = df[actor_col].dropna().tolist()
		x = list(range(len(vals)))
		fig.add_trace(
			go.Scatter(x=x, y=vals, mode="lines", name="Policy Loss",
					   line=dict(color="#F59E0B", width=2)),
			row=2, col=1,
		)

	# 4. Critic/Value Loss
	critic_col = _find_col(df, ["value_loss", "critic_loss", "vf_loss"])
	if critic_col:
		vals = df[critic_col].dropna().tolist()
		x = list(range(len(vals)))
		fig.add_trace(
			go.Scatter(x=x, y=vals, mode="lines", name="Value Loss",
					   line=dict(color="#EF4444", width=2)),
			row=2, col=2,
		)

	fig.update_layout(
		height=600,
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
		margin=dict(l=60, r=30, t=60, b=60),
		**_dark,
	)

	# 构建摘要 Markdown
	summary = _build_summary_md(df, log_info, run_path, reward_col)

	return fig, summary


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
	"""在 DataFrame 中查找匹配的列名 (大小写模糊匹配)。"""
	df_cols_lower = {c.lower(): c for c in df.columns}
	for cand in candidates:
		if cand.lower() in df_cols_lower:
			return df_cols_lower[cand.lower()]
	return None


def _build_summary_md(
	df: pd.DataFrame,
	log_info: Dict[str, Any],
	run_path: str,
	reward_col: Optional[str],
) -> str:
	"""构建训练摘要 Markdown。"""
	lines = ["### Training Summary", ""]

	run_name = os.path.basename(run_path)
	lines.append(f"**Run**: `{run_name}`")

	config = log_info.get("config", {})
	algo = config.get("algorithm", config.get("algo", "N/A"))
	lines.append(f"**Algorithm**: {algo}")

	total_eps = log_info.get("total_episodes") or len(df)
	lines.append(f"**Total Episodes**: {total_eps}")

	if reward_col and reward_col in df.columns:
		best = df[reward_col].max()
		last = df[reward_col].iloc[-1] if len(df) > 0 else None
		lines.append(f"**Best Reward**: {best:.4f}")
		if last is not None:
			lines.append(f"**Last Reward**: {last:.4f}")

	if log_info.get("training_time"):
		lines.append(f"**Training Time**: {log_info['training_time']}")

	if log_info.get("total_steps"):
		lines.append(f"**Total Steps**: {log_info['total_steps']:,}")

	# 配置参数
	if config:
		lines.append("")
		lines.append("**Config**:")
		for k, v in config.items():
			if k not in ("algorithm", "algo"):
				lines.append(f"- {k}: {v}")

	return "\n".join(lines)


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Training Progress 标签页。

	Args:
		shared_states: 共享状态字典

	Returns:
		该 Tab 内关键组件引用
	"""
	components: Dict[str, Any] = {}

	# 内部 state: 存储扫描结果列表
	runs_state = gr.State([])

	with gr.Tab("Training Progress"):
		gr.Markdown("## Training Progress")
		gr.Markdown(
			"Scan a training results directory, select a run, "
			"and view training curves."
		)

		# -- 扫描目录 --
		with gr.Row():
			dir_input = gr.Textbox(
				label="Results Directory",
				value=os.path.abspath(_DEFAULT_RESULTS_DIR),
				scale=4,
			)
			btn_scan = gr.Button("Scan", variant="primary", scale=1)

		scan_status = gr.Textbox(label="Status", interactive=False, lines=1)

		# -- 运行列表 --
		runs_table = gr.Dataframe(
			label="Training Runs",
			interactive=False,
			wrap=True,
		)

		# -- 选择运行 --
		with gr.Row():
			run_selector = gr.Dropdown(
				choices=[],
				label="Select Run",
				interactive=True,
				scale=4,
			)
			btn_show = gr.Button("Show Curves", variant="secondary", scale=1)

		# -- 训练曲线 --
		curves_plot = gr.Plot(label="Training Curves (2x2)")

		# -- 训练摘要 --
		summary_md = gr.Markdown("*Scan a directory and select a run to view details.*")

		components.update({
			"dir_input": dir_input,
			"btn_scan": btn_scan,
			"scan_status": scan_status,
			"runs_table": runs_table,
			"run_selector": run_selector,
			"btn_show": btn_show,
			"curves_plot": curves_plot,
			"summary_md": summary_md,
			"runs_state": runs_state,
		})

		# -- 事件: 扫描目录 --
		def on_scan(dir_path: str):
			"""扫描目录并更新列表和下拉。"""
			df, msg, runs = _scan_results(dir_path)
			# 更新下拉选项
			choices = []
			for r in runs:
				choices.append(r["name"])
			selector_update = gr.update(choices=choices, value=choices[0] if choices else None)
			return df, msg, runs, selector_update

		btn_scan.click(
			fn=on_scan,
			inputs=[dir_input],
			outputs=[runs_table, scan_status, runs_state, run_selector],
		)

		# -- 事件: 显示曲线 --
		def on_show_curves(run_name: str, runs: List[Dict[str, Any]]):
			"""根据选中的运行名展示训练曲线。"""
			if not run_name or not runs:
				return None, "*No run selected.*"

			# 查找 run_path
			run_path = None
			for r in runs:
				if r["name"] == run_name:
					run_path = r["path"]
					break

			if not run_path:
				return None, f"*Run `{run_name}` not found in scan results.*"

			try:
				fig, summary = _build_training_curves(run_path)
				return fig, summary
			except Exception as e:
				logger.error(f"Show curves failed: {e}\n{traceback.format_exc()}")
				return None, f"**Error**: {e}"

		btn_show.click(
			fn=on_show_curves,
			inputs=[run_selector, runs_state],
			outputs=[curves_plot, summary_md],
		)

	return components
