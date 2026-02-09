# -*- coding: utf-8 -*-
"""
SmartGrid Tab: Stress Test
压力测试标签页

支持参数扫描 (load_mult, pv_ratio, lambda_init) 和灵敏度分析，
展示 SmartGrid 在不同条件下的表现。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.render_common.viz.theme import get_plotly_layout, COLORS
from envs.smartgrid.render.engine.stress_test_runner import SmartGridStressTestRunner

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Stress Test 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Stress Test"):

		gr.Markdown("### SmartGrid Stress Test & Sensitivity Analysis")
		gr.Markdown(
			"Sweep parameters: `load_mult` (0.5~2.0), "
			"`pv_ratio` (0.0~2.0), `lambda_init` (0.01~10.0)"
		)

		with gr.Row():
			param_dropdown = gr.Dropdown(
				label="Sweep Parameter",
				choices=["load_mult", "pv_ratio", "lambda_init"],
				value="load_mult",
			)
			sweep_min = gr.Number(label="Min", value=0.5)
			sweep_max = gr.Number(label="Max", value=2.0)
			sweep_steps = gr.Number(label="Steps", value=5, precision=0)

		with gr.Row():
			system_dropdown = gr.Dropdown(
				label="System",
				choices=["13Bus", "34Bus_PV", "123Bus"],
				value="34Bus_PV",
			)
			seed_input = gr.Number(label="Seed", value=42, precision=0)
			max_steps_input = gr.Number(label="Max Steps", value=24, precision=0)

		with gr.Row():
			run_sweep_btn = gr.Button("Run Sweep", variant="primary")
			run_sensitivity_btn = gr.Button("Run Sensitivity", variant="secondary")

		sweep_status = gr.Textbox(
			label="Status",
			interactive=False,
			value="Ready",
		)

		with gr.Row():
			sweep_plot = gr.Plot(label="Parameter Sweep Results")
			sensitivity_plot = gr.Plot(label="Sensitivity Analysis")

		results_table = gr.Dataframe(
			label="Sweep Results",
			interactive=False,
		)

	# --- 回调 ---

	def _on_param_change(param_name: str):
		"""参数选择变化时更新默认范围"""
		defaults = SmartGridStressTestRunner.SWEEP_DEFAULTS
		if param_name in defaults:
			d = defaults[param_name]
			return d["min"], d["max"]
		return 0.0, 1.0

	def _run_sweep(
		param_name: str,
		s_min: float,
		s_max: float,
		s_steps: int,
		system_name: str,
		seed: int,
		max_steps: int,
	):
		"""执行参数扫描"""
		config = {
			"system_name": system_name,
			"max_episode_steps": int(max_steps),
			"use_cmdp": True,
		}

		inference_engine = shared_states.get("inference_engine")

		runner = SmartGridStressTestRunner(
			config=config,
			inference_engine=inference_engine,
		)

		try:
			results = runner.run_sweep(
				param_name=param_name,
				values=np.linspace(float(s_min), float(s_max), int(s_steps)).tolist(),
				seed=int(seed),
				max_steps=int(max_steps),
			)

			# 构建图表
			fig = _build_sweep_chart(param_name, results)

			# 构建表格
			table = _build_results_table(param_name, results)

			return (
				f"Sweep complete: {len(results)} runs",
				fig,
				table,
			)

		except Exception as e:
			logger.error(f"Sweep failed: {e}")
			return f"Error: {e}", None, []

	def _run_sensitivity(
		param_name: str,
		s_min: float,
		s_max: float,
		s_steps: int,
		system_name: str,
		seed: int,
		max_steps: int,
	):
		"""执行灵敏度分析"""
		config = {
			"system_name": system_name,
			"max_episode_steps": int(max_steps),
			"use_cmdp": True,
		}

		inference_engine = shared_states.get("inference_engine")

		runner = SmartGridStressTestRunner(
			config=config,
			inference_engine=inference_engine,
		)

		try:
			results = runner.run_sensitivity(
				param_name=param_name,
				base_value=float((s_min + s_max) / 2),
				delta_pct=0.1,
				seed=int(seed),
				max_steps=int(max_steps),
			)

			fig = _build_sensitivity_chart(param_name, results)

			return f"Sensitivity analysis complete", fig

		except Exception as e:
			logger.error(f"Sensitivity analysis failed: {e}")
			return f"Error: {e}", None

	# --- 绑定 ---

	param_dropdown.change(
		fn=_on_param_change,
		inputs=[param_dropdown],
		outputs=[sweep_min, sweep_max],
	)

	run_sweep_btn.click(
		fn=_run_sweep,
		inputs=[
			param_dropdown, sweep_min, sweep_max, sweep_steps,
			system_dropdown, seed_input, max_steps_input,
		],
		outputs=[sweep_status, sweep_plot, results_table],
	)

	run_sensitivity_btn.click(
		fn=_run_sensitivity,
		inputs=[
			param_dropdown, sweep_min, sweep_max, sweep_steps,
			system_dropdown, seed_input, max_steps_input,
		],
		outputs=[sweep_status, sensitivity_plot],
	)

	components["sweep_plot"] = sweep_plot
	components["sensitivity_plot"] = sensitivity_plot

	return components


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _build_sweep_chart(
	param_name: str,
	results: List[Dict[str, Any]],
) -> go.Figure:
	"""构建参数扫描结果图"""
	layout = get_plotly_layout(
		title=f"Parameter Sweep: {param_name}",
		height=450,
		env_name="smartgrid",
	)
	fig = go.Figure(layout=layout)

	param_vals = [r["params"].get(param_name, 0) for r in results]
	rewards = [r["metrics"].get("total_reward", 0) for r in results]
	violations = [r["metrics"].get("avg_violation_rate", 0) for r in results]
	losses = [r["metrics"].get("avg_loss_kw", 0) for r in results]

	fig.add_trace(go.Scatter(
		x=param_vals, y=rewards,
		mode="lines+markers", name="Total Reward",
		line={"color": COLORS["primary"], "width": 2},
		yaxis="y",
	))

	if any(v > 0 for v in violations):
		fig.add_trace(go.Scatter(
			x=param_vals, y=violations,
			mode="lines+markers", name="Violation Rate",
			line={"color": COLORS["danger"], "width": 2, "dash": "dot"},
			yaxis="y2",
		))

	fig.update_layout(
		xaxis_title=param_name,
		yaxis_title="Total Reward",
		yaxis2={
			"title": "Violation Rate",
			"overlaying": "y",
			"side": "right",
			"titlefont": {"color": COLORS["danger"]},
		},
	)

	return fig


def _build_sensitivity_chart(
	param_name: str,
	results: Dict[str, Any],
) -> go.Figure:
	"""构建灵敏度分析图"""
	layout = get_plotly_layout(
		title=f"Sensitivity: {param_name}",
		height=450,
		env_name="smartgrid",
	)
	fig = go.Figure(layout=layout)

	if "base" in results and "perturbed" in results:
		base_metrics = results["base"].get("metrics", {})
		perturbed = results.get("perturbed", [])

		metric_names = list(base_metrics.keys())
		for metric_name in metric_names[:5]:
			base_val = base_metrics.get(metric_name, 0)
			if base_val == 0:
				continue

			deltas = []
			labels = []
			for p in perturbed:
				p_val = p.get("metrics", {}).get(metric_name, 0)
				delta_pct = ((p_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
				deltas.append(delta_pct)
				labels.append(f"{p.get('delta', 0):+.1%}")

			if deltas:
				fig.add_trace(go.Bar(
					x=labels, y=deltas,
					name=metric_name,
				))

	fig.update_layout(
		xaxis_title="Parameter Perturbation",
		yaxis_title="Metric Change (%)",
		barmode="group",
	)

	return fig


def _build_results_table(
	param_name: str,
	results: List[Dict[str, Any]],
) -> List[List[str]]:
	"""构建扫描结果表格"""
	if not results:
		return []

	metric_keys = list(results[0].get("metrics", {}).keys())
	headers = [param_name] + metric_keys

	rows: List[List[str]] = []
	for r in results:
		row = [f"{r['params'].get(param_name, 0):.4g}"]
		for key in metric_keys:
			val = r["metrics"].get(key, 0)
			row.append(f"{val:.4g}")
		rows.append(row)

	return rows
