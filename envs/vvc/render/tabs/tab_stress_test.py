# -*- coding: utf-8 -*-
"""
Tab 6: Stress Test - 压力测试

支持 load_mult / pv_ratio 扫描，
显示性能热图和敏感性分析结果。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.render_common.engine.inference_engine import InferenceEngine
from envs.render_common.viz.theme import get_plotly_layout, VOLTAGE_COLORSCALE
from envs.vvc.render.engine.stress_test_runner import VVCStressTestRunner

logger = logging.getLogger(__name__)


def _run_sweep(
	inference_engine: Optional[InferenceEngine],
	system: str,
	param_name: str,
	start: float,
	end: float,
	n_points: int,
	use_model: bool,
) -> Tuple[List[Dict[str, Any]], Any, Any, str]:
	"""执行参数扫描。"""
	if n_points < 2:
		return [], gr.Plot(), gr.Plot(), "Need at least 2 points"

	config = {
		"env_args": {
			"system": system,
			"episode_length": 24,
			"worker_idx": 98,
		},
		"seed": 42,
	}

	try:
		runner = VVCStressTestRunner(config=config, inference_engine=inference_engine)
		values = np.linspace(start, end, n_points).tolist()

		results = runner.run_sweep(
			param_name=param_name,
			values=values,
			use_model=use_model,
		)

		if not results:
			return [], gr.Plot(), gr.Plot(), "No results"

		# 奖励 vs 参数值图
		reward_fig = _create_sweep_reward_chart(results, param_name)

		# 电压范围图
		voltage_fig = _create_sweep_voltage_chart(results, param_name)

		status = f"Sweep done: {len(results)} points for {param_name}"
		return results, reward_fig, voltage_fig, status

	except Exception as exc:
		logger.error(f"Sweep failed: {exc}", exc_info=True)
		return [], gr.Plot(), gr.Plot(), f"Sweep failed: {exc}"


def _run_sensitivity(
	inference_engine: Optional[InferenceEngine],
	system: str,
	use_model: bool,
) -> Tuple[List[Dict[str, Any]], Any, str]:
	"""执行敏感性分析 (load_mult + pv_ratio 双参数)。"""
	config = {
		"env_args": {
			"system": system,
			"episode_length": 24,
			"worker_idx": 97,
		},
		"seed": 42,
	}

	try:
		runner = VVCStressTestRunner(config=config, inference_engine=inference_engine)
		results = runner.run_sensitivity(
			param_names=["load_mult", "pv_ratio"],
			use_model=use_model,
		)

		if not results:
			return [], gr.Plot(), "No results"

		heatmap_fig = _create_sensitivity_heatmap(results)
		status = f"Sensitivity analysis done: {len(results)} scenarios"
		return results, heatmap_fig, status

	except Exception as exc:
		logger.error(f"Sensitivity failed: {exc}", exc_info=True)
		return [], gr.Plot(), f"Sensitivity failed: {exc}"


def _create_sweep_reward_chart(
	results: List[Dict[str, Any]],
	param_name: str,
) -> go.Figure:
	"""创建扫描奖励曲线。"""
	layout = get_plotly_layout(title=f"Reward vs {param_name}")
	fig = go.Figure(layout=layout)

	param_vals = [r["params"].get(param_name, 0) for r in results]
	rewards = [r["metrics"].get("total_reward", 0) for r in results]

	fig.add_trace(go.Scatter(
		x=param_vals, y=rewards,
		mode="lines+markers",
		name="Total Reward",
		line=dict(color="#4F46E5", width=2),
		marker=dict(size=8),
	))

	fig.update_layout(
		xaxis_title=param_name,
		yaxis_title="Total Reward",
	)
	return fig


def _create_sweep_voltage_chart(
	results: List[Dict[str, Any]],
	param_name: str,
) -> go.Figure:
	"""创建扫描电压范围图。"""
	layout = get_plotly_layout(title=f"Voltage Range vs {param_name}")
	fig = go.Figure(layout=layout)

	param_vals = [r["params"].get(param_name, 0) for r in results]
	v_mins = [r["metrics"].get("min_voltage", 1.0) for r in results]
	v_maxs = [r["metrics"].get("max_voltage", 1.0) for r in results]
	v_avgs = [r["metrics"].get("avg_voltage", 1.0) for r in results]

	fig.add_trace(go.Scatter(
		x=param_vals, y=v_maxs,
		mode="lines",
		name="V Max",
		line=dict(color="#EF4444", width=1),
		fill=None,
	))
	fig.add_trace(go.Scatter(
		x=param_vals, y=v_mins,
		mode="lines",
		name="V Min",
		line=dict(color="#3B82F6", width=1),
		fill="tonexty",
		fillcolor="rgba(59, 130, 246, 0.1)",
	))
	fig.add_trace(go.Scatter(
		x=param_vals, y=v_avgs,
		mode="lines+markers",
		name="V Avg",
		line=dict(color="#22C55E", width=2),
		marker=dict(size=5),
	))

	# 安全区间
	fig.add_hline(y=0.95, line_dash="dot", line_color="#F59E0B", opacity=0.5)
	fig.add_hline(y=1.05, line_dash="dot", line_color="#F59E0B", opacity=0.5)

	fig.update_layout(
		xaxis_title=param_name,
		yaxis_title="Voltage (pu)",
	)
	return fig


def _create_sensitivity_heatmap(
	results: List[Dict[str, Any]],
) -> go.Figure:
	"""创建双参数敏感性热图。"""
	layout = get_plotly_layout(title="Sensitivity: Reward Heatmap")

	# 提取唯一参数值
	load_vals = sorted(set(r["params"].get("load_mult", 1.0) for r in results))
	pv_vals = sorted(set(r["params"].get("pv_ratio", 1.0) for r in results))

	# 构建矩阵
	z = np.full((len(pv_vals), len(load_vals)), np.nan)

	for r in results:
		lm = r["params"].get("load_mult", 1.0)
		pv = r["params"].get("pv_ratio", 1.0)
		reward = r["metrics"].get("total_reward", 0.0)

		if lm in load_vals and pv in pv_vals:
			li = load_vals.index(lm)
			pi = pv_vals.index(pv)
			z[pi][li] = reward

	fig = go.Figure(
		data=go.Heatmap(
			z=z.tolist(),
			x=[f"{v:.1f}" for v in load_vals],
			y=[f"{v:.1f}" for v in pv_vals],
			colorscale="RdYlGn",
			colorbar=dict(title="Reward"),
		),
		layout=layout,
	)

	fig.update_layout(
		xaxis_title="Load Multiplier",
		yaxis_title="PV Ratio",
	)
	return fig


def _build_results_table(
	results: List[Dict[str, Any]],
) -> List[List[str]]:
	"""构建结果表格。"""
	rows: List[List[str]] = []
	for i, r in enumerate(results):
		params = r.get("params", {})
		metrics = r.get("metrics", {})
		params_str = ", ".join(f"{k}={v:.2f}" for k, v in params.items())
		rows.append([
			str(i),
			params_str,
			f"{metrics.get('total_reward', 0):.4f}",
			f"{metrics.get('avg_voltage', 1.0):.4f}",
			f"{metrics.get('min_voltage', 1.0):.4f}",
			f"{metrics.get('max_voltage', 1.0):.4f}",
			f"{metrics.get('avg_loss', 0):.2f}",
			str(metrics.get("n_violations", 0)),
		])
	return rows


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Stress Test Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	state_inference_engine = shared_states["inference_engine"]

	# 本地状态
	state_results = gr.State([])

	# === 扫描控制 ===
	with gr.Group():
		gr.Markdown("### Parameter Sweep")
		with gr.Row():
			system_dropdown = gr.Dropdown(
				choices=["13Bus", "34Bus", "123Bus"],
				value="13Bus",
				label="System",
				scale=1,
			)
			param_dropdown = gr.Dropdown(
				choices=["load_mult", "pv_ratio"],
				value="load_mult",
				label="Parameter",
				scale=1,
			)
			start_input = gr.Number(label="Start", value=0.5, scale=1)
			end_input = gr.Number(label="End", value=2.0, scale=1)
			n_points_input = gr.Number(label="Points", value=6, precision=0, scale=1)

		with gr.Row():
			use_model_cb = gr.Checkbox(label="Use Model", value=True, scale=2)
			sweep_btn = gr.Button("Run Sweep", variant="primary", scale=1)
			sensitivity_btn = gr.Button("Run Sensitivity", variant="secondary", scale=1)

	# === 结果表 ===
	with gr.Accordion("Results Table", open=True):
		results_table = gr.Dataframe(
			headers=[
				"#", "Parameters", "Reward", "V Avg",
				"V Min", "V Max", "Avg Loss", "Violations",
			],
			datatype=["str"] * 8,
			interactive=False,
			wrap=True,
		)

	# === 图表 ===
	with gr.Row():
		with gr.Column():
			gr.Markdown("### Sweep: Reward")
			reward_plot = gr.Plot(label="Reward")
		with gr.Column():
			gr.Markdown("### Sweep: Voltage Range")
			voltage_plot = gr.Plot(label="Voltage")

	gr.Markdown("### Sensitivity Heatmap")
	heatmap_plot = gr.Plot(label="Sensitivity")

	status_box = gr.Textbox(label="Status", interactive=False, lines=1)

	# --- 事件绑定 ---
	def on_sweep(engine, system, param, start, end, n_points, use_model):
		results, reward_fig, voltage_fig, status = _run_sweep(
			engine, system, param, start, end, int(n_points), use_model,
		)
		table = _build_results_table(results)
		return results, table, reward_fig, voltage_fig, status

	def on_sensitivity(engine, system, use_model):
		results, heatmap_fig, status = _run_sensitivity(
			engine, system, use_model,
		)
		table = _build_results_table(results)
		return results, table, heatmap_fig, status

	sweep_btn.click(
		fn=on_sweep,
		inputs=[
			state_inference_engine, system_dropdown, param_dropdown,
			start_input, end_input, n_points_input, use_model_cb,
		],
		outputs=[state_results, results_table, reward_plot, voltage_plot, status_box],
	)
	sensitivity_btn.click(
		fn=on_sensitivity,
		inputs=[state_inference_engine, system_dropdown, use_model_cb],
		outputs=[state_results, results_table, heatmap_plot, status_box],
	)

	return {
		"reward_plot": reward_plot,
		"voltage_plot": voltage_plot,
		"heatmap_plot": heatmap_plot,
		"status_box": status_box,
	}
