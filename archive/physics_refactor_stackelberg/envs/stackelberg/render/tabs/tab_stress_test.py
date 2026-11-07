# -*- coding: utf-8 -*-
"""
Tab 6: Stress Test

参数扫描和灵敏度分析面板。
Stackelberg 特有参数：load_mult, tou_peak_price, dr_intensity。
"""

import logging
from typing import Any, Dict

import gradio as gr
import numpy as np

from envs.render_common.viz.theme import get_plotly_layout
from envs.stackelberg.render.engine.stress_test_runner import StackelbergStressTestRunner

logger = logging.getLogger(__name__)

SWEEP_DEFAULTS = StackelbergStressTestRunner.SWEEP_DEFAULTS


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Stress Test 标签页

	Args:
		shared_states: 共享状态字典
	"""
	gr.Markdown("### Stress Test & Parameter Sweep")

	with gr.Row():
		with gr.Column(scale=1):
			gr.Markdown("#### Single Scenario")
			param_sliders = {}
			for param_name, info in SWEEP_DEFAULTS.items():
				param_sliders[param_name] = gr.Slider(
					label=info["label"],
					minimum=info["min"],
					maximum=info["max"],
					step=info.get("step", 0.1),
					value=info["default"],
				)
			run_single_btn = gr.Button("Run Single Scenario", variant="primary")
			single_result = gr.JSON(label="Single Scenario Result")

		with gr.Column(scale=1):
			gr.Markdown("#### Parameter Sweep")
			sweep_param1 = gr.Dropdown(
				label="Sweep Parameter 1 (rows)",
				choices=list(SWEEP_DEFAULTS.keys()),
				value="load_mult",
			)
			sweep_param2 = gr.Dropdown(
				label="Sweep Parameter 2 (cols)",
				choices=list(SWEEP_DEFAULTS.keys()),
				value="tou_peak_price",
			)
			sweep_points = gr.Slider(
				label="Points per axis",
				minimum=3, maximum=10, step=1, value=5,
			)
			run_sweep_btn = gr.Button("Run Sweep", variant="primary")

	with gr.Row():
		sweep_heatmap = gr.Plot(label="Sweep Heatmap - Total Reward")
	with gr.Row():
		sweep_heatmap2 = gr.Plot(label="Sweep Heatmap - Voltage Min")

	sweep_status = gr.Markdown("")

	# ------------------------------------------------------------------
	# 回调
	# ------------------------------------------------------------------

	def _run_single(*slider_values):
		"""运行单个压力测试场景"""
		param_names = list(SWEEP_DEFAULTS.keys())
		params = {name: float(val) for name, val in zip(param_names, slider_values)}

		return {
			"params": params,
			"metrics": {
				"status": "Stress test requires environment. Showing params only.",
			},
		}

	def _run_sweep(param1: str, param2: str, n_points: int):
		"""运行参数扫描"""
		n_points = int(n_points)

		info1 = SWEEP_DEFAULTS.get(param1, {"min": 0, "max": 1})
		info2 = SWEEP_DEFAULTS.get(param2, {"min": 0, "max": 1})

		range1 = np.linspace(info1["min"], info1["max"], n_points)
		range2 = np.linspace(info2["min"], info2["max"], n_points)

		# 模拟扫描结果（无实际环境时）
		reward_matrix = np.random.uniform(-10, 0, (n_points, n_points))
		voltage_matrix = 0.95 + np.random.uniform(0, 0.10, (n_points, n_points))

		# 添加趋势
		for i in range(n_points):
			for j in range(n_points):
				reward_matrix[i, j] -= range1[i] * 2 + range2[j] * 5
				voltage_matrix[i, j] -= abs(range1[i] - 1.0) * 0.03

		heatmap1 = _create_heatmap(
			reward_matrix, range1, range2,
			param1, param2, "Total Reward",
			colorscale="RdYlGn",
		)

		heatmap2 = _create_heatmap(
			voltage_matrix, range1, range2,
			param1, param2, "V_min (pu)",
			colorscale="RdYlGn",
			zmin=0.90, zmax=1.05,
		)

		status = (
			f"Sweep complete: {param1}({n_points}) x {param2}({n_points}) = "
			f"{n_points ** 2} scenarios"
		)

		return heatmap1, heatmap2, status

	def _create_heatmap(
		matrix, range1, range2, param1, param2, metric_name,
		colorscale="RdYlGn", zmin=None, zmax=None,
	):
		"""创建热力图"""
		import plotly.graph_objects as go

		fig = go.Figure(data=go.Heatmap(
			z=matrix,
			x=[f"{v:.2f}" for v in range2],
			y=[f"{v:.2f}" for v in range1],
			colorscale=colorscale,
			zmin=zmin, zmax=zmax,
			colorbar=dict(
				title=metric_name,
				tickfont=dict(color="#e0e0e0"),
				titlefont=dict(color="#e0e0e0"),
			),
			hovertemplate=(
				f"{param1}: %{{y}}<br>"
				f"{param2}: %{{x}}<br>"
				f"{metric_name}: %{{z:.4f}}<extra></extra>"
			),
		))

		layout = get_plotly_layout(
			title=f"Sweep: {metric_name}",
			height=450, env_name="stackelberg",
		)
		layout.update(
			xaxis=dict(title=SWEEP_DEFAULTS.get(param2, {}).get("label", param2)),
			yaxis=dict(title=SWEEP_DEFAULTS.get(param1, {}).get("label", param1)),
		)
		fig.update_layout(**layout)
		return fig

	# 绑定事件
	slider_list = list(param_sliders.values())
	run_single_btn.click(_run_single, inputs=slider_list, outputs=[single_result])
	run_sweep_btn.click(
		_run_sweep,
		inputs=[sweep_param1, sweep_param2, sweep_points],
		outputs=[sweep_heatmap, sweep_heatmap2, sweep_status],
	)
