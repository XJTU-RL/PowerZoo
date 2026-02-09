# -*- coding: utf-8 -*-
"""
Tab 6: Scenario Stress Test
场景压力测试标签页 -- 参数化压力测试 + 二维参数扫描热力图

提供两种测试模式:
1. 单场景运行: 调整 6 个参数后运行 1 个 episode
2. 参数扫描: 选择两个参数做网格搜索，生成指标热力图
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.district_dispatch.render.engine.stress_test_runner import (
	SWEEP_PARAM_DEFAULTS,
	StressTestRunner,
)
from envs.district_dispatch.render.viz.plotly.topology_graph import (
	create_topology_figure,
)

logger = logging.getLogger(__name__)

# 参数显示名称映射
_PARAM_LABELS: Dict[str, str] = {
	"load_multiplier": "Load Multiplier",
	"pv_output_ratio": "PV Output Ratio",
	"initial_soc": "Initial SOC",
	"ev_demand_mult": "EV Demand Mult",
	"carbon_intensity": "Carbon Intensity",
	"price_multiplier": "Price Multiplier",
}

# 参数滑块配置: (min, max, default, step)
_SLIDER_CONFIGS: Dict[str, Tuple[float, float, float, float]] = {
	"load_multiplier": (0.5, 2.0, 1.0, 0.1),
	"pv_output_ratio": (0.0, 1.0, 1.0, 0.05),
	"initial_soc": (0.1, 0.9, 0.5, 0.05),
	"ev_demand_mult": (0.0, 3.0, 1.0, 0.1),
	"carbon_intensity": (0.0, 1.2, 0.5, 0.05),
	"price_multiplier": (0.5, 2.0, 1.0, 0.1),
}

# 扫描参数下拉选项
_SWEEP_CHOICES: List[str] = list(_PARAM_LABELS.values())

# 标签名 → 参数名反向映射
_LABEL_TO_PARAM: Dict[str, str] = {v: k for k, v in _PARAM_LABELS.items()}


def _build_single_summary(metrics: Dict[str, float], elapsed: float) -> str:
	"""构建单场景结果的 Markdown 摘要。"""
	lines = [
		"### Single Scenario Results",
		"",
		f"| Metric | Value |",
		f"|--------|-------|",
		f"| Total Reward | {metrics.get('total_reward', 0):.4f} |",
		f"| Voltage Violation | {metrics.get('voltage_violation_pct', 0):.2f}% |",
		f"| Total Loss (kWh) | {metrics.get('total_loss_kwh', 0):.2f} |",
		f"| PV Utilization | {metrics.get('pv_utilization', 0):.1f}% |",
		f"| Episode Length | {metrics.get('episode_length', 0)} steps |",
		f"| Elapsed | {elapsed:.2f}s |",
	]
	return "\n".join(lines)


def _build_heatmap_figure(
	sweep_result: Dict[str, Any],
	metric_name: str = "total_reward",
) -> go.Figure:
	"""从 sweep 结果构建 Plotly 热力图。"""
	heatmaps = sweep_result.get("heatmaps", {})
	p1_name = sweep_result["param1_name"]
	p2_name = sweep_result["param2_name"]
	p1_range = sweep_result["param1_range"]
	p2_range = sweep_result["param2_range"]

	z = heatmaps.get(metric_name)
	if z is None:
		# 回退到第一个可用的指标
		if heatmaps:
			metric_name = next(iter(heatmaps))
			z = heatmaps[metric_name]
		else:
			z = np.zeros((1, 1))

	if isinstance(z, np.ndarray):
		z = z.tolist()

	p1_label = _PARAM_LABELS.get(p1_name, p1_name)
	p2_label = _PARAM_LABELS.get(p2_name, p2_name)

	fig = go.Figure(data=go.Heatmap(
		z=z,
		x=[f"{v:.2f}" for v in p2_range],
		y=[f"{v:.2f}" for v in p1_range],
		colorscale="Viridis",
		colorbar=dict(title=metric_name),
		hovertemplate=(
			f"{p2_label}: %{{x}}<br>"
			f"{p1_label}: %{{y}}<br>"
			f"{metric_name}: %{{z:.4f}}"
			"<extra></extra>"
		),
	))

	fig.update_layout(
		title=f"Parameter Sweep: {metric_name}",
		xaxis_title=p2_label,
		yaxis_title=p1_label,
		template="plotly_dark",
		paper_bgcolor="#0f0f23",
		plot_bgcolor="#0f0f23",
		height=500,
		margin=dict(l=80, r=40, t=60, b=80),
	)
	return fig


def _run_single_scenario(
	load_mult: float,
	pv_ratio: float,
	init_soc: float,
	ev_mult: float,
	carbon: float,
	price_mult: float,
	inference_engine: Any,
	bus_coords: Any,
) -> Tuple[Optional[go.Figure], str, str]:
	"""运行单场景压力测试。

	Returns:
		(topology_fig, summary_md, progress_msg)
	"""
	params = {
		"load_multiplier": load_mult,
		"pv_output_ratio": pv_ratio,
		"initial_soc": init_soc,
		"ev_demand_mult": ev_mult,
		"carbon_intensity": carbon,
		"price_multiplier": price_mult,
	}

	if inference_engine is None:
		return None, "**Error**: No inference engine loaded.", "No model loaded"

	try:
		runner = StressTestRunner(
			config=inference_engine.config
			if hasattr(inference_engine, "config")
			else {},
			inference_engine=inference_engine,
		)
		result = runner.run_single(params, seed=42, collect_snapshots=True)
		metrics = result["metrics"]
		elapsed = result["elapsed_seconds"]

		# 生成拓扑图 (使用最后一个快照)
		topo_fig = None
		episode_data = result.get("episode_data")
		if episode_data and episode_data.snapshots and bus_coords:
			last_snap = episode_data.snapshots[-1]
			try:
				topo_fig = create_topology_figure(last_snap, bus_coords)
			except Exception as e:
				logger.warning(f"Failed to create topology figure: {e}")

		summary = _build_single_summary(metrics, elapsed)
		progress = f"Completed in {elapsed:.2f}s"
		return topo_fig, summary, progress

	except Exception as e:
		logger.error(f"Single scenario failed: {e}\n{traceback.format_exc()}")
		return None, f"**Error**: {e}", f"Failed: {e}"


def _run_parameter_sweep(
	sweep_param1_label: str,
	sweep_param2_label: str,
	n_points: int,
	load_mult: float,
	pv_ratio: float,
	init_soc: float,
	ev_mult: float,
	carbon: float,
	price_mult: float,
	inference_engine: Any,
) -> Tuple[Optional[go.Figure], str]:
	"""运行二维参数扫描。

	Returns:
		(heatmap_fig, progress_msg)
	"""
	if inference_engine is None:
		return None, "No model loaded"

	p1_name = _LABEL_TO_PARAM.get(sweep_param1_label)
	p2_name = _LABEL_TO_PARAM.get(sweep_param2_label)

	if not p1_name or not p2_name:
		return None, f"Invalid parameter selection: {sweep_param1_label}, {sweep_param2_label}"

	if p1_name == p2_name:
		return None, "Please select two different parameters"

	n_points = max(2, min(20, int(n_points)))

	# 构建扫描范围
	p1_info = SWEEP_PARAM_DEFAULTS.get(p1_name, {"min": 0.0, "max": 2.0})
	p2_info = SWEEP_PARAM_DEFAULTS.get(p2_name, {"min": 0.0, "max": 2.0})
	p1_range = np.linspace(p1_info["min"], p1_info["max"], n_points)
	p2_range = np.linspace(p2_info["min"], p2_info["max"], n_points)

	# 基准参数 (使用当前滑块值)
	base_params = {
		"load_multiplier": load_mult,
		"pv_output_ratio": pv_ratio,
		"initial_soc": init_soc,
		"ev_demand_mult": ev_mult,
		"carbon_intensity": carbon,
		"price_multiplier": price_mult,
	}

	try:
		runner = StressTestRunner(
			config=inference_engine.config
			if hasattr(inference_engine, "config")
			else {},
			inference_engine=inference_engine,
		)
		sweep_result = runner.run_sweep(
			param1_name=p1_name,
			param1_range=p1_range,
			param2_name=p2_name,
			param2_range=p2_range,
			seed=42,
			base_params=base_params,
		)

		fig = _build_heatmap_figure(sweep_result, metric_name="total_reward")
		total_runs = sweep_result.get("total_runs", 0)
		elapsed = sweep_result.get("elapsed_seconds", 0)
		msg = f"Sweep complete: {total_runs} runs in {elapsed:.1f}s"
		return fig, msg

	except Exception as e:
		logger.error(f"Parameter sweep failed: {e}\n{traceback.format_exc()}")
		return None, f"Sweep failed: {e}"


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Scenario Stress Test 标签页。

	Args:
		shared_states: 共享状态字典，包含:
			- "snapshots": gr.State
			- "bus_coords": gr.State
			- "model_loaded": gr.State
			- "inference_engine": gr.State
			- "episode_runner": gr.State

	Returns:
		该 Tab 内关键组件引用
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Scenario Stress Test"):
		gr.Markdown("## Scenario Stress Test")
		gr.Markdown(
			"Adjust environment parameters and run stress tests. "
			"Supports single scenario execution and 2D parameter sweep."
		)

		# -- 参数滑块面板 --
		with gr.Row():
			with gr.Column(scale=1):
				sliders: Dict[str, gr.Slider] = {}
				for param_key, (p_min, p_max, p_default, p_step) in _SLIDER_CONFIGS.items():
					label = _PARAM_LABELS[param_key]
					sliders[param_key] = gr.Slider(
						minimum=p_min,
						maximum=p_max,
						value=p_default,
						step=p_step,
						label=label,
					)
				components["sliders"] = sliders

		# -- 按钮栏 --
		with gr.Row():
			btn_single = gr.Button("Run Single Scenario", variant="primary")
			btn_sweep = gr.Button("Run Parameter Sweep", variant="secondary")

		# -- Sweep 设置 --
		with gr.Row():
			sweep_param1 = gr.Dropdown(
				choices=_SWEEP_CHOICES,
				value=_SWEEP_CHOICES[0],
				label="Sweep Parameter 1",
			)
			sweep_param2 = gr.Dropdown(
				choices=_SWEEP_CHOICES,
				value=_SWEEP_CHOICES[1],
				label="Sweep Parameter 2",
			)
			sweep_n_points = gr.Number(
				value=5,
				label="Sweep Points (per axis)",
				minimum=2,
				maximum=20,
				precision=0,
			)

		# -- 进度信息 --
		progress_box = gr.Textbox(
			label="Progress",
			interactive=False,
			lines=1,
		)

		# -- 结果区域 --
		with gr.Row():
			with gr.Column(scale=1):
				gr.Markdown("### Single Scenario Result")
				single_plot = gr.Plot(label="Topology (Final Step)")
				single_summary = gr.Markdown("*Run a scenario to see results.*")

			with gr.Column(scale=1):
				gr.Markdown("### Parameter Sweep Result")
				sweep_plot = gr.Plot(label="Heatmap")

		components.update({
			"btn_single": btn_single,
			"btn_sweep": btn_sweep,
			"sweep_param1": sweep_param1,
			"sweep_param2": sweep_param2,
			"sweep_n_points": sweep_n_points,
			"progress_box": progress_box,
			"single_plot": single_plot,
			"single_summary": single_summary,
			"sweep_plot": sweep_plot,
		})

		# -- 事件绑定: 单场景 --
		slider_list = [
			sliders["load_multiplier"],
			sliders["pv_output_ratio"],
			sliders["initial_soc"],
			sliders["ev_demand_mult"],
			sliders["carbon_intensity"],
			sliders["price_multiplier"],
		]

		btn_single.click(
			fn=_run_single_scenario,
			inputs=slider_list + [
				shared_states["inference_engine"],
				shared_states["bus_coords"],
			],
			outputs=[single_plot, single_summary, progress_box],
		)

		# -- 事件绑定: 参数扫描 --
		btn_sweep.click(
			fn=_run_parameter_sweep,
			inputs=[
				sweep_param1,
				sweep_param2,
				sweep_n_points,
			] + slider_list + [
				shared_states["inference_engine"],
			],
			outputs=[sweep_plot, progress_box],
		)

	return components
