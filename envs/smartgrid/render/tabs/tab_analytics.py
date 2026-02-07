# -*- coding: utf-8 -*-
"""
SmartGrid Tab: Analytics
分析标签页

包含 CMDP 奖励分解、Lagrangian lambda 轨迹分析、
电压统计等 SmartGrid 特有的分析视图。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

from envs.smartgrid.render.viz.plotly.reward_breakdown import create_reward_breakdown
from envs.smartgrid.render.viz.plotly.lagrangian_trajectory import (
	create_lagrangian_trajectory,
	create_lambda_convergence_summary,
)
from envs.smartgrid.render.viz.plotly.voltage_profile import create_voltage_timeseries

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Analytics 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Analytics"):

		gr.Markdown("### CMDP Reward & Lagrangian Analysis")

		with gr.Row():
			source_dropdown = gr.Dropdown(
				label="Data Source",
				choices=["Live Inference", "Loaded Recording"],
				value="Live Inference",
			)
			analyze_btn = gr.Button("Analyze", variant="primary")

		with gr.Row():
			reward_breakdown_plot = gr.Plot(label="CMDP Reward Breakdown")
			lagrangian_plot = gr.Plot(label="Lagrangian Lambda Trajectory")

		gr.Markdown("### Voltage Time Series")

		voltage_ts_plot = gr.Plot(label="Voltage Time Series (Representative Buses)")

		gr.Markdown("### Episode Statistics")

		with gr.Row():
			stats_table = gr.Dataframe(
				label="Key Metrics",
				headers=["Metric", "Value"],
				interactive=False,
			)
			lambda_summary_plot = gr.Plot(
				label="Lambda Convergence (Cross-Episode)",
				visible=True,
			)

	# --- 回调 ---

	def _analyze(source: str):
		"""执行分析"""
		if source == "Live Inference":
			snapshots = shared_states.get("live_snapshots", [])
		else:
			ep = shared_states.get("loaded_episode")
			snapshots = ep.snapshots if ep else []

		if not snapshots:
			empty_msg = "No data available"
			return None, None, None, [[empty_msg, ""]], None

		# 奖励分解
		reward_fig = create_reward_breakdown(snapshots)

		# Lagrangian 轨迹
		lagrangian_fig = create_lagrangian_trajectory(snapshots)

		# 电压时间序列
		voltage_ts_fig = create_voltage_timeseries(snapshots)

		# 统计表
		stats = _compute_episode_stats(snapshots)

		# Lambda 收敛摘要 (如果有跨 episode 数据)
		lambda_history = shared_states.get("lambda_history", [])
		cost_history = shared_states.get("cost_history", [])
		lambda_fig = None
		if lambda_history and cost_history:
			lambda_fig = create_lambda_convergence_summary(
				lambda_history, cost_history
			)

		return reward_fig, lagrangian_fig, voltage_ts_fig, stats, lambda_fig

	def _compute_episode_stats(
		snapshots: List[Dict[str, Any]],
	) -> List[List[str]]:
		"""计算 episode 统计指标"""
		rows: List[List[str]] = []

		rows.append(["Total Steps", str(len(snapshots))])

		# 电压统计
		all_v: List[float] = []
		violations = 0
		for snap in snapshots:
			buses = snap.get("buses", {})
			for bus_info in buses.values():
				v_pu = bus_info.get("v_mag_pu", [])
				for v in v_pu:
					fv = float(v)
					all_v.append(fv)
					if fv < 0.95 or fv > 1.05:
						violations += 1

		if all_v:
			arr = np.array(all_v)
			rows.append(["V Mean (p.u.)", f"{float(np.mean(arr)):.4f}"])
			rows.append(["V Min (p.u.)", f"{float(np.min(arr)):.4f}"])
			rows.append(["V Max (p.u.)", f"{float(np.max(arr)):.4f}"])
			rows.append(["V Std (p.u.)", f"{float(np.std(arr)):.4f}"])
			viol_rate = violations / len(all_v) * 100
			rows.append(["Voltage Violation Rate", f"{viol_rate:.2f}%"])

		# 损耗统计
		loss_vals: List[float] = []
		for snap in snapshots:
			circuit = snap.get("circuit", {})
			loss = circuit.get("total_loss_kw")
			if isinstance(loss, (int, float)):
				loss_vals.append(float(loss))

		if loss_vals:
			rows.append(["Avg Loss (kW)", f"{float(np.mean(loss_vals)):.2f}"])
			rows.append(["Total Loss (kWh)", f"{float(np.sum(loss_vals)):.2f}"])

		# Lambda 统计
		lambda_vals: List[float] = []
		for snap in snapshots:
			lagrangian = snap.get("lagrangian", {})
			info = snap.get("info", {})
			lmbda = lagrangian.get("lmbda", info.get("lambda"))
			if lmbda is not None:
				lambda_vals.append(float(lmbda))

		if lambda_vals:
			rows.append(["Lambda (initial)", f"{lambda_vals[0]:.4f}"])
			rows.append(["Lambda (final)", f"{lambda_vals[-1]:.4f}"])
			rows.append(["Lambda (range)", f"[{min(lambda_vals):.4f}, {max(lambda_vals):.4f}]"])

		# 累计奖励
		if snapshots:
			last = snapshots[-1]
			cum_r = last.get("cumulative_reward", 0.0)
			rows.append(["Cumulative Reward", f"{cum_r:.4f}"])

		return rows

	# --- 绑定 ---

	analyze_btn.click(
		fn=_analyze,
		inputs=[source_dropdown],
		outputs=[
			reward_breakdown_plot,
			lagrangian_plot,
			voltage_ts_plot,
			stats_table,
			lambda_summary_plot,
		],
	)

	components["reward_breakdown_plot"] = reward_breakdown_plot
	components["lagrangian_plot"] = lagrangian_plot
	components["stats_table"] = stats_table

	return components
