# -*- coding: utf-8 -*-
"""
DSR Tab: Analytics
分析标签页

包含恢复效率分析、网络状态跟踪、优先级负荷饼图、
奖励分解等 DSR 特有的分析视图。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

from envs.dsr.render.viz.plotly.reward_breakdown import create_reward_breakdown
from envs.dsr.render.viz.plotly.restoration_progress import create_restoration_progress
from envs.dsr.render.viz.plotly.network_state_chart import create_network_state_chart
from envs.dsr.render.viz.plotly.voltage_profile import create_voltage_profile

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

		gr.Markdown("### Restoration Efficiency & Network State Analysis")

		with gr.Row():
			source_dropdown = gr.Dropdown(
				label="Data Source",
				choices=["Live Inference", "Loaded Recording"],
				value="Live Inference",
			)
			analyze_btn = gr.Button("Analyze", variant="primary")

		with gr.Row():
			restoration_plot = gr.Plot(label="Restoration Progress (Detailed)")
			network_state_plot = gr.Plot(label="Network State Distribution")

		gr.Markdown("### Reward & Voltage Analysis")

		with gr.Row():
			reward_breakdown_plot = gr.Plot(label="Reward Breakdown")
			voltage_ts_plot = gr.Plot(label="Voltage Profile (Energized Buses)")

		gr.Markdown("### Episode Statistics")

		stats_table = gr.Dataframe(
			label="Key Metrics",
			headers=["Metric", "Value"],
			interactive=False,
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
			return None, None, None, None, [[empty_msg, ""]]

		# 恢复进度
		restoration_fig = create_restoration_progress(snapshots)

		# 网络状态分布
		network_fig = create_network_state_chart(snapshots)

		# 奖励分解
		reward_fig = create_reward_breakdown(snapshots)

		# 电压时间序列
		voltage_fig = create_voltage_profile(snapshots)

		# 统计表
		stats = _compute_episode_stats(snapshots)

		return restoration_fig, network_fig, reward_fig, voltage_fig, stats

	def _compute_episode_stats(
		snapshots: List[Dict[str, Any]],
	) -> List[List[str]]:
		"""计算 episode 统计指标"""
		rows: List[List[str]] = []

		rows.append(["Total Steps", str(len(snapshots))])

		# 恢复统计
		if snapshots:
			last = snapshots[-1]
			rest = last.get("restoration_data", {})
			rows.append(["Final Restoration (%)", f"{rest.get('restoration_pct', 0):.1f}"])
			rows.append(["Total Restored (kW)", f"{rest.get('total_restored_kw', 0):.1f}"])
			rows.append(["Energized Buses", str(len(rest.get("energized_buses", [])))])
			rows.append(["De-energized Buses", str(len(rest.get("de_energized_buses", [])))])
			rows.append(["Fault Lines", str(len(rest.get("fault_lines", [])))])

		# 电压统计 (仅 energized 母线)
		all_v: List[float] = []
		violations = 0
		for snap in snapshots:
			buses = snap.get("buses", {})
			for bus_info in buses.values():
				if not bus_info.get("is_energized", False):
					continue
				v_pu = bus_info.get("v_mag_pu", [])
				for v in v_pu:
					fv = float(v)
					all_v.append(fv)
					if fv < 0.95 or fv > 1.05:
						violations += 1

		if all_v:
			arr = np.array(all_v)
			rows.append(["V Mean (p.u.) [Energized]", f"{float(np.mean(arr)):.4f}"])
			rows.append(["V Min (p.u.) [Energized]", f"{float(np.min(arr)):.4f}"])
			rows.append(["V Max (p.u.) [Energized]", f"{float(np.max(arr)):.4f}"])
			viol_rate = violations / len(all_v) * 100
			rows.append(["Voltage Violation Rate", f"{viol_rate:.2f}%"])

		# 累计奖励
		if snapshots:
			last = snapshots[-1]
			cum_r = last.get("cumulative_reward", 0.0)
			rows.append(["Cumulative Reward", f"{cum_r:.4f}"])

		# 优先级恢复摘要
		if snapshots:
			last_rest = snapshots[-1].get("restoration_data", {})
			breakdown = last_rest.get("priority_breakdown", {})
			for priority in ["critical", "high", "medium", "low"]:
				stats = breakdown.get(priority, {})
				if stats.get("total_count", 0) > 0:
					rows.append([
						f"{priority.title()} Load Restoration",
						f"{stats.get('restored_count', 0)}/{stats.get('total_count', 0)} "
						f"({stats.get('pct', 0):.0f}%)",
					])

		return rows

	# --- 绑定 ---

	analyze_btn.click(
		fn=_analyze,
		inputs=[source_dropdown],
		outputs=[
			restoration_plot,
			network_state_plot,
			reward_breakdown_plot,
			voltage_ts_plot,
			stats_table,
		],
	)

	components["restoration_plot"] = restoration_plot
	components["network_state_plot"] = network_state_plot
	components["stats_table"] = stats_table

	return components
