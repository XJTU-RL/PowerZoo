# -*- coding: utf-8 -*-
"""
Tab 4: Analytics

数据分析面板，包含 6 种图表：
- 电压热力图
- 电压时间序列
- 设备调度
- 奖励分解
- 电价信号分析 (Stackelberg 特色)
- Leader-Follower 博弈动态 (Stackelberg 特色)
"""

import logging
from typing import Any, Dict

import gradio as gr

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Analytics 标签页

	Args:
		shared_states: 共享状态字典
	"""
	gr.Markdown("### Analytics Dashboard")

	with gr.Row():
		data_source = gr.Radio(
			label="Data Source",
			choices=["Live Inference", "Loaded Episode"],
			value="Loaded Episode",
		)
		refresh_btn = gr.Button("Refresh All Charts", variant="primary")

	with gr.Tabs():
		with gr.Tab("Voltage Heatmap"):
			heatmap_plot = gr.Plot(label="Voltage Heatmap")

		with gr.Tab("Voltage Timeseries"):
			voltage_ts_plot = gr.Plot(label="Voltage Timeseries")

		with gr.Tab("Device Schedule"):
			device_plot = gr.Plot(label="Device Schedule")

		with gr.Tab("Reward Breakdown"):
			with gr.Row():
				reward_ts_plot = gr.Plot(label="UC vs Consumer Rewards")
			with gr.Row():
				reward_bar_plot = gr.Plot(label="Cumulative Reward by Agent")
				reward_radar_plot = gr.Plot(label="Performance Radar")

		with gr.Tab("Price Signal Analysis"):
			price_signal_plot = gr.Plot(label="Price Signal & Market Dynamics")
			price_compare_plot = gr.Plot(label="TOU Base vs Effective Price")

		with gr.Tab("Leader-Follower Dynamics"):
			game_dynamics_plot = gr.Plot(label="Leader-Follower Game Dynamics")
			action_heatmap_plot = gr.Plot(label="Action Heatmap")

	summary_md = gr.Markdown("Click 'Refresh All Charts' to generate analytics.")

	# ------------------------------------------------------------------
	# 回调
	# ------------------------------------------------------------------

	def _refresh_analytics(snapshots):
		"""刷新所有图表"""
		if not snapshots or not isinstance(snapshots, list):
			empty_msg = "No data available. Load episode data from Tab 1."
			return (None,) * 8 + (empty_msg,)

		results = {}

		# 电压热力图
		try:
			from envs.stackelberg.render.viz.plotly.voltage_heatmap import create_voltage_heatmap
			results["heatmap"] = create_voltage_heatmap(snapshots)
		except Exception as exc:
			logger.debug(f"Heatmap failed: {exc}")
			results["heatmap"] = None

		# 电压时间序列
		try:
			from envs.stackelberg.render.viz.plotly.voltage_profile import create_voltage_timeseries
			results["voltage_ts"] = create_voltage_timeseries(snapshots)
		except Exception as exc:
			logger.debug(f"Voltage TS failed: {exc}")
			results["voltage_ts"] = None

		# 设备调度
		try:
			from envs.stackelberg.render.viz.plotly.device_schedule_chart import create_device_schedule
			results["device"] = create_device_schedule(snapshots)
		except Exception as exc:
			logger.debug(f"Device schedule failed: {exc}")
			results["device"] = None

		# 奖励分解
		try:
			from envs.stackelberg.render.viz.plotly.reward_breakdown import (
				create_reward_timeseries,
				create_reward_comparison_bar,
				create_reward_radar,
			)
			results["reward_ts"] = create_reward_timeseries(snapshots)
			results["reward_bar"] = create_reward_comparison_bar(snapshots)
			results["reward_radar"] = create_reward_radar(snapshots)
		except Exception as exc:
			logger.debug(f"Reward breakdown failed: {exc}")
			results["reward_ts"] = None
			results["reward_bar"] = None
			results["reward_radar"] = None

		# 电价信号
		try:
			from envs.stackelberg.render.viz.plotly.price_signal_chart import (
				create_price_signal_chart,
				create_price_comparison,
			)
			results["price_signal"] = create_price_signal_chart(snapshots)
			results["price_compare"] = create_price_comparison(snapshots)
		except Exception as exc:
			logger.debug(f"Price signal failed: {exc}")
			results["price_signal"] = None
			results["price_compare"] = None

		# Leader-Follower 动态
		try:
			from envs.stackelberg.render.viz.plotly.leader_follower_chart import (
				create_leader_follower_chart,
				create_action_heatmap,
			)
			results["game_dynamics"] = create_leader_follower_chart(snapshots)
			results["action_heatmap"] = create_action_heatmap(snapshots)
		except Exception as exc:
			logger.debug(f"Leader-Follower failed: {exc}")
			results["game_dynamics"] = None
			results["action_heatmap"] = None

		# 汇总统计
		n_steps = len(snapshots)
		total_reward = sum(s.get("step_reward", 0.0) for s in snapshots)
		summary = f"**{n_steps} steps** | Total reward: {total_reward:.4f}"

		return (
			results.get("heatmap"),
			results.get("voltage_ts"),
			results.get("device"),
			results.get("reward_ts"),
			results.get("reward_bar"),
			results.get("reward_radar"),
			results.get("price_signal"),
			results.get("price_compare"),
			results.get("game_dynamics"),
			results.get("action_heatmap"),
			summary,
		)

	outputs = [
		heatmap_plot, voltage_ts_plot, device_plot,
		reward_ts_plot, reward_bar_plot, reward_radar_plot,
		price_signal_plot, price_compare_plot,
		game_dynamics_plot, action_heatmap_plot,
		summary_md,
	]

	refresh_btn.click(
		_refresh_analytics,
		inputs=[shared_states["snapshots"]],
		outputs=outputs,
	)
