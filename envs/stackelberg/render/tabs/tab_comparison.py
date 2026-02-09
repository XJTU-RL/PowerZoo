# -*- coding: utf-8 -*-
"""
Tab 5: Comparison

多 Episode 对比分析，支持最多 3 个 episode 并排比较。
"""

import json
import logging
import os
from typing import Any, Dict, List

import gradio as gr
import numpy as np

from envs.render_common.viz.theme import get_plotly_layout

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Comparison 标签页

	Args:
		shared_states: 共享状态字典
	"""
	gr.Markdown("### Episode Comparison (Up to 3)")

	with gr.Row():
		ep_paths = []
		for i in range(3):
			ep_paths.append(gr.Textbox(
				label=f"Episode {i + 1} Path",
				placeholder="/path/to/episode.json",
			))

	compare_btn = gr.Button("Compare", variant="primary")

	with gr.Row():
		reward_compare_plot = gr.Plot(label="Reward Comparison")
		voltage_compare_plot = gr.Plot(label="Voltage Comparison")

	with gr.Row():
		price_compare_plot = gr.Plot(label="Price Comparison")
		game_compare_plot = gr.Plot(label="Leader-Follower Comparison")

	summary_table = gr.Markdown("Load episode files and click Compare.")

	# ------------------------------------------------------------------
	# 回调
	# ------------------------------------------------------------------

	def _load_episode(path: str) -> List[Dict[str, Any]]:
		"""加载 episode 文件"""
		if not path or not os.path.isfile(path):
			return []
		try:
			with open(path, "r") as f:
				data = json.load(f)
			if isinstance(data, list):
				return data
			return data.get("snapshots", [data])
		except Exception:
			return []

	def _compare(path1: str, path2: str, path3: str):
		"""执行对比分析"""
		episodes = []
		labels = []
		for i, path in enumerate([path1, path2, path3]):
			snaps = _load_episode(path)
			if snaps:
				episodes.append(snaps)
				labels.append(f"Episode {i + 1}")

		if not episodes:
			return None, None, None, None, "No valid episodes loaded."

		# 奖励对比
		reward_fig = _create_reward_comparison(episodes, labels)

		# 电压对比
		voltage_fig = _create_voltage_comparison(episodes, labels)

		# 电价对比
		price_fig = _create_price_comparison(episodes, labels)

		# 博弈动态对比
		game_fig = _create_game_comparison(episodes, labels)

		# 汇总表格
		summary = _build_summary_table(episodes, labels)

		return reward_fig, voltage_fig, price_fig, game_fig, summary

	def _create_reward_comparison(episodes, labels):
		"""创建奖励对比图"""
		import plotly.graph_objects as go

		fig = go.Figure()
		colors = ["#F59E0B", "#3B82F6", "#10B981"]

		for idx, (snaps, label) in enumerate(zip(episodes, labels)):
			rewards = [s.get("step_reward", 0.0) for s in snaps if s.get("step", 0) > 0]
			steps = list(range(1, len(rewards) + 1))
			cumulative = np.cumsum(rewards).tolist()

			fig.add_trace(go.Scatter(
				x=steps, y=cumulative,
				mode="lines+markers",
				name=label,
				line=dict(color=colors[idx % len(colors)], width=2),
			))

		layout = get_plotly_layout(
			title="Cumulative Reward Comparison",
			height=400, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Step"), yaxis=dict(title="Cumulative Reward"))
		fig.update_layout(**layout)
		return fig

	def _create_voltage_comparison(episodes, labels):
		"""创建电压对比图"""
		import plotly.graph_objects as go

		fig = go.Figure()
		colors = ["#F59E0B", "#3B82F6", "#10B981"]

		for idx, (snaps, label) in enumerate(zip(episodes, labels)):
			v_means = [
				s.get("voltage_summary", {}).get("v_mean", 1.0)
				for s in snaps
			]
			steps = list(range(len(v_means)))
			fig.add_trace(go.Scatter(
				x=steps, y=v_means,
				mode="lines", name=f"{label} V_mean",
				line=dict(color=colors[idx % len(colors)], width=2),
			))

		fig.add_hline(y=0.95, line_dash="dot", line_color="rgba(239,68,68,0.5)")
		fig.add_hline(y=1.05, line_dash="dot", line_color="rgba(239,68,68,0.5)")

		layout = get_plotly_layout(
			title="Voltage Comparison",
			height=400, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Step"), yaxis=dict(title="Voltage (pu)"))
		fig.update_layout(**layout)
		return fig

	def _create_price_comparison(episodes, labels):
		"""创建电价对比图"""
		import plotly.graph_objects as go

		fig = go.Figure()
		colors = ["#F59E0B", "#3B82F6", "#10B981"]

		for idx, (snaps, label) in enumerate(zip(episodes, labels)):
			prices = []
			for s in snaps:
				market = s.get("market_data", {})
				uc_act = market.get("uc_actions", {})
				prices.append(uc_act.get("effective_price", 0.0))

			steps = list(range(len(prices)))
			fig.add_trace(go.Scatter(
				x=steps, y=prices,
				mode="lines+markers", name=f"{label} Price",
				line=dict(color=colors[idx % len(colors)], width=2),
			))

		layout = get_plotly_layout(
			title="Effective Price Comparison",
			height=400, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Step"), yaxis=dict(title="Price ($/kWh)"))
		fig.update_layout(**layout)
		return fig

	def _create_game_comparison(episodes, labels):
		"""创建博弈动态对比图"""
		import plotly.graph_objects as go

		fig = go.Figure()
		colors = ["#F59E0B", "#3B82F6", "#10B981"]

		for idx, (snaps, label) in enumerate(zip(episodes, labels)):
			uc_rewards = [s.get("uc_reward", 0.0) for s in snaps if s.get("step", 0) > 0]
			consumer_rewards = [s.get("avg_consumer_reward", 0.0) for s in snaps if s.get("step", 0) > 0]

			steps = list(range(1, len(uc_rewards) + 1))
			fig.add_trace(go.Scatter(
				x=steps, y=uc_rewards,
				mode="lines", name=f"{label} UC",
				line=dict(color=colors[idx % len(colors)], width=2),
			))
			fig.add_trace(go.Scatter(
				x=steps, y=consumer_rewards,
				mode="lines", name=f"{label} Consumer",
				line=dict(color=colors[idx % len(colors)], width=1, dash="dash"),
			))

		layout = get_plotly_layout(
			title="UC vs Consumer Reward Comparison",
			height=400, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Step"), yaxis=dict(title="Reward"))
		fig.update_layout(**layout)
		return fig

	def _build_summary_table(episodes, labels):
		"""构建对比汇总表"""
		rows = ["| Metric | " + " | ".join(labels) + " |"]
		rows.append("|---|" + "|".join(["---"] * len(labels)) + "|")

		# 总奖励
		row = "| Total Reward |"
		for snaps in episodes:
			total = sum(s.get("step_reward", 0.0) for s in snaps)
			row += f" {total:.4f} |"
		rows.append(row)

		# 平均电压
		row = "| Avg V_mean |"
		for snaps in episodes:
			v_means = [s.get("voltage_summary", {}).get("v_mean", 1.0) for s in snaps]
			row += f" {np.mean(v_means):.4f} |"
		rows.append(row)

		# UC 总奖励
		row = "| UC Total Reward |"
		for snaps in episodes:
			uc_total = sum(s.get("uc_reward", 0.0) for s in snaps)
			row += f" {uc_total:.4f} |"
		rows.append(row)

		# Episode 长度
		row = "| Episode Length |"
		for snaps in episodes:
			row += f" {len(snaps)} |"
		rows.append(row)

		return "\n".join(rows)

	compare_btn.click(
		_compare,
		inputs=ep_paths,
		outputs=[reward_compare_plot, voltage_compare_plot, price_compare_plot, game_compare_plot, summary_table],
	)
