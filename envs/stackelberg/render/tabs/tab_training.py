# -*- coding: utf-8 -*-
"""
Tab 7: Training

训练结果可视化面板。
扫描结果目录、展示训练曲线和汇总统计。
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List

import gradio as gr
import numpy as np

from envs.render_common.viz.theme import get_plotly_layout

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Training 标签页

	Args:
		shared_states: 共享状态字典
	"""
	gr.Markdown("### Training Results Visualization")

	with gr.Row():
		results_dir = gr.Textbox(
			label="Results Directory",
			value="./results",
			placeholder="/path/to/training/results/",
		)
		scan_btn = gr.Button("Scan Results", variant="secondary")

	run_dropdown = gr.Dropdown(
		label="Select Training Run", choices=[], interactive=True,
	)
	load_btn = gr.Button("Load Training Data", variant="primary")

	with gr.Row():
		reward_curve_plot = gr.Plot(label="Training Reward Curve")
		loss_curve_plot = gr.Plot(label="Training Loss Curve")

	with gr.Row():
		uc_reward_plot = gr.Plot(label="UC Leader Reward")
		consumer_reward_plot = gr.Plot(label="Consumer Avg Reward")

	summary_md = gr.Markdown("Scan and select a training run to visualize.")

	# ------------------------------------------------------------------
	# 回调
	# ------------------------------------------------------------------

	def _scan_results(path: str):
		"""扫描训练结果目录"""
		if not os.path.isdir(path):
			return gr.update(choices=["Directory not found"], value=None)

		runs = []
		for item in sorted(os.listdir(path), reverse=True):
			full = os.path.join(path, item)
			if os.path.isdir(full):
				# 检查是否包含训练输出
				has_logs = any(
					os.path.exists(os.path.join(full, f))
					for f in ["logs", "models", "train_logs.json", "progress.csv"]
				)
				if has_logs or "stackelberg" in item.lower():
					runs.append(item)

		if not runs:
			# 列出所有子目录作为候选
			runs = [
				d for d in sorted(os.listdir(path), reverse=True)
				if os.path.isdir(os.path.join(path, d))
			][:20]

		if not runs:
			return gr.update(choices=["No training runs found"], value=None)

		return gr.update(choices=runs, value=runs[0])

	def _load_training(run_name: str, base_dir: str):
		"""加载训练数据并生成图表"""
		if not run_name or run_name in ["No training runs found", "Directory not found"]:
			return None, None, None, None, "No valid run selected."

		run_path = os.path.join(base_dir, run_name)

		# 尝试加载训练日志
		reward_data = _try_load_rewards(run_path)
		loss_data = _try_load_losses(run_path)

		# 生成图表
		reward_fig = _create_reward_curve(reward_data)
		loss_fig = _create_loss_curve(loss_data)

		# UC vs Consumer 分离
		uc_fig = _create_agent_reward_curve(reward_data, "UC Leader", "#F59E0B")
		consumer_fig = _create_agent_reward_curve(reward_data, "Consumer Avg", "#3B82F6")

		# 汇总
		summary = _build_training_summary(reward_data, loss_data, run_name)

		return reward_fig, loss_fig, uc_fig, consumer_fig, summary

	def _try_load_rewards(run_path: str) -> Dict[str, List[float]]:
		"""尝试从各种格式加载奖励数据"""
		data: Dict[str, List[float]] = {"episodes": [], "rewards": []}

		# 尝试 JSON 格式
		for json_file in glob.glob(os.path.join(run_path, "**/*.json"), recursive=True):
			try:
				with open(json_file, "r") as f:
					content = json.load(f)
				if isinstance(content, dict):
					if "episode_rewards" in content:
						data["rewards"] = content["episode_rewards"]
						data["episodes"] = list(range(len(data["rewards"])))
						return data
			except Exception:
				continue

		# 尝试 CSV 格式
		for csv_file in glob.glob(os.path.join(run_path, "**/*.csv"), recursive=True):
			try:
				import csv
				with open(csv_file, "r") as f:
					reader = csv.DictReader(f)
					for row in reader:
						if "reward" in row or "episode_reward" in row:
							r = float(row.get("reward", row.get("episode_reward", 0)))
							data["rewards"].append(r)
				if data["rewards"]:
					data["episodes"] = list(range(len(data["rewards"])))
					return data
			except Exception:
				continue

		return data

	def _try_load_losses(run_path: str) -> Dict[str, List[float]]:
		"""尝试加载损失数据"""
		data: Dict[str, List[float]] = {"steps": [], "losses": []}

		for json_file in glob.glob(os.path.join(run_path, "**/*.json"), recursive=True):
			try:
				with open(json_file, "r") as f:
					content = json.load(f)
				if isinstance(content, dict) and "losses" in content:
					data["losses"] = content["losses"]
					data["steps"] = list(range(len(data["losses"])))
					return data
			except Exception:
				continue

		return data

	def _create_reward_curve(data):
		"""创建奖励曲线"""
		import plotly.graph_objects as go

		fig = go.Figure()
		if data["rewards"]:
			fig.add_trace(go.Scatter(
				x=data["episodes"], y=data["rewards"],
				mode="lines", name="Episode Reward",
				line=dict(color="#F59E0B", width=1),
				opacity=0.5,
			))
			# 移动平均
			if len(data["rewards"]) > 10:
				window = min(50, len(data["rewards"]) // 5)
				ma = np.convolve(data["rewards"], np.ones(window) / window, mode="valid")
				fig.add_trace(go.Scatter(
					x=list(range(window - 1, len(data["rewards"]))),
					y=ma.tolist(),
					mode="lines", name=f"MA({window})",
					line=dict(color="#F59E0B", width=2),
				))

		layout = get_plotly_layout(
			title="Training Reward Curve", height=400, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Episode"), yaxis=dict(title="Reward"))
		fig.update_layout(**layout)
		return fig

	def _create_loss_curve(data):
		"""创建损失曲线"""
		import plotly.graph_objects as go

		fig = go.Figure()
		if data["losses"]:
			fig.add_trace(go.Scatter(
				x=data["steps"], y=data["losses"],
				mode="lines", name="Loss",
				line=dict(color="#EF4444", width=1),
			))

		layout = get_plotly_layout(
			title="Training Loss Curve", height=400, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Step"), yaxis=dict(title="Loss"))
		fig.update_layout(**layout)
		return fig

	def _create_agent_reward_curve(data, agent_name, color):
		"""创建单 agent 奖励曲线"""
		import plotly.graph_objects as go

		fig = go.Figure()
		# 使用总奖励作为代理（需要实际分离的数据）
		if data["rewards"]:
			fig.add_trace(go.Scatter(
				x=data["episodes"], y=data["rewards"],
				mode="lines", name=agent_name,
				line=dict(color=color, width=1.5),
			))

		layout = get_plotly_layout(
			title=f"{agent_name} Reward", height=350, env_name="stackelberg",
		)
		layout.update(xaxis=dict(title="Episode"), yaxis=dict(title="Reward"))
		fig.update_layout(**layout)
		return fig

	def _build_training_summary(reward_data, loss_data, run_name):
		"""构建训练汇总"""
		lines = [f"**Run: {run_name}**\n"]

		if reward_data["rewards"]:
			rewards = reward_data["rewards"]
			lines.append(f"- Episodes: {len(rewards)}")
			lines.append(f"- Mean reward: {np.mean(rewards):.4f}")
			lines.append(f"- Best reward: {max(rewards):.4f}")
			lines.append(f"- Final 10 avg: {np.mean(rewards[-10:]):.4f}")
		else:
			lines.append("- No reward data found")

		if loss_data["losses"]:
			losses = loss_data["losses"]
			lines.append(f"- Loss steps: {len(losses)}")
			lines.append(f"- Final loss: {losses[-1]:.6f}")
		else:
			lines.append("- No loss data found")

		return "\n".join(lines)

	# 绑定事件
	scan_btn.click(_scan_results, inputs=[results_dir], outputs=[run_dropdown])
	load_btn.click(
		_load_training,
		inputs=[run_dropdown, results_dir],
		outputs=[reward_curve_plot, loss_curve_plot, uc_reward_plot, consumer_reward_plot, summary_md],
	)
