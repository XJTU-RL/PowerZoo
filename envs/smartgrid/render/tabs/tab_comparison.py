# -*- coding: utf-8 -*-
"""
SmartGrid Tab: Comparison
Episode 对比标签页

支持两个 episode 的并排对比:
奖励曲线、电压分布、损耗趋势、Lambda 轨迹。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.engine.episode_reader import EpisodeData, EpisodeReader
from envs.render_common.engine.episode_recorder import EpisodeRecorder
from envs.render_common.viz.theme import get_plotly_layout, COLORS

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Comparison 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Comparison"):

		gr.Markdown("### Episode Comparison (Side-by-Side)")

		with gr.Row():
			with gr.Column():
				gr.Markdown("**Episode A**")
				source_a = gr.Dropdown(
					label="Source A",
					choices=["Live Inference", "From File"],
					value="Live Inference",
				)
				file_a = gr.Textbox(
					label="File Path A",
					placeholder="Path to .npz file",
					visible=True,
				)

			with gr.Column():
				gr.Markdown("**Episode B**")
				source_b = gr.Dropdown(
					label="Source B",
					choices=["Loaded Recording", "From File"],
					value="From File",
				)
				file_b = gr.Textbox(
					label="File Path B",
					placeholder="Path to .npz file",
					visible=True,
				)

		compare_btn = gr.Button("Compare", variant="primary")
		compare_status = gr.Textbox(
			label="Status",
			interactive=False,
			value="Select two episodes to compare",
		)

		with gr.Row():
			reward_cmp_plot = gr.Plot(label="Cumulative Reward Comparison")
			voltage_cmp_plot = gr.Plot(label="Voltage Distribution Comparison")

		with gr.Row():
			loss_cmp_plot = gr.Plot(label="Power Loss Trend Comparison")
			lambda_cmp_plot = gr.Plot(label="Lambda Trajectory Comparison")

		summary_table = gr.Dataframe(
			label="Summary Comparison",
			headers=["Metric", "Episode A", "Episode B", "Delta"],
			interactive=False,
		)

	# --- 回调 ---

	def _compare(src_a: str, path_a: str, src_b: str, path_b: str):
		"""执行对比"""
		snaps_a = _load_snapshots(src_a, path_a, shared_states)
		snaps_b = _load_snapshots(src_b, path_b, shared_states)

		if not snaps_a or not snaps_b:
			msg = "Error: Cannot load one or both episodes"
			return msg, None, None, None, None, []

		# 奖励对比
		reward_fig = _compare_rewards(snaps_a, snaps_b)

		# 电压分布对比
		voltage_fig = _compare_voltage_dist(snaps_a, snaps_b)

		# 损耗趋势对比
		loss_fig = _compare_losses(snaps_a, snaps_b)

		# Lambda 对比
		lambda_fig = _compare_lambda(snaps_a, snaps_b)

		# 汇总表
		summary = _build_summary_table(snaps_a, snaps_b)

		status = f"Compared: A ({len(snaps_a)} steps) vs B ({len(snaps_b)} steps)"
		return status, reward_fig, voltage_fig, loss_fig, lambda_fig, summary

	# --- 绑定 ---

	compare_btn.click(
		fn=_compare,
		inputs=[source_a, file_a, source_b, file_b],
		outputs=[
			compare_status,
			reward_cmp_plot,
			voltage_cmp_plot,
			loss_cmp_plot,
			lambda_cmp_plot,
			summary_table,
		],
	)

	components["reward_cmp_plot"] = reward_cmp_plot
	components["summary_table"] = summary_table

	return components


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _load_snapshots(
	source: str,
	file_path: str,
	shared_states: Dict[str, Any],
) -> List[Dict[str, Any]]:
	"""根据来源加载快照列表"""
	if source == "Live Inference":
		return shared_states.get("live_snapshots", [])
	elif source == "Loaded Recording":
		ep = shared_states.get("loaded_episode")
		return ep.snapshots if ep else []
	elif source == "From File" and file_path:
		try:
			ep = EpisodeReader.load_episode(file_path)
			return ep.snapshots
		except Exception as e:
			logger.error(f"Failed to load {file_path}: {e}")
			return []
	return []


def _compare_rewards(
	snaps_a: List[Dict[str, Any]],
	snaps_b: List[Dict[str, Any]],
) -> go.Figure:
	"""对比累计奖励曲线"""
	layout = get_plotly_layout(title="Cumulative Reward", height=400, env_name="smartgrid")
	fig = go.Figure(layout=layout)

	for label, snaps, color in [
		("Episode A", snaps_a, COLORS["primary"]),
		("Episode B", snaps_b, COLORS["success"]),
	]:
		cum_rewards = []
		for snap in snaps:
			cr = snap.get("cumulative_reward")
			if cr is not None:
				cum_rewards.append(float(cr))

		if cum_rewards:
			x = [f"Day {i + 1}" for i in range(len(cum_rewards))]
			fig.add_trace(go.Scatter(
				x=x, y=cum_rewards,
				mode="lines", name=label,
				line={"color": color, "width": 2},
			))

	return fig


def _compare_voltage_dist(
	snaps_a: List[Dict[str, Any]],
	snaps_b: List[Dict[str, Any]],
) -> go.Figure:
	"""对比电压分布"""
	layout = get_plotly_layout(title="Voltage Distribution", height=400, env_name="smartgrid")
	fig = go.Figure(layout=layout)

	for label, snaps, color in [
		("Episode A", snaps_a, COLORS["primary"]),
		("Episode B", snaps_b, COLORS["success"]),
	]:
		all_v: List[float] = []
		for snap in snaps:
			buses = snap.get("buses", {})
			for bus_info in buses.values():
				v_pu = bus_info.get("v_mag_pu", [])
				all_v.extend(float(v) for v in v_pu)

		if all_v:
			fig.add_trace(go.Histogram(
				x=all_v, name=label,
				marker_color=color, opacity=0.6,
				nbinsx=50,
			))

	fig.update_layout(barmode="overlay")
	return fig


def _compare_losses(
	snaps_a: List[Dict[str, Any]],
	snaps_b: List[Dict[str, Any]],
) -> go.Figure:
	"""对比损耗趋势"""
	layout = get_plotly_layout(title="Power Loss Trend", height=400, env_name="smartgrid")
	fig = go.Figure(layout=layout)

	for label, snaps, color in [
		("Episode A", snaps_a, COLORS["primary"]),
		("Episode B", snaps_b, COLORS["success"]),
	]:
		losses = []
		for snap in snaps:
			circuit = snap.get("circuit", {})
			loss = circuit.get("total_loss_kw")
			if isinstance(loss, (int, float)):
				losses.append(float(loss))

		if losses:
			x = [f"Day {i + 1}" for i in range(len(losses))]
			fig.add_trace(go.Scatter(
				x=x, y=losses,
				mode="lines", name=label,
				line={"color": color, "width": 1.5},
			))

	return fig


def _compare_lambda(
	snaps_a: List[Dict[str, Any]],
	snaps_b: List[Dict[str, Any]],
) -> go.Figure:
	"""对比 Lambda 轨迹"""
	layout = get_plotly_layout(title="Lambda Trajectory", height=400, env_name="smartgrid")
	fig = go.Figure(layout=layout)

	for label, snaps, color in [
		("Episode A", snaps_a, COLORS["primary"]),
		("Episode B", snaps_b, COLORS["success"]),
	]:
		lambdas: List[float] = []
		for snap in snaps:
			lagrangian = snap.get("lagrangian", {})
			info = snap.get("info", {})
			lmbda = lagrangian.get("lmbda", info.get("lambda"))
			if lmbda is not None:
				lambdas.append(float(lmbda))

		if lambdas:
			x = [f"Day {i + 1}" for i in range(len(lambdas))]
			fig.add_trace(go.Scatter(
				x=x, y=lambdas,
				mode="lines", name=label,
				line={"color": color, "width": 2},
			))

	return fig


def _build_summary_table(
	snaps_a: List[Dict[str, Any]],
	snaps_b: List[Dict[str, Any]],
) -> List[List[str]]:
	"""构建对比汇总表"""
	rows: List[List[str]] = []

	# 步数
	rows.append(["Steps", str(len(snaps_a)), str(len(snaps_b)), "-"])

	# 累计奖励
	r_a = snaps_a[-1].get("cumulative_reward", 0.0) if snaps_a else 0.0
	r_b = snaps_b[-1].get("cumulative_reward", 0.0) if snaps_b else 0.0
	rows.append(["Total Reward", f"{r_a:.4f}", f"{r_b:.4f}", f"{r_b - r_a:+.4f}"])

	# 电压违规率
	for label, snaps in [("A", snaps_a), ("B", snaps_b)]:
		viol = 0
		total = 0
		for snap in snaps:
			buses = snap.get("buses", {})
			for bus_info in buses.values():
				v_pu = bus_info.get("v_mag_pu", [])
				for v in v_pu:
					total += 1
					fv = float(v)
					if fv < 0.95 or fv > 1.05:
						viol += 1
		rate = (viol / total * 100) if total > 0 else 0
		if label == "A":
			rate_a = rate
		else:
			rate_b = rate

	rows.append([
		"V Violation Rate",
		f"{rate_a:.2f}%",
		f"{rate_b:.2f}%",
		f"{rate_b - rate_a:+.2f}%",
	])

	# 平均损耗
	for label, snaps in [("A", snaps_a), ("B", snaps_b)]:
		losses = []
		for snap in snaps:
			circuit = snap.get("circuit", {})
			loss = circuit.get("total_loss_kw")
			if isinstance(loss, (int, float)):
				losses.append(float(loss))
		avg = float(np.mean(losses)) if losses else 0.0
		if label == "A":
			avg_a = avg
		else:
			avg_b = avg

	rows.append([
		"Avg Loss (kW)",
		f"{avg_a:.2f}",
		f"{avg_b:.2f}",
		f"{avg_b - avg_a:+.2f}",
	])

	return rows
