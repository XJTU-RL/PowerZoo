"""
Multi-Episode Comparison Tab
Tab 5: 多 episode 对比 -- 加载最多 3 个 episode 进行并排分析。

支持:
- 加载 1-3 个 episode 文件 (A=Blue, B=Orange, C=Green)
- 4 个对比图 (累积奖励/最小电压/动作分布/总损耗)
- Summary 对比表
"""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.district_dispatch.render.engine.episode_reader import (
	EpisodeData,
	EpisodeReader,
)
from envs.district_dispatch.render.viz.theme import get_plotly_layout

# Episode 颜色方案
EPISODE_COLORS: List[str] = ["#3B82F6", "#F97316", "#10B981"]
EPISODE_LABELS: List[str] = ["Episode A", "Episode B", "Episode C"]
EPISODE_COLOR_NAMES: List[str] = ["Blue", "Orange", "Green"]


def _empty_figure(title: str = "No Data") -> go.Figure:
	"""创建空白占位图。

	Args:
		title: 图表标题

	Returns:
		go.Figure: 空白图
	"""
	fig = go.Figure()
	fig.update_layout(**get_plotly_layout(title, height=400))
	return fig


def _load_single_episode(
	filepath: str,
) -> Optional[EpisodeData]:
	"""安全加载单个 episode。

	Args:
		filepath: .npz 文件路径

	Returns:
		EpisodeData 或 None (加载失败时)
	"""
	if not filepath or not filepath.strip():
		return None
	try:
		return EpisodeReader.load_episode(filepath.strip())
	except Exception:
		return None


def _extract_cumulative_reward(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""计算累积奖励曲线。

	Args:
		snapshots: 快照列表

	Returns:
		累积奖励列表
	"""
	cumulative: List[float] = []
	total = 0.0
	for snap in snapshots:
		rewards = snap.get("rewards", np.array([]))
		if hasattr(rewards, "__len__") and len(rewards) > 0:
			total += float(np.sum(rewards))
		cumulative.append(total)
	return cumulative


def _extract_min_voltage(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步最小电压序列。

	Args:
		snapshots: 快照列表

	Returns:
		每步最小电压列表
	"""
	result: List[float] = []
	for snap in snapshots:
		buses = snap.get("buses", {})
		min_v = 1.0
		for bus_data in buses.values():
			vpu_list = bus_data.get("vpu", [1.0])
			if vpu_list:
				min_v = min(min_v, min(vpu_list))
		result.append(min_v)
	return result


def _extract_total_loss(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步总损耗序列。

	Args:
		snapshots: 快照列表

	Returns:
		每步总损耗 (kW) 列表
	"""
	return [
		snap.get("circuit", {}).get("total_loss_kw", 0.0)
		for snap in snapshots
	]


def _extract_all_actions(
	snapshots: List[Dict[str, Any]],
) -> np.ndarray:
	"""提取所有步骤的动作值，展平为一维。

	Args:
		snapshots: 快照列表

	Returns:
		np.ndarray: 展平后的动作数组
	"""
	all_actions: List[float] = []
	for snap in snapshots:
		actions = snap.get("actions", np.array([]))
		if hasattr(actions, "flatten"):
			all_actions.extend(actions.flatten().tolist())
		elif hasattr(actions, "__iter__"):
			all_actions.extend([float(a) for a in actions])
	return np.array(all_actions) if all_actions else np.array([0.0])


def _create_cumulative_reward_chart(
	episodes: List[Tuple[str, EpisodeData]],
) -> go.Figure:
	"""创建累积奖励对比图。

	Args:
		episodes: (label, episode_data) 列表

	Returns:
		go.Figure: 叠加折线图
	"""
	fig = go.Figure()

	for i, (label, ep) in enumerate(episodes):
		cumulative = _extract_cumulative_reward(ep.snapshots)
		steps = list(range(len(cumulative)))
		color = EPISODE_COLORS[i % len(EPISODE_COLORS)]

		fig.add_trace(go.Scatter(
			x=steps,
			y=cumulative,
			mode="lines",
			name=label,
			line=dict(color=color, width=2),
			hovertemplate=f"{label}<br>Step: %{{x}}<br>Cumulative Reward: %{{y:.4f}}<extra></extra>",
		))

	layout = get_plotly_layout("Cumulative Reward Comparison", height=400)
	layout.update(
		xaxis=dict(title="Step"),
		yaxis=dict(title="Cumulative Reward"),
		hovermode="x unified",
	)
	fig.update_layout(**layout)
	return fig


def _create_min_voltage_chart(
	episodes: List[Tuple[str, EpisodeData]],
) -> go.Figure:
	"""创建最小电压对比图。

	Args:
		episodes: (label, episode_data) 列表

	Returns:
		go.Figure: 叠加折线图
	"""
	fig = go.Figure()

	for i, (label, ep) in enumerate(episodes):
		min_v = _extract_min_voltage(ep.snapshots)
		steps = list(range(len(min_v)))
		color = EPISODE_COLORS[i % len(EPISODE_COLORS)]

		fig.add_trace(go.Scatter(
			x=steps,
			y=min_v,
			mode="lines",
			name=label,
			line=dict(color=color, width=2),
			hovertemplate=f"{label}<br>Step: %{{x}}<br>V_min: %{{y:.4f}} pu<extra></extra>",
		))

	# 限制线
	fig.add_hline(
		y=0.95, line_dash="dash", line_color="#EF4444",
		annotation_text="V_min limit (0.95)",
		annotation_font_color="#EF4444",
		annotation_font_size=10,
	)

	layout = get_plotly_layout("Min Bus Voltage Comparison", height=400)
	layout.update(
		xaxis=dict(title="Step"),
		yaxis=dict(title="Min Voltage (pu)", range=[0.88, 1.08]),
		hovermode="x unified",
	)
	fig.update_layout(**layout)
	return fig


def _create_action_distribution_chart(
	episodes: List[Tuple[str, EpisodeData]],
) -> go.Figure:
	"""创建动作分布对比图 (Box plot)。

	Args:
		episodes: (label, episode_data) 列表

	Returns:
		go.Figure: Box plot
	"""
	fig = go.Figure()

	for i, (label, ep) in enumerate(episodes):
		actions = _extract_all_actions(ep.snapshots)
		color = EPISODE_COLORS[i % len(EPISODE_COLORS)]

		fig.add_trace(go.Box(
			y=actions,
			name=label,
			marker_color=color,
			boxmean="sd",
			hoverinfo="y+name",
		))

	layout = get_plotly_layout("Action Distribution Comparison", height=400)
	layout.update(
		yaxis=dict(title="Action Value"),
	)
	fig.update_layout(**layout)
	return fig


def _create_total_loss_chart(
	episodes: List[Tuple[str, EpisodeData]],
) -> go.Figure:
	"""创建总损耗对比图。

	Args:
		episodes: (label, episode_data) 列表

	Returns:
		go.Figure: 叠加折线图
	"""
	fig = go.Figure()

	for i, (label, ep) in enumerate(episodes):
		losses = _extract_total_loss(ep.snapshots)
		steps = list(range(len(losses)))
		color = EPISODE_COLORS[i % len(EPISODE_COLORS)]

		fig.add_trace(go.Scatter(
			x=steps,
			y=losses,
			mode="lines",
			name=label,
			line=dict(color=color, width=2),
			hovertemplate=f"{label}<br>Step: %{{x}}<br>Loss: %{{y:.2f}} kW<extra></extra>",
		))

	layout = get_plotly_layout("Total Loss Comparison", height=400)
	layout.update(
		xaxis=dict(title="Step"),
		yaxis=dict(title="Loss (kW)"),
		hovermode="x unified",
	)
	fig.update_layout(**layout)
	return fig


def _compute_comparison_table(
	episodes: List[Tuple[str, EpisodeData]],
) -> List[List[str]]:
	"""计算多 episode 对比汇总表。

	Args:
		episodes: (label, episode_data) 列表

	Returns:
		二维列表 [[metric, ep_a, ep_b, ep_c], ...]
	"""
	headers_row = ["Metric"] + [label for label, _ in episodes]
	rows: List[List[str]] = []

	# Total Reward
	row = ["Total Reward"]
	for _, ep in episodes:
		row.append(f"{ep.total_reward:.4f}")
	rows.append(row)

	# Avg Voltage Min
	row = ["Avg Min Voltage"]
	for _, ep in episodes:
		min_vs = _extract_min_voltage(ep.snapshots)
		avg = sum(min_vs) / len(min_vs) if min_vs else 1.0
		row.append(f"{avg:.4f} pu")
	rows.append(row)

	# Total Loss
	row = ["Total Loss"]
	for _, ep in episodes:
		losses = _extract_total_loss(ep.snapshots)
		total = sum(losses)
		row.append(f"{total:.2f} kW")
	rows.append(row)

	# PV Utilization
	row = ["PV Utilization"]
	for _, ep in episodes:
		total_avail = 0.0
		total_actual = 0.0
		for snap in ep.snapshots:
			pv = snap.get("devices", {}).get("pv", {})
			total_avail += pv.get("available_kw", 0.0)
			total_actual += pv.get("output_kw", 0.0)
		pct = (total_actual / total_avail * 100) if total_avail > 0 else 0.0
		row.append(f"{pct:.1f}%")
	rows.append(row)

	# Voltage Violation %
	row = ["Voltage Violation %"]
	for _, ep in episodes:
		violation = 0
		total_readings = 0
		for snap in ep.snapshots:
			for bus_data in snap.get("buses", {}).values():
				for v in bus_data.get("vpu", []):
					total_readings += 1
					if v < 0.95 or v > 1.05:
						violation += 1
		pct = (violation / total_readings * 100) if total_readings > 0 else 0.0
		row.append(f"{pct:.2f}%")
	rows.append(row)

	# Episode Length
	row = ["Episode Length"]
	for _, ep in episodes:
		row.append(f"{ep.episode_length} steps")
	rows.append(row)

	return rows


def _compare_episodes(
	path_a: str,
	path_b: str,
	path_c: str,
) -> Tuple[
	go.Figure, go.Figure, go.Figure, go.Figure,
	List[List[str]], str,
]:
	"""加载并对比 1-3 个 episode。

	Args:
		path_a: Episode A 文件路径
		path_b: Episode B 文件路径
		path_c: Episode C 文件路径

	Returns:
		tuple: 4 个对比图 + 汇总表 + 状态信息
	"""
	# 加载所有有效 episode
	episodes: List[Tuple[str, EpisodeData]] = []
	errors: List[str] = []

	for path, label in [(path_a, "Episode A"), (path_b, "Episode B"), (path_c, "Episode C")]:
		if not path or not path.strip():
			continue
		ep = _load_single_episode(path)
		if ep is not None and len(ep.snapshots) > 0:
			episodes.append((label, ep))
		elif path.strip():
			errors.append(f"Failed to load {label}: {path}")

	if not episodes:
		empty = _empty_figure("No valid episodes loaded")
		status = "No episodes loaded."
		if errors:
			status += " Errors: " + "; ".join(errors)
		return empty, empty, empty, empty, [["Status", status]], status

	# 生成对比图
	cumulative_fig = _create_cumulative_reward_chart(episodes)
	voltage_fig = _create_min_voltage_chart(episodes)
	action_fig = _create_action_distribution_chart(episodes)
	loss_fig = _create_total_loss_chart(episodes)

	# 汇总表
	summary = _compute_comparison_table(episodes)

	status_parts = [f"Loaded {len(episodes)} episode(s)."]
	if errors:
		status_parts.append("Errors: " + "; ".join(errors))
	status = " ".join(status_parts)

	return cumulative_fig, voltage_fig, action_fig, loss_fig, summary, status


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Multi-Episode Comparison Tab。

	Args:
		shared_states: 共享状态字典

	Returns:
		Tab 内关键组件引用字典
	"""
	with gr.Tab("Comparison"):
		# === Episode 选择区域 ===
		gr.Markdown("### Load Episodes for Comparison (max 3)")

		with gr.Row():
			with gr.Column(scale=1):
				ep_a_path = gr.Textbox(
					label="Episode A (Blue)",
					placeholder="/path/to/episode_a.npz",
				)
			with gr.Column(scale=1):
				ep_b_path = gr.Textbox(
					label="Episode B (Orange)",
					placeholder="/path/to/episode_b.npz",
				)
			with gr.Column(scale=1):
				ep_c_path = gr.Textbox(
					label="Episode C (Green)",
					placeholder="/path/to/episode_c.npz",
				)

		with gr.Row():
			compare_btn = gr.Button(
				"Compare", variant="primary", scale=1,
			)
			status_text = gr.Textbox(
				label="Status", interactive=False, scale=3,
			)

		# === 对比图表 (2x2) ===
		with gr.Row():
			cumulative_reward_plot = gr.Plot(
				label="Cumulative Reward",
			)
			min_voltage_plot = gr.Plot(
				label="Min Bus Voltage",
			)
		with gr.Row():
			action_dist_plot = gr.Plot(
				label="Action Distribution",
			)
			total_loss_plot = gr.Plot(
				label="Total Loss",
			)

		# === Summary 对比表 ===
		summary_table = gr.Dataframe(
			label="Comparison Summary",
			interactive=False,
		)

		# === Event: Compare ===
		compare_btn.click(
			fn=_compare_episodes,
			inputs=[ep_a_path, ep_b_path, ep_c_path],
			outputs=[
				cumulative_reward_plot,
				min_voltage_plot,
				action_dist_plot,
				total_loss_plot,
				summary_table,
				status_text,
			],
		)

	return {
		"ep_a_path": ep_a_path,
		"ep_b_path": ep_b_path,
		"ep_c_path": ep_c_path,
		"compare_btn": compare_btn,
		"cumulative_reward_plot": cumulative_reward_plot,
		"min_voltage_plot": min_voltage_plot,
		"action_dist_plot": action_dist_plot,
		"total_loss_plot": total_loss_plot,
		"summary_table": summary_table,
	}
