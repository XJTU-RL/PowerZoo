# -*- coding: utf-8 -*-
"""
Tab 5: Comparison - 多 Episode 对比

支持加载多个 episode，叠加奖励曲线、
电压统计对比表、设备行为差异分析。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go

from envs.render_common.engine.episode_reader import EpisodeData, EpisodeReader
from envs.render_common.viz.theme import COLORS, get_plotly_layout

logger = logging.getLogger(__name__)

# 对比用颜色序列
COMPARE_COLORS = [
	"#4F46E5", "#EF4444", "#22C55E", "#F59E0B",
	"#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6",
]


def _load_episodes(
	paths_text: str,
) -> Tuple[List[EpisodeData], List[str], str]:
	"""加载多个 episode 文件。

	Args:
		paths_text: 换行分隔的文件路径

	Returns:
		(episode列表, 标签列表, 状态消息)
	"""
	if not paths_text.strip():
		return [], [], "No paths provided"

	paths = [p.strip() for p in paths_text.strip().split("\n") if p.strip()]
	episodes: List[EpisodeData] = []
	labels: List[str] = []

	for i, path in enumerate(paths):
		try:
			ep = EpisodeReader.load_episode(path)
			episodes.append(ep)
			# 从路径生成简短标签
			label = path.split("/")[-1].replace(".npz", "")
			labels.append(f"[{i}] {label}")
		except Exception as exc:
			logger.warning(f"Failed to load {path}: {exc}")
			labels.append(f"[{i}] FAILED")

	status = f"Loaded {len(episodes)}/{len(paths)} episodes"
	return episodes, labels, status


def _create_reward_comparison(
	episodes: List[EpisodeData],
	labels: List[str],
) -> go.Figure:
	"""创建奖励曲线叠加对比图。"""
	layout = get_plotly_layout(title="Reward Comparison")
	fig = go.Figure(layout=layout)

	for i, (ep, label) in enumerate(zip(episodes, labels)):
		color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
		snaps = ep.snapshots

		steps = [s.get("step", j) for j, s in enumerate(snaps)]
		rewards = [s.get("step_reward", 0.0) for s in snaps]

		# 累计奖励
		cum_rewards = []
		cum = 0.0
		for r in rewards:
			cum += r if isinstance(r, (int, float)) else 0.0
			cum_rewards.append(cum)

		fig.add_trace(go.Scatter(
			x=steps, y=cum_rewards,
			mode="lines",
			name=label,
			line=dict(color=color, width=2),
		))

	fig.update_layout(
		xaxis_title="Step",
		yaxis_title="Cumulative Reward",
	)
	return fig


def _create_voltage_comparison(
	episodes: List[EpisodeData],
	labels: List[str],
) -> go.Figure:
	"""创建电压范围对比图。"""
	layout = get_plotly_layout(title="Voltage Range Comparison")
	fig = go.Figure(layout=layout)

	for i, (ep, label) in enumerate(zip(episodes, labels)):
		color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
		snaps = ep.snapshots

		steps = [s.get("step", j) for j, s in enumerate(snaps)]
		v_mins = [s.get("circuit", {}).get("v_min_pu", 1.0) for s in snaps]
		v_maxs = [s.get("circuit", {}).get("v_max_pu", 1.0) for s in snaps]

		# V min 线
		fig.add_trace(go.Scatter(
			x=steps, y=v_mins,
			mode="lines",
			name=f"{label} V_min",
			line=dict(color=color, width=1, dash="dash"),
			legendgroup=label,
		))
		# V max 线
		fig.add_trace(go.Scatter(
			x=steps, y=v_maxs,
			mode="lines",
			name=f"{label} V_max",
			line=dict(color=color, width=1),
			legendgroup=label,
		))

	# 安全区间
	fig.add_hline(y=0.95, line_dash="dot", line_color=COLORS["warning"], opacity=0.5)
	fig.add_hline(y=1.05, line_dash="dot", line_color=COLORS["warning"], opacity=0.5)

	fig.update_layout(
		xaxis_title="Step",
		yaxis_title="Voltage (pu)",
	)
	return fig


def _create_loss_comparison(
	episodes: List[EpisodeData],
	labels: List[str],
) -> go.Figure:
	"""创建功率损耗对比图。"""
	layout = get_plotly_layout(title="Power Loss Comparison")
	fig = go.Figure(layout=layout)

	for i, (ep, label) in enumerate(zip(episodes, labels)):
		color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
		snaps = ep.snapshots

		steps = [s.get("step", j) for j, s in enumerate(snaps)]
		losses = [s.get("circuit", {}).get("total_loss_kw", 0.0) for s in snaps]

		fig.add_trace(go.Scatter(
			x=steps, y=losses,
			mode="lines+markers",
			name=label,
			line=dict(color=color, width=2),
			marker=dict(size=4),
		))

	fig.update_layout(
		xaxis_title="Step",
		yaxis_title="Power Loss (kW)",
	)
	return fig


def _build_comparison_table(
	episodes: List[EpisodeData],
	labels: List[str],
) -> List[List[str]]:
	"""构建对比统计表。"""
	rows: List[List[str]] = []

	for ep, label in zip(episodes, labels):
		snaps = ep.snapshots
		if not snaps:
			rows.append([label, "0", "-", "-", "-", "-", "-"])
			continue

		n_steps = len(snaps)
		total_reward = ep.total_reward

		all_v_min = [s.get("circuit", {}).get("v_min_pu", 1.0) for s in snaps]
		all_v_max = [s.get("circuit", {}).get("v_max_pu", 1.0) for s in snaps]
		all_losses = [s.get("circuit", {}).get("total_loss_kw", 0.0) for s in snaps]

		# 计算违规步数
		violation_steps = 0
		for s in snaps:
			circuit = s.get("circuit", {})
			if circuit.get("v_min_pu", 1.0) < 0.95 or circuit.get("v_max_pu", 1.0) > 1.05:
				violation_steps += 1

		rows.append([
			label,
			str(n_steps),
			f"{total_reward:.4f}",
			f"{min(all_v_min):.4f}",
			f"{max(all_v_max):.4f}",
			f"{sum(all_losses) / len(all_losses):.2f}",
			f"{violation_steps}/{n_steps}",
		])

	return rows


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Comparison Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	# 本地状态
	state_episodes = gr.State([])
	state_labels = gr.State([])

	# === 数据加载 ===
	with gr.Group():
		gr.Markdown("### Load Episodes for Comparison")
		gr.Markdown("*Enter one file path per line*")
		paths_input = gr.Textbox(
			label="Episode File Paths",
			placeholder="recorded_episodes/ep_001.npz\nrecorded_episodes/ep_002.npz",
			lines=4,
		)
		load_btn = gr.Button("Load & Compare", variant="primary")

	# === 对比统计表 ===
	with gr.Accordion("Comparison Table", open=True):
		compare_table = gr.Dataframe(
			headers=[
				"Episode", "Steps", "Total Reward",
				"V Min", "V Max", "Avg Loss (kW)", "Violation Steps",
			],
			datatype=["str"] * 7,
			interactive=False,
			wrap=True,
		)

	# === 对比图表 ===
	with gr.Row():
		with gr.Column():
			gr.Markdown("### Cumulative Reward")
			reward_plot = gr.Plot(label="Reward Comparison")
		with gr.Column():
			gr.Markdown("### Voltage Range")
			voltage_plot = gr.Plot(label="Voltage Comparison")

	gr.Markdown("### Power Loss")
	loss_plot = gr.Plot(label="Loss Comparison")

	status_box = gr.Textbox(label="Status", interactive=False, lines=1)

	# --- 事件绑定 ---
	def on_load(paths_text):
		episodes, labels, status = _load_episodes(paths_text)
		if not episodes:
			return (
				[], [], [], gr.Plot(), gr.Plot(), gr.Plot(), status,
			)

		table = _build_comparison_table(episodes, labels)
		reward_fig = _create_reward_comparison(episodes, labels)
		voltage_fig = _create_voltage_comparison(episodes, labels)
		loss_fig = _create_loss_comparison(episodes, labels)

		return episodes, labels, table, reward_fig, voltage_fig, loss_fig, status

	load_btn.click(
		fn=on_load,
		inputs=[paths_input],
		outputs=[
			state_episodes, state_labels, compare_table,
			reward_plot, voltage_plot, loss_plot, status_box,
		],
	)

	return {
		"reward_plot": reward_plot,
		"voltage_plot": voltage_plot,
		"loss_plot": loss_plot,
		"status_box": status_box,
	}
