# -*- coding: utf-8 -*-
"""
Tab 3: Playback - Episode 回放

提供 step slider 逐步回放已录制的 episode，
显示拓扑图、电压热图、奖励曲线。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from envs.render_common.engine.episode_reader import EpisodeData, EpisodeReader
from envs.vvc.render.viz.plotly.topology_graph import create_topology_figure
from envs.vvc.render.viz.plotly.voltage_heatmap import create_voltage_heatmap
from envs.vvc.render.viz.plotly.voltage_profile import create_voltage_profile
from envs.vvc.render.viz.plotly.reward_breakdown import create_reward_breakdown
from envs.vvc.render.viz.plotly.device_schedule_chart import create_device_schedule

logger = logging.getLogger(__name__)


def _load_episode(
	file_path: str,
) -> Tuple[Optional[EpisodeData], List[Dict[str, Any]], str]:
	"""加载 episode 文件。"""
	if not file_path:
		return None, [], "No file path"

	try:
		ep = EpisodeReader.load_episode(file_path)
		snaps = ep.snapshots
		status = (
			f"Loaded: {ep.episode_length} steps | "
			f"Total Reward: {ep.total_reward:.4f}"
		)
		return ep, snaps, status
	except Exception as exc:
		logger.error(f"Load failed: {exc}", exc_info=True)
		return None, [], f"Load failed: {exc}"


def _render_step(
	snapshots: List[Dict[str, Any]],
	step_idx: int,
	bus_coords: Dict,
) -> Tuple[Any, Any, str]:
	"""渲染指定步的可视化。"""
	if not snapshots or step_idx < 0 or step_idx >= len(snapshots):
		return gr.Plot(), gr.Plot(), "Invalid step"

	snap = snapshots[step_idx]

	topo_fig = create_topology_figure(
		bus_data=snap.get("buses", {}),
		line_data=snap.get("lines", {}),
		bus_coords=bus_coords if bus_coords else None,
		device_data=snap.get("devices", {}),
		step=snap.get("step", step_idx),
	)
	vp_fig = create_voltage_profile(snap, step=snap.get("step", step_idx))

	reward = snap.get("step_reward", 0.0)
	cum_reward = snap.get("cumulative_reward", 0.0)
	circuit = snap.get("circuit", {})
	v_min = circuit.get("v_min_pu", 0.0)
	v_max = circuit.get("v_max_pu", 0.0)
	loss = circuit.get("total_loss_kw", 0.0)

	info = (
		f"Step {snap.get('step', step_idx)} | "
		f"Reward: {reward:.4f} | Cum: {cum_reward:.4f} | "
		f"V: [{v_min:.4f}, {v_max:.4f}] | Loss: {loss:.2f} kW"
	)
	return topo_fig, vp_fig, info


def _render_episode_charts(
	snapshots: List[Dict[str, Any]],
) -> Tuple[Any, Any, Any]:
	"""渲染全 episode 图表。"""
	if not snapshots:
		return gr.Plot(), gr.Plot(), gr.Plot()

	heatmap_fig = create_voltage_heatmap(snapshots)
	reward_fig = create_reward_breakdown(snapshots)
	device_fig = create_device_schedule(snapshots)

	return heatmap_fig, reward_fig, device_fig


def _build_summary_md(episode: Optional[EpisodeData]) -> str:
	"""构建 episode 摘要 Markdown。"""
	if episode is None:
		return "*No episode loaded*"

	lines = [
		"### Episode Summary",
		f"- **Steps**: {episode.episode_length}",
		f"- **Total Reward**: {episode.total_reward:.4f}",
	]

	if episode.config_summary:
		cs = episode.config_summary
		lines.append(f"- **System**: {cs.get('system', 'N/A')}")
		lines.append(f"- **Capacitors**: {cs.get('cap_num', 'N/A')}")
		lines.append(f"- **Regulators**: {cs.get('reg_num', 'N/A')}")
		lines.append(f"- **Batteries**: {cs.get('bat_num', 'N/A')}")
		lines.append(f"- **PV Systems**: {cs.get('pv_num', 'N/A')}")

	if episode.metadata:
		meta = episode.metadata
		if "algorithm" in meta:
			lines.append(f"- **Algorithm**: {meta['algorithm']}")
		if "seed" in meta:
			lines.append(f"- **Seed**: {meta['seed']}")

	return "\n".join(lines)


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Playback Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	state_bus_coords = shared_states["bus_coords"]

	# 本地状态
	state_episode = gr.State(None)
	state_snapshots = gr.State([])

	# === 加载区 ===
	with gr.Group():
		gr.Markdown("### Load Episode")
		with gr.Row():
			file_input = gr.Textbox(
				label="Episode File Path",
				placeholder="e.g., recorded_episodes/vvc_episode_001.npz",
				scale=4,
			)
			load_btn = gr.Button("Load", variant="primary", scale=1)

	# === 摘要 + 滑块 ===
	with gr.Row():
		with gr.Column(scale=2):
			summary_md = gr.Markdown("*No episode loaded*")
		with gr.Column(scale=3):
			step_slider = gr.Slider(
				minimum=0, maximum=23, step=1, value=0,
				label="Step",
				interactive=True,
			)

	# === 单步可视化 ===
	with gr.Row():
		with gr.Column(scale=3):
			gr.Markdown("### Network Topology")
			topo_plot = gr.Plot(label="Topology")
		with gr.Column(scale=2):
			gr.Markdown("### Voltage Profile")
			vp_plot = gr.Plot(label="Voltage")

	step_info = gr.Textbox(label="Step Info", interactive=False, lines=1)

	# === 全 episode 图表 ===
	with gr.Row():
		with gr.Column():
			gr.Markdown("### Voltage Heatmap")
			heatmap_plot = gr.Plot(label="Heatmap")
		with gr.Column():
			gr.Markdown("### Reward Breakdown")
			reward_plot = gr.Plot(label="Reward")

	gr.Markdown("### Device Schedule")
	device_plot = gr.Plot(label="Device Schedule")

	status_box = gr.Textbox(label="Status", interactive=False, lines=1)

	# --- 事件绑定 ---
	def on_load(file_path):
		ep, snaps, status = _load_episode(file_path)
		summary = _build_summary_md(ep)

		slider_max = max(len(snaps) - 1, 0)
		slider_update = gr.Slider(maximum=slider_max, value=0)

		heatmap, reward, device = _render_episode_charts(snaps)

		return ep, snaps, summary, slider_update, heatmap, reward, device, status

	def on_step_change(snapshots, step_idx, bus_coords):
		return _render_step(snapshots, int(step_idx), bus_coords)

	load_btn.click(
		fn=on_load,
		inputs=[file_input],
		outputs=[
			state_episode, state_snapshots, summary_md,
			step_slider, heatmap_plot, reward_plot, device_plot, status_box,
		],
	)
	step_slider.change(
		fn=on_step_change,
		inputs=[state_snapshots, step_slider, state_bus_coords],
		outputs=[topo_plot, vp_plot, step_info],
	)

	return {
		"topo_plot": topo_plot,
		"heatmap_plot": heatmap_plot,
		"reward_plot": reward_plot,
		"device_plot": device_plot,
		"status_box": status_box,
		"step_slider": step_slider,
	}
