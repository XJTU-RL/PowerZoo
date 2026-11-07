# -*- coding: utf-8 -*-
"""
DSR Tab: Playback
Episode 回放标签页

回放已录制或已运行的 episode，支持 0-20 步滑块、
拓扑图、电压热力图、设备时刻表等同步展示。

DSR 特色: 短 episode (10-20 步)，故障恢复过程可视化。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr

from envs.dsr.render.assets.bus_coordinates import get_bus_coordinates
from envs.dsr.render.viz.plotly.topology_graph import create_topology_graph
from envs.dsr.render.viz.plotly.voltage_heatmap import create_voltage_heatmap
from envs.dsr.render.viz.plotly.voltage_profile import create_voltage_profile_at_step
from envs.dsr.render.viz.plotly.device_schedule_chart import create_device_schedule_chart

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Playback 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Playback"):

		gr.Markdown("### Episode Playback (Step 0 ~ 20)")
		gr.Markdown("*DSR episodes are short (10-20 steps): fault injection → service restoration*")

		with gr.Row():
			source_dropdown = gr.Dropdown(
				label="Episode Source",
				choices=["Live Inference", "Loaded Recording"],
				value="Live Inference",
			)
			load_btn = gr.Button("Load Episode", variant="primary")
			episode_info = gr.Textbox(
				label="Episode Info",
				interactive=False,
				value="No episode loaded",
			)

		step_slider = gr.Slider(
			minimum=0,
			maximum=20,
			step=1,
			value=0,
			label="Restoration Step",
			interactive=True,
		)

		with gr.Row():
			topology_plot = gr.Plot(label="Topology at Current Step")
			voltage_profile_plot = gr.Plot(label="Voltage Profile at Current Step")

		with gr.Row():
			voltage_heatmap_plot = gr.Plot(label="Voltage Heatmap (All Steps)")
			device_schedule_plot = gr.Plot(label="Device Schedule (All Steps)")

	# --- 回调 ---

	def _load_episode(source: str):
		"""加载 episode 到回放"""
		if source == "Live Inference":
			snapshots = shared_states.get("live_snapshots", [])
			system_name = shared_states.get("live_system", "13Bus")
		else:
			ep = shared_states.get("loaded_episode")
			if ep is None:
				return "No recording loaded", None, None, None, None, gr.update(maximum=0)
			snapshots = ep.snapshots
			system_name = ep.config_summary.get("system_name", "13Bus")

		if not snapshots:
			return "No snapshots available", None, None, None, None, gr.update(maximum=0)

		shared_states["playback_snapshots"] = snapshots
		shared_states["playback_system"] = system_name
		n_steps = len(snapshots)

		# 全局视图
		bus_coords = get_bus_coordinates(system_name)
		heatmap_fig = create_voltage_heatmap(snapshots)
		device_fig = create_device_schedule_chart(snapshots)

		# 初始帧
		topo_fig = create_topology_graph(snapshots[0], bus_coords)
		volt_fig = create_voltage_profile_at_step(snapshots[0])

		# 恢复概要
		last_rest = snapshots[-1].get("restoration_data", {})
		rest_pct = last_rest.get("restoration_pct", 0.0)
		info_text = (
			f"{system_name}: {n_steps} steps, "
			f"restoration={rest_pct:.1f}%, source={source}"
		)

		return (
			info_text,
			topo_fig,
			volt_fig,
			heatmap_fig,
			device_fig,
			gr.update(maximum=n_steps - 1, value=0),
		)

	def _on_slider_change(step_idx: int):
		"""滑块变化时更新当前帧"""
		snapshots = shared_states.get("playback_snapshots", [])
		system_name = shared_states.get("playback_system", "13Bus")

		if not snapshots:
			return None, None

		idx = max(0, min(int(step_idx), len(snapshots) - 1))
		snap = snapshots[idx]

		bus_coords = get_bus_coordinates(system_name)
		topo_fig = create_topology_graph(snap, bus_coords)
		volt_fig = create_voltage_profile_at_step(snap)

		return topo_fig, volt_fig

	# --- 绑定 ---

	load_btn.click(
		fn=_load_episode,
		inputs=[source_dropdown],
		outputs=[
			episode_info,
			topology_plot,
			voltage_profile_plot,
			voltage_heatmap_plot,
			device_schedule_plot,
			step_slider,
		],
	)

	step_slider.change(
		fn=_on_slider_change,
		inputs=[step_slider],
		outputs=[topology_plot, voltage_profile_plot],
	)

	components["step_slider"] = step_slider
	components["topology_plot"] = topology_plot
	components["voltage_heatmap_plot"] = voltage_heatmap_plot

	return components
