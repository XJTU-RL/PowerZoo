# -*- coding: utf-8 -*-
"""
Tab 3: Playback

回放已保存的 episode 数据，支持逐步和自动播放。
"""

import logging
from typing import Any, Dict, List

import gradio as gr

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Playback 标签页

	Args:
		shared_states: 共享状态字典
	"""
	gr.Markdown("### Episode Playback")

	with gr.Row():
		with gr.Column(scale=1):
			gr.Markdown("#### Controls")
			step_slider = gr.Slider(
				label="Step", minimum=0, maximum=24,
				step=1, value=0,
			)
			with gr.Row():
				prev_btn = gr.Button("Prev", variant="secondary")
				play_btn = gr.Button("Play", variant="primary")
				pause_btn = gr.Button("Pause", variant="secondary")
				next_btn = gr.Button("Next", variant="secondary")
			speed = gr.Slider(
				label="Speed", minimum=0.5, maximum=5.0,
				step=0.5, value=1.0,
			)
			info_md = gr.Markdown("Load episode data from Tab 1")

		with gr.Column(scale=3):
			with gr.Row():
				topology_plot = gr.Plot(label="Topology")
				voltage_plot = gr.Plot(label="Voltage Profile")
			with gr.Row():
				price_plot = gr.Plot(label="Price Signal")
				reward_plot = gr.Plot(label="Reward Breakdown")

	timer = gr.Timer(value=1.0, active=False)

	# ------------------------------------------------------------------
	# 回调
	# ------------------------------------------------------------------

	def _update_view(step_idx: int, snapshots):
		"""更新所有图表到指定步"""
		if not snapshots or not isinstance(snapshots, list):
			return None, None, None, None, f"No episode data loaded"

		idx = int(step_idx)
		idx = max(0, min(idx, len(snapshots) - 1))

		snap = snapshots[idx]
		step = snap.get("step", idx)

		topo_fig = None
		voltage_fig = None
		price_fig = None
		reward_fig = None

		try:
			from envs.stackelberg.render.viz.plotly.topology_graph import create_topology_figure
			topo_fig = create_topology_figure(snap)
		except Exception:
			pass

		try:
			from envs.stackelberg.render.viz.plotly.voltage_profile import create_voltage_profile
			voltage_fig = create_voltage_profile(snap)
		except Exception:
			pass

		try:
			from envs.stackelberg.render.viz.plotly.price_signal_chart import create_price_signal_chart
			price_fig = create_price_signal_chart(snapshots[:idx + 1])
		except Exception:
			pass

		try:
			from envs.stackelberg.render.viz.plotly.reward_breakdown import create_reward_timeseries
			reward_fig = create_reward_timeseries(snapshots[:idx + 1])
		except Exception:
			pass

		info = (
			f"**Step {step}** | "
			f"V_mean={snap.get('voltage_summary', {}).get('v_mean', '-'):.4f} | "
			f"Reward={snap.get('step_reward', 0):.4f}"
			if snap.get("voltage_summary")
			else f"**Step {step}**"
		)

		return topo_fig, voltage_fig, price_fig, reward_fig, info

	def _prev(step_idx, snapshots):
		new_idx = max(0, int(step_idx) - 1)
		results = _update_view(new_idx, snapshots)
		return (new_idx,) + results

	def _next(step_idx, snapshots):
		max_idx = len(snapshots) - 1 if isinstance(snapshots, list) and snapshots else 0
		new_idx = min(max_idx, int(step_idx) + 1)
		results = _update_view(new_idx, snapshots)
		return (new_idx,) + results

	outputs = [topology_plot, voltage_plot, price_plot, reward_plot, info_md]
	slider_outputs = [step_slider] + outputs

	step_slider.change(
		_update_view,
		inputs=[step_slider, shared_states["snapshots"]],
		outputs=outputs,
	)

	prev_btn.click(_prev, inputs=[step_slider, shared_states["snapshots"]], outputs=slider_outputs)
	next_btn.click(_next, inputs=[step_slider, shared_states["snapshots"]], outputs=slider_outputs)

	play_btn.click(lambda: gr.Timer(active=True), outputs=[timer])
	pause_btn.click(lambda: gr.Timer(active=False), outputs=[timer])

	timer.tick(
		_next,
		inputs=[step_slider, shared_states["snapshots"]],
		outputs=slider_outputs,
	)
