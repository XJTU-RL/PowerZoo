# -*- coding: utf-8 -*-
"""
Tab 2: Live Inference

实时逐步推理，分离 UC Leader 面板 (5 sliders) 和
Consumer 面板 (N x 3 sliders) 的手动覆盖。
支持自动推进和速度控制。
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)

# 动作范围定义
UC_ACTIONS = [
	("Price Adjustment", -1.0, 1.0, 0.0),
	("DR Signal", -1.0, 1.0, 0.0),
	("ESS Charge", -1.0, 1.0, 0.0),
	("ESS Discharge", -1.0, 1.0, 0.0),
	("Reserve", -1.0, 1.0, 0.0),
]

CONSUMER_ACTIONS = [
	("Load Adjustment", -1.0, 1.0, 0.0),
	("DER Output", -1.0, 1.0, 0.0),
	("Flexibility", -1.0, 1.0, 0.0),
]


class LiveState:
	"""Live inference 运行时状态"""

	def __init__(self):
		self.runner = None
		self.current_step: int = 0
		self.max_steps: int = 24
		self.snapshots: List[Dict[str, Any]] = []
		self.running: bool = False
		self.n_consumers: int = 2

	def reset(self) -> None:
		self.current_step = 0
		self.snapshots.clear()
		self.running = False


def create_tab(shared_states: Dict[str, Any]) -> None:
	"""创建 Live Inference 标签页

	Args:
		shared_states: 共享状态字典
	"""
	live_state = LiveState()

	gr.Markdown("### Live Inference - Stackelberg Game")

	with gr.Row():
		# 左侧：控制面板
		with gr.Column(scale=1):
			gr.Markdown("#### Control")
			with gr.Row():
				reset_btn = gr.Button("Reset", variant="secondary")
				step_btn = gr.Button("Step", variant="primary")
			with gr.Row():
				play_btn = gr.Button("Play", variant="primary")
				pause_btn = gr.Button("Pause", variant="secondary")
			speed_slider = gr.Slider(
				label="Speed (steps/sec)", minimum=0.5, maximum=5.0,
				step=0.5, value=1.0,
			)
			step_display = gr.Markdown("**Step: 0 / 24**")

			gr.Markdown("---")
			gr.Markdown("#### UC Leader Manual Override (5D)")
			uc_enable = gr.Checkbox(label="Enable UC Override", value=False)
			uc_sliders = []
			for name, vmin, vmax, default in UC_ACTIONS:
				s = gr.Slider(
					label=f"UC: {name}",
					minimum=vmin, maximum=vmax,
					step=0.01, value=default,
				)
				uc_sliders.append(s)

			gr.Markdown("---")
			gr.Markdown("#### Consumer Override (3D each)")
			consumer_enable = gr.Checkbox(label="Enable Consumer Override", value=False)
			consumer_sliders = []
			for c_idx in range(3):  # 支持最多 3 个 Consumer 的滑块
				with gr.Accordion(f"Consumer {c_idx + 1}", open=False):
					for name, vmin, vmax, default in CONSUMER_ACTIONS:
						s = gr.Slider(
							label=f"C{c_idx + 1}: {name}",
							minimum=vmin, maximum=vmax,
							step=0.01, value=default,
						)
						consumer_sliders.append(s)

		# 右侧：可视化
		with gr.Column(scale=3):
			with gr.Row():
				topology_plot = gr.Plot(label="Topology")
			with gr.Row():
				with gr.Column():
					price_plot = gr.Plot(label="Price Signal")
				with gr.Column():
					reward_plot = gr.Plot(label="Rewards")
			with gr.Row():
				with gr.Column():
					voltage_plot = gr.Plot(label="Voltage Profile")
				with gr.Column():
					game_plot = gr.Plot(label="Leader-Follower Dynamics")
			info_md = gr.Markdown("Awaiting inference...")

	# Timer 组件用于自动推进
	timer = gr.Timer(value=1.0, active=False)

	# ------------------------------------------------------------------
	# 回调函数
	# ------------------------------------------------------------------

	def _do_reset():
		"""重置 episode"""
		live_state.reset()
		return (
			"**Step: 0 / 24**",
			None, None, None, None, None,
			"Episode reset. Click Step or Play to begin.",
		)

	def _do_step(uc_override_enabled, *slider_values):
		"""执行单步推理"""
		live_state.current_step += 1
		step = live_state.current_step

		if step > live_state.max_steps:
			live_state.running = False
			return _make_outputs("Episode complete.")

		# 构建手动覆盖动作
		actions = None
		if uc_override_enabled:
			n_agents = 1 + live_state.n_consumers
			actions = np.zeros((n_agents, 5), dtype=np.float32)
			# UC 动作 (前 5 个 slider)
			for i in range(5):
				actions[0, i] = float(slider_values[i])

		# 如果有 runner 则用 runner 执行
		if live_state.runner is not None:
			try:
				snapshot = live_state.runner.run_step(step, actions=actions)
				live_state.snapshots.append(snapshot)
			except Exception as exc:
				logger.error(f"Step {step} failed: {exc}")
				return _make_outputs(f"Step {step} error: {exc}")
		else:
			# 模拟模式
			snapshot = _mock_snapshot(step)
			live_state.snapshots.append(snapshot)

		return _make_outputs(f"Step {step} / {live_state.max_steps}")

	def _make_outputs(status_msg: str):
		"""根据当前快照生成所有输出"""
		step = live_state.current_step
		snaps = live_state.snapshots

		step_md = f"**Step: {step} / {live_state.max_steps}**"

		# 生成图表
		topo_fig = _try_chart("topology", snaps)
		price_fig = _try_chart("price_signal", snaps)
		reward_fig = _try_chart("reward", snaps)
		voltage_fig = _try_chart("voltage", snaps)
		game_fig = _try_chart("leader_follower", snaps)

		return (
			step_md,
			topo_fig, price_fig, reward_fig, voltage_fig, game_fig,
			status_msg,
		)

	def _try_chart(chart_type: str, snaps: list):
		"""安全生成图表"""
		if not snaps:
			return None

		try:
			if chart_type == "topology":
				from envs.stackelberg.render.viz.plotly.topology_graph import create_topology_figure
				return create_topology_figure(snaps[-1])
			elif chart_type == "price_signal":
				from envs.stackelberg.render.viz.plotly.price_signal_chart import create_price_signal_chart
				return create_price_signal_chart(snaps)
			elif chart_type == "reward":
				from envs.stackelberg.render.viz.plotly.reward_breakdown import create_reward_timeseries
				return create_reward_timeseries(snaps)
			elif chart_type == "voltage":
				from envs.stackelberg.render.viz.plotly.voltage_profile import create_voltage_profile
				return create_voltage_profile(snaps[-1])
			elif chart_type == "leader_follower":
				from envs.stackelberg.render.viz.plotly.leader_follower_chart import create_leader_follower_chart
				return create_leader_follower_chart(snaps)
		except Exception as exc:
			logger.debug(f"Chart {chart_type} failed: {exc}")

		return None

	def _mock_snapshot(step: int) -> Dict[str, Any]:
		"""生成模拟快照 (无实际环境时)"""
		hour = step % 24
		return {
			"step": step,
			"voltage_summary": {
				"v_min": 0.95 + np.random.uniform(-0.02, 0.02),
				"v_mean": 1.0 + np.random.uniform(-0.01, 0.01),
				"v_max": 1.05 + np.random.uniform(-0.02, 0.02),
			},
			"market_data": {
				"hour": hour,
				"tou_base_price": [0.04, 0.08, 0.15][
					0 if hour < 7 or hour >= 22
					else (2 if 11 <= hour < 17 else 1)
				],
				"uc_actions": {
					"price": np.random.uniform(-0.5, 0.5),
					"DR_signal": np.random.uniform(-1, 1),
					"effective_price": 0.08 + np.random.uniform(-0.03, 0.03),
					"dr_signal_value": np.random.uniform(-0.5, 0.5),
				},
				"consumer_actions": [
					{
						"load_adjustment": np.random.uniform(-0.5, 0.5),
						"DER_output": np.random.uniform(0, 1),
						"flexibility": np.random.uniform(0, 1),
					}
					for _ in range(2)
				],
			},
			"uc_reward": np.random.uniform(-2, 0),
			"avg_consumer_reward": np.random.uniform(-1, 0),
			"step_reward": np.random.uniform(-3, 0),
			"actions": [
				[np.random.uniform(-1, 1) for _ in range(5)],
				*[[np.random.uniform(-1, 1) for _ in range(3)] for _ in range(2)],
			],
			"rewards": [
				[np.random.uniform(-2, 0)],
				*[[np.random.uniform(-1, 0)] for _ in range(2)],
			],
			"buses": {},
			"circuit": {},
			"devices": {},
		}

	# 绑定事件
	all_sliders = [uc_enable] + uc_sliders + consumer_sliders
	outputs = [step_display, topology_plot, price_plot, reward_plot, voltage_plot, game_plot, info_md]

	reset_btn.click(_do_reset, outputs=outputs)
	step_btn.click(_do_step, inputs=all_sliders, outputs=outputs)

	def _on_play():
		live_state.running = True
		return gr.Timer(active=True)

	def _on_pause():
		live_state.running = False
		return gr.Timer(active=False)

	play_btn.click(_on_play, outputs=[timer])
	pause_btn.click(_on_pause, outputs=[timer])

	def _auto_step(uc_override_enabled, *slider_values):
		if not live_state.running:
			return [gr.update()] * 7
		return _do_step(uc_override_enabled, *slider_values)

	timer.tick(_auto_step, inputs=all_sliders, outputs=outputs)
