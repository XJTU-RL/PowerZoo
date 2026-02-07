# -*- coding: utf-8 -*-
"""
SmartGrid Tab: Live Inference
实时推理标签页

逐步运行 SmartGrid 环境，支持手动覆盖动作，
实时展示拓扑图和电压分布。
"""

import logging
from typing import Any, Dict, Optional

import gradio as gr
import numpy as np

from envs.smartgrid.render.engine.episode_runner import SmartGridEpisodeRunner
from envs.smartgrid.render.engine.manual_override import SmartGridManualOverride
from envs.smartgrid.render.viz.plotly.topology_graph import create_topology_graph
from envs.smartgrid.render.viz.plotly.voltage_profile import create_voltage_profile_at_step
from envs.smartgrid.render.assets.bus_coordinates import get_bus_coordinates

logger = logging.getLogger(__name__)


def create_tab(shared_states: Dict[str, Any]) -> Dict[str, Any]:
	"""创建 Live Inference 标签页

	Args:
		shared_states: 跨标签页共享状态字典

	Returns:
		标签页组件字典
	"""
	components: Dict[str, Any] = {}

	with gr.Tab("Live Inference"):

		gr.Markdown("### SmartGrid Live Inference (360-step annual simulation)")

		with gr.Row():
			system_dropdown = gr.Dropdown(
				label="System",
				choices=["13Bus", "34Bus_PV", "123Bus", "8500-Node"],
				value="34Bus_PV",
			)
			seed_input = gr.Number(label="Seed", value=42, precision=0)
			init_btn = gr.Button("Initialize Env", variant="primary")

		with gr.Row():
			step_btn = gr.Button("Step", variant="secondary")
			auto_run_btn = gr.Button("Auto Run (10 steps)", variant="secondary")
			reset_btn = gr.Button("Reset", variant="stop")

		with gr.Row():
			step_display = gr.Textbox(
				label="Current Step",
				value="Day 0 / 360",
				interactive=False,
			)
			reward_display = gr.Textbox(
				label="Cumulative Reward",
				value="0.0000",
				interactive=False,
			)
			lambda_display = gr.Textbox(
				label="Lambda",
				value="N/A",
				interactive=False,
			)

		with gr.Row():
			topology_plot = gr.Plot(label="Topology Graph")
			voltage_plot = gr.Plot(label="Voltage Profile")

		gr.Markdown("### Manual Action Override")

		with gr.Row():
			override_agent = gr.Number(label="Agent ID", value=0, precision=0)
			override_values = gr.Textbox(
				label="Action Values (comma-separated)",
				placeholder="e.g., 1, 16, 0.5, 0.8",
			)
			override_btn = gr.Button("Set Override", variant="secondary")
			clear_override_btn = gr.Button("Clear Overrides", variant="secondary")

		override_status = gr.Textbox(
			label="Override Status",
			interactive=False,
			value="No overrides set",
		)

		action_labels = gr.JSON(label="Action Labels", visible=False)

	# --- 回调 ---

	def _initialize(system_name: str, seed: int):
		"""初始化环境"""
		config = {
			"system_name": system_name,
			"max_episode_steps": 360,
			"use_cmdp": True,
		}

		inference_engine = shared_states.get("inference_engine")

		runner = SmartGridEpisodeRunner(
			config=config,
			inference_engine=inference_engine,
			collect_snapshots=True,
		)

		shared_states["live_runner"] = runner
		shared_states["live_snapshots"] = []
		shared_states["live_system"] = system_name

		# 初始化环境
		runner._init_env(seed=int(seed))

		# 初始快照
		init_snap = runner._take_snapshot(step=0, actions=None, rewards=None, infos=None)
		shared_states["live_snapshots"].append(init_snap)

		# 可视化
		bus_coords = get_bus_coordinates(system_name)
		topo_fig = create_topology_graph(init_snap, bus_coords)
		volt_fig = create_voltage_profile_at_step(init_snap)

		# 初始化手动覆盖
		try:
			override = SmartGridManualOverride.from_env(runner._env)
			shared_states["manual_override"] = override
			labels = override.get_action_labels(0)
		except Exception:
			labels = []

		return (
			"Day 0 / 360",
			"0.0000",
			"N/A",
			topo_fig,
			volt_fig,
			labels,
		)

	def _step_once():
		"""执行单步"""
		runner = shared_states.get("live_runner")
		if runner is None:
			return "Not initialized", "0.0000", "N/A", None, None

		override = shared_states.get("manual_override")
		step = runner._current_step + 1

		# 获取动作
		actions = None
		if override is not None and override.has_overrides():
			actions = override.apply(runner._random_actions())

		snapshot = runner.run_step(step=step, actions=actions)
		shared_states["live_snapshots"].append(snapshot)

		# 提取状态
		cum_reward = snapshot.get("cumulative_reward", 0.0)
		info = snapshot.get("info", {})
		lagrangian = snapshot.get("lagrangian", {})
		lmbda = lagrangian.get("lmbda", info.get("lambda", "N/A"))
		if isinstance(lmbda, float):
			lmbda_str = f"{lmbda:.4f}"
		else:
			lmbda_str = str(lmbda)

		# 可视化
		system_name = shared_states.get("live_system", "34Bus_PV")
		bus_coords = get_bus_coordinates(system_name)
		topo_fig = create_topology_graph(snapshot, bus_coords)
		volt_fig = create_voltage_profile_at_step(snapshot)

		done = snapshot.get("done", False)
		step_label = f"Day {step} / 360" + (" [DONE]" if done else "")

		return step_label, f"{cum_reward:.4f}", lmbda_str, topo_fig, volt_fig

	def _auto_run_10():
		"""自动运行 10 步"""
		results = None
		for _ in range(10):
			results = _step_once()
			if results and "DONE" in str(results[0]):
				break
		return results if results else ("Error", "0.0", "N/A", None, None)

	def _reset():
		"""重置环境"""
		runner = shared_states.get("live_runner")
		if runner is not None:
			runner.close()
		shared_states.pop("live_runner", None)
		shared_states.pop("live_snapshots", None)
		shared_states.pop("manual_override", None)

		return "Day 0 / 360 [Reset]", "0.0000", "N/A", None, None

	def _set_override(agent_id: int, values_str: str):
		"""设置动作覆盖"""
		override = shared_states.get("manual_override")
		if override is None:
			return "Error: Environment not initialized"

		try:
			values = [float(v.strip()) for v in values_str.split(",")]
			action_arr = np.array(values, dtype=np.float32)
			override.set_override(int(agent_id), action_arr)
			return f"Override set for agent {int(agent_id)}: {values}"
		except Exception as e:
			return f"Error: {e}"

	def _clear_overrides():
		"""清除所有覆盖"""
		override = shared_states.get("manual_override")
		if override is not None:
			override.clear_all()
		return "All overrides cleared"

	# --- 绑定 ---

	init_btn.click(
		fn=_initialize,
		inputs=[system_dropdown, seed_input],
		outputs=[
			step_display, reward_display, lambda_display,
			topology_plot, voltage_plot, action_labels,
		],
	)

	step_btn.click(
		fn=_step_once,
		inputs=[],
		outputs=[step_display, reward_display, lambda_display, topology_plot, voltage_plot],
	)

	auto_run_btn.click(
		fn=_auto_run_10,
		inputs=[],
		outputs=[step_display, reward_display, lambda_display, topology_plot, voltage_plot],
	)

	reset_btn.click(
		fn=_reset,
		inputs=[],
		outputs=[step_display, reward_display, lambda_display, topology_plot, voltage_plot],
	)

	override_btn.click(
		fn=_set_override,
		inputs=[override_agent, override_values],
		outputs=[override_status],
	)

	clear_override_btn.click(
		fn=_clear_overrides,
		inputs=[],
		outputs=[override_status],
	)

	components["topology_plot"] = topology_plot
	components["voltage_plot"] = voltage_plot
	components["step_display"] = step_display

	return components
