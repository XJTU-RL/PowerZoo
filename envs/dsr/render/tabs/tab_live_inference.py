# -*- coding: utf-8 -*-
"""
DSR Tab: Live Inference
实时推理标签页

逐步运行 DSR 环境，支持手动覆盖动作，
实时展示拓扑图、恢复进度和动作掩码状态。

DSR 特色:
- 异构 agent 分组展示 (Switch / PV / Load)
- 动作掩码可视化 (合法=彩色, 非法=灰色)
- 恢复进度条 + 故障线路列表
"""

import logging
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

from envs.dsr.render.engine.episode_runner import DSREpisodeRunner
from envs.dsr.render.engine.manual_override import DSRManualOverride
from envs.dsr.render.viz.plotly.topology_graph import create_topology_graph
from envs.dsr.render.viz.plotly.restoration_progress import create_restoration_progress
from envs.dsr.render.assets.bus_coordinates import get_bus_coordinates

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

		gr.Markdown("### DSR Live Inference (Service Restoration)")

		with gr.Row():
			system_dropdown = gr.Dropdown(
				label="System",
				choices=["13Bus", "123Bus", "8500-Node"],
				value="13Bus",
			)
			seed_input = gr.Number(label="Seed", value=42, precision=0)
			init_btn = gr.Button("Initialize Env", variant="primary")

		with gr.Row():
			step_btn = gr.Button("Step", variant="secondary")
			auto_run_btn = gr.Button("Auto Run (5 steps)", variant="secondary")
			reset_btn = gr.Button("Reset", variant="stop")

		with gr.Row():
			step_display = gr.Textbox(
				label="Current Step",
				value="Step 0 / 20",
				interactive=False,
			)
			reward_display = gr.Textbox(
				label="Cumulative Reward",
				value="0.0000",
				interactive=False,
			)
			restoration_display = gr.Textbox(
				label="Restoration %",
				value="0.0%",
				interactive=False,
			)

		with gr.Row():
			topology_plot = gr.Plot(label="Network Topology (Fault Highlighted)")
			restoration_plot = gr.Plot(label="Restoration Progress")

		gr.Markdown("### Fault & Agent Status")

		with gr.Row():
			fault_list = gr.Textbox(
				label="Active Fault Lines",
				value="No faults",
				interactive=False,
				lines=3,
			)
			action_mask_display = gr.JSON(
				label="Action Masks (per Agent)",
				visible=True,
			)

		gr.Markdown("### Manual Action Override")
		gr.Markdown(
			"Agent types: **Switch** (restore line), "
			"**PV** (power level), **Load** (connect/disconnect)"
		)

		with gr.Row():
			override_agent = gr.Number(label="Agent ID", value=0, precision=0)
			override_action = gr.Number(
				label="Action Index (discrete)",
				value=0,
				precision=0,
			)
			override_btn = gr.Button("Set Override", variant="secondary")
			clear_override_btn = gr.Button("Clear Overrides", variant="secondary")

		override_status = gr.Textbox(
			label="Override Status",
			interactive=False,
			value="No overrides set",
		)

	# --- 回调 ---

	def _initialize(system_name: str, seed: int):
		"""初始化环境"""
		config = {
			"system_name": system_name,
			"max_episode_steps": 20,
		}

		inference_engine = shared_states.get("inference_engine")

		runner = DSREpisodeRunner(
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

		# 恢复进度 (仅初始快照)
		rest_fig = create_restoration_progress([init_snap])

		# 故障列表
		rest_data = init_snap.get("restoration_data", {})
		faults = rest_data.get("fault_lines", [])
		fault_text = "\n".join(faults) if faults else "No faults"

		# 动作掩码
		mask_info = _format_action_masks(init_snap)

		# 初始化手动覆盖
		try:
			override = DSRManualOverride.from_env(runner._env)
			shared_states["manual_override"] = override
		except Exception:
			pass

		rest_pct = rest_data.get("restoration_pct", 0.0)

		return (
			"Step 0 / 20",
			"0.0000",
			f"{rest_pct:.1f}%",
			topo_fig,
			rest_fig,
			fault_text,
			mask_info,
		)

	def _step_once():
		"""执行单步"""
		runner = shared_states.get("live_runner")
		if runner is None:
			return "Not initialized", "0.0000", "0.0%", None, None, "N/A", None

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

		# 可视化
		system_name = shared_states.get("live_system", "13Bus")
		bus_coords = get_bus_coordinates(system_name)
		topo_fig = create_topology_graph(snapshot, bus_coords)

		# 恢复进度
		all_snaps = shared_states.get("live_snapshots", [])
		rest_fig = create_restoration_progress(all_snaps)

		# 故障列表
		rest_data = snapshot.get("restoration_data", {})
		faults = rest_data.get("fault_lines", [])
		fault_text = "\n".join(faults) if faults else "No faults"

		# 动作掩码
		mask_info = _format_action_masks(snapshot)

		rest_pct = rest_data.get("restoration_pct", 0.0)
		done = snapshot.get("done", False)
		max_steps = 20
		step_label = f"Step {step} / {max_steps}" + (" [DONE]" if done else "")

		return (
			step_label,
			f"{cum_reward:.4f}",
			f"{rest_pct:.1f}%",
			topo_fig,
			rest_fig,
			fault_text,
			mask_info,
		)

	def _auto_run_5():
		"""自动运行 5 步"""
		results = None
		for _ in range(5):
			results = _step_once()
			if results and "DONE" in str(results[0]):
				break
		return results if results else ("Error", "0.0", "0.0%", None, None, "N/A", None)

	def _reset():
		"""重置环境"""
		runner = shared_states.get("live_runner")
		if runner is not None:
			runner.close()
		shared_states.pop("live_runner", None)
		shared_states.pop("live_snapshots", None)
		shared_states.pop("manual_override", None)

		return "Step 0 / 20 [Reset]", "0.0000", "0.0%", None, None, "No faults", None

	def _set_override(agent_id: int, action_idx: int):
		"""设置动作覆盖 (离散动作)"""
		override = shared_states.get("manual_override")
		if override is None:
			return "Error: Environment not initialized"

		try:
			action_arr = np.array([int(action_idx)], dtype=np.int64)
			override.set_override(int(agent_id), action_arr)
			return f"Override set for agent {int(agent_id)}: action={int(action_idx)}"
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
			step_display, reward_display, restoration_display,
			topology_plot, restoration_plot,
			fault_list, action_mask_display,
		],
	)

	step_btn.click(
		fn=_step_once,
		inputs=[],
		outputs=[
			step_display, reward_display, restoration_display,
			topology_plot, restoration_plot,
			fault_list, action_mask_display,
		],
	)

	auto_run_btn.click(
		fn=_auto_run_5,
		inputs=[],
		outputs=[
			step_display, reward_display, restoration_display,
			topology_plot, restoration_plot,
			fault_list, action_mask_display,
		],
	)

	reset_btn.click(
		fn=_reset,
		inputs=[],
		outputs=[
			step_display, reward_display, restoration_display,
			topology_plot, restoration_plot,
			fault_list, action_mask_display,
		],
	)

	override_btn.click(
		fn=_set_override,
		inputs=[override_agent, override_action],
		outputs=[override_status],
	)

	clear_override_btn.click(
		fn=_clear_overrides,
		inputs=[],
		outputs=[override_status],
	)

	components["topology_plot"] = topology_plot
	components["restoration_plot"] = restoration_plot
	components["step_display"] = step_display

	return components


def _format_action_masks(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	"""格式化动作掩码为可读的 JSON 结构

	Args:
		snapshot: 快照字典

	Returns:
		格式化的动作掩码字典
	"""
	avail = snapshot.get("available_actions")
	agent_types = snapshot.get("agent_types", [])

	if avail is None:
		return None

	result = {}
	if isinstance(avail, list):
		for idx, mask in enumerate(avail):
			agent_type = agent_types[idx] if idx < len(agent_types) else "unknown"
			if isinstance(mask, (list, tuple)):
				n_valid = sum(1 for v in mask if v)
				result[f"agent_{idx} ({agent_type})"] = {
					"valid_actions": n_valid,
					"total_actions": len(mask),
					"mask": [int(v) for v in mask],
				}

	return result
