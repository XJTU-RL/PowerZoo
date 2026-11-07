# -*- coding: utf-8 -*-
"""
Tab 2: Live Inference - 实时推理和拓扑可视化

驱动 EpisodeRunner 执行逐步推理，实时更新拓扑图、设备面板、
系统指标和奖励分解。支持手动动作覆盖。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.district_dispatch.render.engine.episode_runner import EpisodeRunner
from envs.district_dispatch.render.engine.inference_engine import InferenceEngine
from envs.district_dispatch.render.engine.manual_override import (
	ACTION_LABELS,
	ManualOverride,
)
from envs.district_dispatch.render.viz.plotly.topology_graph import (
	create_topology_figure,
)
from envs.district_dispatch.render.viz.theme import (
	ZONE_COLORS,
	ZONE_NAMES,
	get_plotly_layout,
)

logger = logging.getLogger(__name__)

# 速度预设: label -> Timer interval (seconds)
SPEED_MAP = {
	"0.5x": 2.0,
	"1x": 1.0,
	"2x": 0.5,
	"5x": 0.2,
}


# ------------------------------------------------------------------
# 状态容器
# ------------------------------------------------------------------

class LiveState:
	"""Live Inference 运行时状态容器。

	封装推理循环所需的全部可变状态，避免多个 gr.State 之间的同步问题。
	"""

	def __init__(self):
		self.running: bool = False
		self.paused: bool = False
		self.recording: bool = False
		self.current_step: int = 0
		self.snapshots: List[Dict[str, Any]] = []
		self.episode_runner: Optional[EpisodeRunner] = None
		self.override: Optional[ManualOverride] = None
		self.speed: str = "1x"


# ------------------------------------------------------------------
# 显示格式化辅助函数
# ------------------------------------------------------------------

def _format_device_panel(snapshot: Dict[str, Any]) -> str:
	"""将快照中的设备信息格式化为 Markdown。

	Args:
		snapshot: 单步快照字典

	Returns:
		设备状态 Markdown 字符串
	"""
	districts = snapshot.get("districts", [])
	if not districts:
		return "*No device data*"

	lines = ["### Device Status"]
	for d in districts:
		zone_name = ZONE_NAMES[d["district_id"]] if d["district_id"] < len(ZONE_NAMES) else f"Zone {d['district_id']}"
		lines.append(f"\n**{zone_name}**")
		lines.append(f"- Net Load: {d.get('net_load', 0):.1f} kW")
		lines.append(f"- Exchange In: {d.get('exchange_in_kw', 0):.1f} kW")
		lines.append(f"- Exchange Out: {d.get('exchange_out_kw', 0):.1f} kW")
		lines.append(f"- V range: [{d.get('v_min', 0):.4f}, {d.get('v_max', 0):.4f}] pu")

	return "\n".join(lines)


def _format_system_metrics(snapshot: Dict[str, Any]) -> str:
	"""将快照中的系统级指标格式化为 Markdown。"""
	vs = snapshot.get("voltage_summary", {})
	step = snapshot.get("step", 0)
	lines = ["### System Metrics", f"- **Step**: {step}", f"- **Time**: {step * 0.25:.2f} h"]

	for key in ("v_min", "v_max", "v_mean"):
		val = vs.get(key)
		label = key.replace("_", " ").title()
		lines.append(f"- **{label}**: {val:.4f} pu" if isinstance(val, (int, float)) else f"- **{label}**: -")

	cum_r = snapshot.get("cumulative_reward")
	if cum_r is not None:
		lines.append(f"- **Cumulative Reward**: {cum_r:.4f}")
	return "\n".join(lines)


def _format_reward_breakdown(snapshot: Dict[str, Any]) -> str:
	"""将快照中的奖励分量格式化为 Markdown。"""
	rewards = snapshot.get("rewards")
	if rewards is None:
		return "*No reward data*"

	lines = ["### Reward Breakdown"]
	if isinstance(rewards, (list, np.ndarray)):
		for i, r in enumerate(rewards):
			zone = ZONE_NAMES[i] if i < len(ZONE_NAMES) else f"Agent {i}"
			val = float(r[0]) if isinstance(r, (list, np.ndarray)) and len(r) > 0 else float(r)
			lines.append(f"- **{zone}**: {val:.4f}")

	step_r = snapshot.get("step_reward")
	if step_r is not None:
		lines.append(f"- **Total**: {step_r:.4f}")

	components = snapshot.get("reward_components", {})
	if components:
		lines.append("\n**Components:**")
		for key, val in components.items():
			if isinstance(val, (int, float)):
				lines.append(f"- {key}: {val:.4f}")
	return "\n".join(lines)


def _empty_topology_figure() -> go.Figure:
	"""创建空白拓扑图占位。

	Returns:
		空白 Plotly Figure
	"""
	fig = go.Figure()
	fig.update_layout(**get_plotly_layout(title="Topology (no data)", height=550))
	fig.add_annotation(
		text="Start inference to see topology",
		xref="paper", yref="paper", x=0.5, y=0.5,
		showarrow=False, font=dict(size=16),
	)
	return fig


# ------------------------------------------------------------------
# 核心推理逻辑
# ------------------------------------------------------------------

def _do_step(
	live_state: LiveState,
	bus_coords: Dict,
	inference_engine: Optional[InferenceEngine],
) -> Tuple[go.Figure, str, str, str, LiveState]:
	"""执行单步推理并更新所有面板。

	Args:
		live_state: 运行时状态
		bus_coords: 母线坐标字典
		inference_engine: 推理引擎

	Returns:
		(topology_fig, device_md, metrics_md, reward_md, updated_live_state)
	"""
	if live_state.episode_runner is None:
		return (
			_empty_topology_figure(),
			"*Runner not initialized*",
			"*No metrics*",
			"*No rewards*",
			live_state,
		)

	live_state.current_step += 1
	step = live_state.current_step

	# 执行单步
	snapshot = live_state.episode_runner.run_step(step=step)

	# 检查 episode 结束
	if snapshot.get("done", False):
		live_state.running = False

	# 录制
	if live_state.recording:
		live_state.snapshots.append(snapshot)

	# 拓扑图
	if bus_coords:
		try:
			topo_fig = create_topology_figure(snapshot, bus_coords)
		except Exception as exc:
			logger.warning(f"Topology figure failed: {exc}")
			topo_fig = _empty_topology_figure()
	else:
		topo_fig = _empty_topology_figure()

	# 面板
	device_md = _format_device_panel(snapshot)
	metrics_md = _format_system_metrics(snapshot)
	reward_md = _format_reward_breakdown(snapshot)

	return topo_fig, device_md, metrics_md, reward_md, live_state


# ------------------------------------------------------------------
# Tab 创建入口
# ------------------------------------------------------------------

def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Live Inference Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	state_bus_coords = shared_states["bus_coords"]
	state_inference_engine = shared_states["inference_engine"]
	state_model_loaded = shared_states["model_loaded"]
	state_snapshots = shared_states["snapshots"]

	# 内部状态
	live_state = gr.State(LiveState())

	# === 控制栏 ===
	with gr.Row():
		start_btn = gr.Button("Start", variant="primary")
		pause_btn = gr.Button("Pause")
		step_btn = gr.Button("Step")
		stop_btn = gr.Button("Stop", variant="stop")
		record_btn = gr.Button("Record")
		speed_dropdown = gr.Dropdown(
			choices=["0.5x", "1x", "2x", "5x"],
			value="1x",
			label="Speed",
			scale=1,
		)

	# === 状态栏 ===
	with gr.Row():
		status_md = gr.Markdown("**Status**: Idle | Step: 0 | Time: 0.00h")

	# === 主视图 + 侧边栏 ===
	with gr.Row():
		# 主视图: 拓扑图
		with gr.Column(scale=3):
			topology_plot = gr.Plot(
				label="Network Topology",
				value=_empty_topology_figure(),
			)
		# 侧边栏
		with gr.Column(scale=2):
			device_panel = gr.Markdown("*Waiting for inference...*")
			metrics_panel = gr.Markdown("*No metrics*")

	# === 奖励分解 ===
	with gr.Row():
		reward_panel = gr.Markdown("*No reward data*")

	# === 手动覆盖 ===
	with gr.Accordion("Manual Override", open=False):
		with gr.Row():
			override_enable = gr.Checkbox(label="Enable Override", value=False)
			agent_select = gr.Dropdown(
				choices=[f"Zone {i}" for i in range(3)],
				value="Zone 0",
				label="Agent",
				scale=1,
			)
		with gr.Row():
			# 5 个动作维度的 slider
			action_sliders = []
			action_labels = list(ACTION_LABELS.values())
			for i in range(5):
				label = action_labels[i] if i < len(action_labels) else f"Action {i}"
				slider = gr.Slider(
					minimum=-1.0,
					maximum=1.0,
					step=0.01,
					value=0.0,
					label=label,
				)
				action_sliders.append(slider)
		with gr.Row():
			apply_override_btn = gr.Button("Apply Override")
			reset_override_btn = gr.Button("Reset to Model")

	# === 自动推理 Timer ===
	timer = gr.Timer(interval=1.0, active=False)

	# ------------------------------------------------------------------
	# 回调函数
	# ------------------------------------------------------------------

	def on_start(ls, engine, bus_coords, model_loaded):
		"""启动推理循环。"""
		if not model_loaded or engine is None:
			return ls, "**Status**: Model not loaded. Go to Model & Data tab first.", gr.Timer(active=False)

		# 初始化 EpisodeRunner (使用随机动作兜底)
		from envs.district_dispatch.core.config import DistrictDispatchConfig
		config = DistrictDispatchConfig()

		ls.episode_runner = EpisodeRunner(
			config=config,
			inference_engine=engine,
			collect_snapshots=True,
		)
		# 触发环境初始化
		ls.episode_runner._init_core(seed=None)
		if engine is not None:
			engine.reset_rnn_states()

		ls.running = True
		ls.paused = False
		ls.current_step = 0
		ls.snapshots = []

		return ls, "**Status**: Running | Step: 0 | Time: 0.00h", gr.Timer(active=True)

	def on_pause(ls):
		"""暂停/恢复推理。"""
		if ls.running:
			ls.paused = not ls.paused
			state_str = "Paused" if ls.paused else "Running"
			return ls, f"**Status**: {state_str} | Step: {ls.current_step}", gr.Timer(active=not ls.paused)
		return ls, "**Status**: Not running", gr.Timer(active=False)

	def on_step(ls, bus_coords, engine):
		"""单步推理。"""
		if ls.episode_runner is None:
			return _empty_topology_figure(), "*No runner*", "*No metrics*", "*No rewards*", ls, "**Status**: Not initialized"

		topo, dev, met, rew, ls = _do_step(ls, bus_coords, engine)
		step = ls.current_step
		status_text = f"**Status**: Step {step} | Time: {step * 0.25:.2f}h"
		if not ls.running:
			status_text += " | Episode Done"
		return topo, dev, met, rew, ls, status_text

	def on_stop(ls):
		"""停止推理，释放资源。"""
		if ls.episode_runner is not None:
			ls.episode_runner.close()
			ls.episode_runner = None
		ls.running = False
		ls.paused = False
		ls.current_step = 0
		return ls, "**Status**: Stopped", gr.Timer(active=False), _empty_topology_figure()

	def on_record(ls):
		"""切换录制模式。"""
		ls.recording = not ls.recording
		state_str = "ON" if ls.recording else "OFF"
		return ls, f"**Status**: Recording {state_str} | Snapshots: {len(ls.snapshots)}"

	def on_timer_tick(ls, bus_coords, engine):
		"""Timer 自动推理回调。"""
		if not ls.running or ls.paused or ls.episode_runner is None:
			return _empty_topology_figure(), "*Idle*", "*No metrics*", "*No rewards*", ls, f"**Status**: {'Paused' if ls.paused else 'Idle'}"

		topo, dev, met, rew, ls = _do_step(ls, bus_coords, engine)
		step = ls.current_step
		done_str = " | Episode Done" if not ls.running else ""
		timer_active = ls.running and not ls.paused
		status_text = f"**Status**: Running | Step: {step} | Time: {step * 0.25:.2f}h{done_str}"
		return topo, dev, met, rew, ls, status_text

	def on_speed_change(speed_val, ls):
		"""速度切换回调。"""
		ls.speed = speed_val
		interval = SPEED_MAP.get(speed_val, 1.0)
		return ls, gr.Timer(interval=interval)

	def on_apply_override(ls, enable, agent_str, *slider_vals):
		"""应用手动覆盖。"""
		if ls.episode_runner is None:
			return ls, "**Override**: No runner"

		# 解析 agent index
		agent_idx = 0
		if agent_str:
			try:
				agent_idx = int(agent_str.split()[-1])
			except (ValueError, IndexError):
				pass

		# 创建或更新 override
		if ls.override is None:
			action_dims = [5] * 3  # 3-zone, 5-dim each
			ls.override = ManualOverride(n_agents=3, action_dims=action_dims)

		ls.override.enabled = enable
		if enable:
			for i, val in enumerate(slider_vals):
				if i < 5:
					ls.override.set_override(agent_idx, i, float(val))

		summary = ls.override.get_override_summary()
		return ls, f"**Override**: {summary['total_overrides']} active overrides"

	def on_reset_override(ls):
		"""重置手动覆盖。"""
		if ls.override is not None:
			ls.override.clear_all()
			ls.override.enabled = False
		return ls, "**Override**: Reset to model output"

	# ------------------------------------------------------------------
	# 事件绑定
	# ------------------------------------------------------------------

	# 启动
	start_btn.click(
		fn=on_start,
		inputs=[live_state, state_inference_engine, state_bus_coords, state_model_loaded],
		outputs=[live_state, status_md, timer],
	)

	# 暂停
	pause_btn.click(
		fn=on_pause,
		inputs=[live_state],
		outputs=[live_state, status_md, timer],
	)

	# 单步
	step_btn.click(
		fn=on_step,
		inputs=[live_state, state_bus_coords, state_inference_engine],
		outputs=[topology_plot, device_panel, metrics_panel, reward_panel, live_state, status_md],
	)

	# 停止
	stop_btn.click(
		fn=on_stop,
		inputs=[live_state],
		outputs=[live_state, status_md, timer, topology_plot],
	)

	# 录制
	record_btn.click(
		fn=on_record,
		inputs=[live_state],
		outputs=[live_state, status_md],
	)

	# Timer 驱动自动推理
	timer.tick(
		fn=on_timer_tick,
		inputs=[live_state, state_bus_coords, state_inference_engine],
		outputs=[topology_plot, device_panel, metrics_panel, reward_panel, live_state, status_md],
	)

	# 速度切换
	speed_dropdown.change(
		fn=on_speed_change,
		inputs=[speed_dropdown, live_state],
		outputs=[live_state, timer],
	)

	# 手动覆盖
	apply_override_btn.click(
		fn=on_apply_override,
		inputs=[live_state, override_enable, agent_select] + action_sliders,
		outputs=[live_state, status_md],
	)

	reset_override_btn.click(
		fn=on_reset_override,
		inputs=[live_state],
		outputs=[live_state, status_md],
	)

	return {
		"topology_plot": topology_plot,
		"device_panel": device_panel,
		"metrics_panel": metrics_panel,
		"reward_panel": reward_panel,
		"status_md": status_md,
		"timer": timer,
		"start_btn": start_btn,
		"stop_btn": stop_btn,
	}
