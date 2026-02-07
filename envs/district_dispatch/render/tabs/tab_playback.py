"""
Episode Playback Tab
Tab 3: 录制回放 -- 加载已录制的 episode 并逐步回放。

支持:
- 加载 .npz episode 文件
- 时间滑块逐步浏览
- Prev/Play/Pause/Next 控制
- 拓扑图 + 4 条时间序列图 (带当前步骤竖线标注)
- 自动播放 (gr.Timer 驱动)
"""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.district_dispatch.render.engine.episode_reader import (
	EpisodeData,
	EpisodeReader,
)
from envs.district_dispatch.render.viz.plotly.topology_graph import (
	create_topology_figure,
)
from envs.district_dispatch.render.viz.theme import (
	COLORS,
	get_plotly_layout,
)


def _empty_figure(title: str = "No Data") -> go.Figure:
	"""创建空白占位图。

	Args:
		title: 图表标题

	Returns:
		go.Figure: 空白图
	"""
	fig = go.Figure()
	fig.update_layout(**get_plotly_layout(title, height=300))
	return fig


def _extract_voltage_min_series(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步最小电压标幺值序列。

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


def _extract_total_power_series(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步总负荷功率序列。

	Args:
		snapshots: 快照列表

	Returns:
		每步总负荷功率 (kW) 列表
	"""
	return [
		snap.get("circuit", {}).get("total_load_kw", 0.0)
		for snap in snapshots
	]


def _extract_soc_series(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步储能 SOC 序列。

	Args:
		snapshots: 快照列表

	Returns:
		每步 SOC (0~1) 列表
	"""
	return [
		snap.get("devices", {}).get("storage", {}).get("soc", 0.5)
		for snap in snapshots
	]


def _extract_reward_series(
	snapshots: List[Dict[str, Any]],
) -> List[float]:
	"""提取每步总奖励序列。

	Args:
		snapshots: 快照列表

	Returns:
		每步总奖励列表
	"""
	result: List[float] = []
	for snap in snapshots:
		rewards = snap.get("rewards", np.array([]))
		if hasattr(rewards, "__len__") and len(rewards) > 0:
			result.append(float(np.sum(rewards)))
		else:
			result.append(0.0)
	return result


def _create_timeseries_figure(
	timestamps: List[float],
	values: List[float],
	current_step: int,
	title: str,
	y_label: str,
	line_color: str = "#4F46E5",
	y_range: Optional[List[float]] = None,
) -> go.Figure:
	"""创建带当前步骤竖线标注的时间序列图。

	Args:
		timestamps: 时间戳列表
		values: 数值列表
		current_step: 当前步骤索引
		title: 图表标题
		y_label: Y 轴标签
		line_color: 折线颜色
		y_range: Y 轴范围 [min, max]

	Returns:
		go.Figure: Plotly 时间序列图
	"""
	fig = go.Figure()

	time_labels = [f"{t:.2f}" for t in timestamps]

	fig.add_trace(go.Scatter(
		x=time_labels,
		y=values,
		mode="lines",
		line=dict(color=line_color, width=2),
		name=y_label,
		hovertemplate=f"Time: %{{x}}h<br>{y_label}: %{{y:.4f}}<extra></extra>",
	))

	# 当前步骤竖线
	if 0 <= current_step < len(timestamps):
		current_x = time_labels[current_step]
		fig.add_vline(
			x=current_step,
			line_dash="dash",
			line_color="#F59E0B",
			line_width=2,
			annotation_text=f"Step {current_step}",
			annotation_position="top right",
			annotation_font_color="#F59E0B",
			annotation_font_size=10,
		)

	layout = get_plotly_layout(title=title, height=250)
	layout.update(
		xaxis=dict(title="Time (h)"),
		yaxis=dict(title=y_label),
		margin=dict(l=60, r=20, t=40, b=40),
	)
	if y_range:
		layout["yaxis"]["range"] = y_range
	fig.update_layout(**layout)

	return fig


def _load_episode(
	filepath: str,
) -> Tuple[
	Optional[List[Dict[str, Any]]],
	int,
	str,
	gr.Slider,
]:
	"""加载 episode 文件并返回初始状态。

	Args:
		filepath: .npz 文件路径

	Returns:
		tuple: (snapshots, max_step, info_markdown, slider_update)
	"""
	if not filepath or not filepath.strip():
		return (
			None, 0,
			"**No file specified.**",
			gr.Slider(maximum=0, value=0),
		)

	filepath = filepath.strip()
	try:
		episode: EpisodeData = EpisodeReader.load_episode(filepath)
	except Exception as e:
		return (
			None, 0,
			f"**Error loading episode:** {e}",
			gr.Slider(maximum=0, value=0),
		)

	n_steps = len(episode.snapshots)
	if n_steps == 0:
		return (
			None, 0,
			"**Episode has no snapshots.**",
			gr.Slider(maximum=0, value=0),
		)

	# 构建元数据 Markdown
	meta = episode.metadata
	info_lines = [
		"### Episode Info",
		f"- **Length**: {episode.episode_length} steps",
		f"- **Total Reward**: {episode.total_reward:.4f}",
		f"- **Algorithm**: {meta.get('algorithm', 'N/A')}",
		f"- **Seed**: {meta.get('seed', 'N/A')}",
		f"- **Timestamp**: {meta.get('save_timestamp', 'N/A')}",
	]
	if episode.config_summary:
		info_lines.append("- **Config**: " + str(episode.config_summary)[:200])

	return (
		episode.snapshots,
		n_steps - 1,
		"\n".join(info_lines),
		gr.Slider(maximum=max(0, n_steps - 1), value=0),
	)


def _update_step(
	step_idx: int,
	snapshots: Optional[List[Dict[str, Any]]],
	bus_coords: Optional[Dict[str, Tuple[float, float]]],
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
	"""更新指定步骤的所有图表。

	Args:
		step_idx: 当前步骤索引
		snapshots: 快照列表
		bus_coords: 母线坐标字典

	Returns:
		tuple: (topology_fig, voltage_fig, power_fig, soc_fig, reward_fig)
	"""
	if not snapshots or bus_coords is None:
		empty = _empty_figure("Load an episode first")
		return empty, empty, empty, empty, empty

	step_idx = int(step_idx)
	step_idx = max(0, min(step_idx, len(snapshots) - 1))
	snapshot = snapshots[step_idx]

	# 1. 拓扑图
	topo_fig = create_topology_figure(snapshot, bus_coords)

	# 2. 时间序列数据
	timestamps = [
		snap.get("timestamp_h", i * 0.25)
		for i, snap in enumerate(snapshots)
	]

	# Voltage Min
	voltage_fig = _create_timeseries_figure(
		timestamps,
		_extract_voltage_min_series(snapshots),
		step_idx,
		"Min Bus Voltage",
		"Vpu",
		line_color="#10B981",
		y_range=[0.88, 1.12],
	)

	# Total Power
	power_fig = _create_timeseries_figure(
		timestamps,
		_extract_total_power_series(snapshots),
		step_idx,
		"Total Load Power",
		"kW",
		line_color="#3B82F6",
	)

	# SOC
	soc_fig = _create_timeseries_figure(
		timestamps,
		_extract_soc_series(snapshots),
		step_idx,
		"Storage SOC",
		"SOC",
		line_color="#7C3AED",
		y_range=[0.0, 1.0],
	)

	# Reward
	reward_fig = _create_timeseries_figure(
		timestamps,
		_extract_reward_series(snapshots),
		step_idx,
		"Step Reward",
		"Reward",
		line_color="#F59E0B",
	)

	return topo_fig, voltage_fig, power_fig, soc_fig, reward_fig


def _step_prev(current: int) -> int:
	"""前进到上一步。"""
	return max(0, int(current) - 1)


def _step_next(current: int, snapshots: Optional[List]) -> int:
	"""前进到下一步。"""
	if not snapshots:
		return 0
	return min(int(current) + 1, len(snapshots) - 1)


def _toggle_play(is_playing: bool) -> Tuple[bool, str, gr.Timer]:
	"""切换播放/暂停状态。

	Args:
		is_playing: 当前是否正在播放

	Returns:
		tuple: (new_is_playing, button_label, timer_update)
	"""
	new_state = not is_playing
	label = "Pause" if new_state else "Play"
	return new_state, label, gr.Timer(active=new_state)


def _auto_advance(
	current: int,
	snapshots: Optional[List],
	is_playing: bool,
) -> Tuple[int, bool, str, gr.Timer]:
	"""Timer tick 自动前进。

	Args:
		current: 当前步骤
		snapshots: 快照列表
		is_playing: 是否正在播放

	Returns:
		tuple: (new_step, is_playing, button_label, timer_update)
	"""
	if not is_playing or not snapshots:
		return int(current), is_playing, "Play", gr.Timer(active=False)

	new_step = int(current) + 1
	if new_step >= len(snapshots):
		# 到达末尾，停止播放
		return len(snapshots) - 1, False, "Play", gr.Timer(active=False)

	return new_step, True, "Pause", gr.Timer(active=True)


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Episode Playback Tab。

	Args:
		shared_states: 共享状态字典，包含 snapshots, bus_coords 等

	Returns:
		Tab 内关键组件引用字典
	"""
	with gr.Tab("Episode Playback"):
		# 内部状态
		playback_snapshots = gr.State(value=None)
		is_playing = gr.State(value=False)

		# === 文件选择 ===
		with gr.Row():
			filepath_input = gr.Textbox(
				label="Episode File Path (.npz)",
				placeholder="/path/to/episode.npz",
				scale=4,
			)
			load_btn = gr.Button("Load", variant="primary", scale=1)

		# === Episode 信息 ===
		episode_info = gr.Markdown("Load an episode to begin playback.")

		# === 时间滑块 ===
		step_slider = gr.Slider(
			minimum=0, maximum=95, value=0, step=1,
			label="Timestep",
		)

		# === 控制栏 ===
		with gr.Row():
			prev_btn = gr.Button("Prev", scale=1)
			play_btn = gr.Button("Play", variant="secondary", scale=1)
			next_btn = gr.Button("Next", scale=1)
			speed_dropdown = gr.Dropdown(
				choices=["0.5x", "1x", "2x", "4x"],
				value="1x",
				label="Speed",
				scale=1,
			)

		# Timer (默认不激活)
		playback_timer = gr.Timer(value=1.0, active=False)

		# === 主视图: 拓扑图 ===
		topology_plot = gr.Plot(label="Topology View")

		# === 时间序列图 (2x2) ===
		with gr.Row():
			voltage_plot = gr.Plot(label="Min Bus Voltage")
			power_plot = gr.Plot(label="Total Load Power")
		with gr.Row():
			soc_plot = gr.Plot(label="Storage SOC")
			reward_plot = gr.Plot(label="Step Reward")

		# === Event: Load ===
		load_btn.click(
			fn=_load_episode,
			inputs=[filepath_input],
			outputs=[playback_snapshots, gr.State(), episode_info, step_slider],
		)

		# === Event: Slider Change ===
		step_slider.change(
			fn=_update_step,
			inputs=[step_slider, playback_snapshots, shared_states["bus_coords"]],
			outputs=[topology_plot, voltage_plot, power_plot, soc_plot, reward_plot],
		)

		# === Event: Prev/Next ===
		prev_btn.click(
			fn=_step_prev,
			inputs=[step_slider],
			outputs=[step_slider],
		)
		next_btn.click(
			fn=_step_next,
			inputs=[step_slider, playback_snapshots],
			outputs=[step_slider],
		)

		# === Event: Play/Pause ===
		play_btn.click(
			fn=_toggle_play,
			inputs=[is_playing],
			outputs=[is_playing, play_btn, playback_timer],
		)

		# === Event: Timer Tick ===
		playback_timer.tick(
			fn=_auto_advance,
			inputs=[step_slider, playback_snapshots, is_playing],
			outputs=[step_slider, is_playing, play_btn, playback_timer],
		)

	return {
		"filepath_input": filepath_input,
		"load_btn": load_btn,
		"step_slider": step_slider,
		"topology_plot": topology_plot,
		"voltage_plot": voltage_plot,
		"power_plot": power_plot,
		"soc_plot": soc_plot,
		"reward_plot": reward_plot,
		"playback_snapshots": playback_snapshots,
	}
