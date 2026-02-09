"""
Analytics Dashboard Tab
Tab 4: 综合分析仪表盘 -- 多维度图表展示 episode 数据。

支持:
- 数据源选择 (Live Session / Loaded Episode)
- 10 个分析图表 (5 行 x 2 列)
- System Summary 数据表
- 一键刷新全部图表
"""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from envs.district_dispatch.render.viz.plotly.device_schedule_chart import (
	create_device_schedule,
)
from envs.district_dispatch.render.viz.plotly.exchange_sankey import (
	create_exchange_sankey,
)
from envs.district_dispatch.render.viz.plotly.power_flow_diagram import (
	create_power_balance_chart,
)
from envs.district_dispatch.render.viz.plotly.reward_breakdown import (
	create_reward_stacked,
)
from envs.district_dispatch.render.viz.plotly.topology_graph import (
	create_topology_figure,
)
from envs.district_dispatch.render.viz.plotly.voltage_heatmap import (
	create_voltage_heatmap,
)
from envs.district_dispatch.render.viz.plotly.voltage_profile import (
	create_voltage_profile,
)
from envs.district_dispatch.render.viz.theme import get_plotly_layout


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


def _compute_summary_table(
	snapshots: List[Dict[str, Any]],
) -> List[List[str]]:
	"""从快照序列计算汇总指标表。

	Args:
		snapshots: 快照列表

	Returns:
		二维列表 [[metric, value], ...]
	"""
	if not snapshots:
		return [["No Data", "N/A"]]

	n_steps = len(snapshots)

	# 累计奖励
	total_reward = 0.0
	for snap in snapshots:
		rewards = snap.get("rewards", np.array([]))
		if hasattr(rewards, "__len__") and len(rewards) > 0:
			total_reward += float(np.sum(rewards))

	# 电压统计
	all_min_v: List[float] = []
	all_max_v: List[float] = []
	violation_count = 0
	total_bus_readings = 0

	for snap in snapshots:
		buses = snap.get("buses", {})
		for bus_data in buses.values():
			vpu_list = bus_data.get("vpu", [])
			for v in vpu_list:
				total_bus_readings += 1
				if v < 0.95 or v > 1.05:
					violation_count += 1
			if vpu_list:
				all_min_v.append(min(vpu_list))
				all_max_v.append(max(vpu_list))

	avg_min_v = sum(all_min_v) / len(all_min_v) if all_min_v else 1.0
	violation_pct = (
		(violation_count / total_bus_readings * 100)
		if total_bus_readings > 0 else 0.0
	)

	# 总损耗
	total_loss = sum(
		snap.get("circuit", {}).get("total_loss_kw", 0.0)
		for snap in snapshots
	)

	# PV 利用率
	total_pv_available = 0.0
	total_pv_actual = 0.0
	for snap in snapshots:
		pv = snap.get("devices", {}).get("pv", {})
		total_pv_available += pv.get("available_kw", 0.0)
		total_pv_actual += pv.get("output_kw", 0.0)
	pv_utilization = (
		(total_pv_actual / total_pv_available * 100)
		if total_pv_available > 0 else 0.0
	)

	return [
		["Episode Length", f"{n_steps} steps"],
		["Total Reward", f"{total_reward:.4f}"],
		["Avg Min Voltage", f"{avg_min_v:.4f} pu"],
		["Voltage Violation %", f"{violation_pct:.2f}%"],
		["Total Loss", f"{total_loss:.2f} kW"],
		["PV Utilization", f"{pv_utilization:.1f}%"],
		["Avg Loss/Step", f"{total_loss / n_steps:.2f} kW"],
	]


def _refresh_analytics(
	data_source: str,
	live_snapshots: Optional[List[Dict[str, Any]]],
	loaded_snapshots: Optional[List[Dict[str, Any]]],
	bus_coords: Optional[Dict[str, Tuple[float, float]]],
) -> Tuple[
	go.Figure, go.Figure, go.Figure, go.Figure,
	go.Figure, go.Figure, List[List[str]],
]:
	"""刷新所有分析图表。

	Args:
		data_source: 数据源选择 ("Live Session" or "Loaded Episode")
		live_snapshots: 实时会话快照
		loaded_snapshots: 已加载的 episode 快照
		bus_coords: 母线坐标字典

	Returns:
		tuple: 6 个图表 + 1 个汇总表
	"""
	# 选择数据源
	if data_source == "Live Session":
		snapshots = live_snapshots
	else:
		snapshots = loaded_snapshots

	if not snapshots or bus_coords is None:
		empty = _empty_figure("No Data Available")
		return (
			empty, empty, empty, empty, empty, empty,
			[["Status", "No data loaded"]],
		)

	# Row 1: Voltage Profile + Voltage Heatmap
	voltage_profile_fig = create_voltage_profile(snapshots[-1], bus_coords)
	voltage_heatmap_fig = create_voltage_heatmap(snapshots)

	# Row 2: Power Balance + Exchange Sankey
	power_balance_fig = create_power_balance_chart(snapshots)
	exchange_sankey_fig = create_exchange_sankey(snapshots[-1])

	# Row 3: Device Schedule (单独一行宽图)
	device_schedule_fig = create_device_schedule(snapshots)

	# Row 4: Reward Breakdown
	reward_stacked_fig = create_reward_stacked(snapshots)

	# Row 5: Summary Table
	summary_data = _compute_summary_table(snapshots)

	return (
		voltage_profile_fig,
		voltage_heatmap_fig,
		power_balance_fig,
		exchange_sankey_fig,
		device_schedule_fig,
		reward_stacked_fig,
		summary_data,
	)


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Analytics Dashboard Tab。

	Args:
		shared_states: 共享状态字典，包含 snapshots, bus_coords 等

	Returns:
		Tab 内关键组件引用字典
	"""
	with gr.Tab("Analytics"):
		# 内部状态: 存储 loaded episode 快照
		loaded_snapshots = gr.State(value=None)

		# === 数据源选择 + 刷新 ===
		with gr.Row():
			data_source = gr.Dropdown(
				choices=["Live Session", "Loaded Episode"],
				value="Live Session",
				label="Data Source",
				scale=2,
			)
			refresh_btn = gr.Button(
				"Refresh All", variant="primary", scale=1,
			)

		# === Row 1: Voltage ===
		with gr.Row():
			voltage_profile_plot = gr.Plot(label="Voltage Profile")
			voltage_heatmap_plot = gr.Plot(label="Voltage Heatmap")

		# === Row 2: Power ===
		with gr.Row():
			power_balance_plot = gr.Plot(label="Power Balance")
			exchange_sankey_plot = gr.Plot(label="Inter-Zone Exchange")

		# === Row 3: Device Schedule ===
		device_schedule_plot = gr.Plot(label="Device Schedule")

		# === Row 4: Reward ===
		reward_stacked_plot = gr.Plot(label="Reward Breakdown")

		# === Row 5: Summary Table ===
		summary_table = gr.Dataframe(
			headers=["Metric", "Value"],
			datatype=["str", "str"],
			label="System Summary",
			interactive=False,
		)

		# === Event: Refresh ===
		refresh_btn.click(
			fn=_refresh_analytics,
			inputs=[
				data_source,
				shared_states["snapshots"],
				loaded_snapshots,
				shared_states["bus_coords"],
			],
			outputs=[
				voltage_profile_plot,
				voltage_heatmap_plot,
				power_balance_plot,
				exchange_sankey_plot,
				device_schedule_plot,
				reward_stacked_plot,
				summary_table,
			],
		)

	return {
		"data_source": data_source,
		"refresh_btn": refresh_btn,
		"voltage_profile_plot": voltage_profile_plot,
		"voltage_heatmap_plot": voltage_heatmap_plot,
		"power_balance_plot": power_balance_plot,
		"exchange_sankey_plot": exchange_sankey_plot,
		"device_schedule_plot": device_schedule_plot,
		"reward_stacked_plot": reward_stacked_plot,
		"summary_table": summary_table,
		"loaded_snapshots": loaded_snapshots,
	}
