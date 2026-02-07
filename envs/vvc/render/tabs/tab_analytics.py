# -*- coding: utf-8 -*-
"""
Tab 4: Analytics - 分析面板

提供电压统计、功率损耗趋势、设备利用率分析、
功率流向图等深度分析可视化。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from envs.render_common.engine.episode_reader import EpisodeData, EpisodeReader
from envs.vvc.render.viz.plotly.voltage_heatmap import create_voltage_heatmap
from envs.vvc.render.viz.plotly.reward_breakdown import create_reward_breakdown
from envs.vvc.render.viz.plotly.device_schedule_chart import create_device_schedule
from envs.vvc.render.viz.plotly.power_flow_diagram import create_power_flow_diagram

logger = logging.getLogger(__name__)


def _compute_voltage_stats(
	snapshots: List[Dict[str, Any]],
) -> List[List[str]]:
	"""计算逐步电压统计表。"""
	rows: List[List[str]] = []

	for snap in snapshots:
		step = snap.get("step", 0)
		circuit = snap.get("circuit", {})
		v_mean = circuit.get("v_mean_pu", 1.0)
		v_min = circuit.get("v_min_pu", 1.0)
		v_max = circuit.get("v_max_pu", 1.0)
		v_std = circuit.get("v_std_pu", 0.0)
		loss = circuit.get("total_loss_kw", 0.0)

		# 计算越限母线数
		buses = snap.get("buses", {})
		n_violation = 0
		for bd in buses.values():
			v_list = bd.get("v_mag_pu", [1.0])
			for v in v_list:
				if v < 0.95 or v > 1.05:
					n_violation += 1
					break

		rows.append([
			str(step),
			f"{v_mean:.4f}",
			f"{v_min:.4f}",
			f"{v_max:.4f}",
			f"{v_std:.4f}" if v_std else "-",
			f"{loss:.2f}",
			str(n_violation),
		])

	return rows


def _compute_device_utilization(
	snapshots: List[Dict[str, Any]],
) -> List[List[str]]:
	"""计算设备利用率统计。"""
	if not snapshots:
		return []

	# 统计电容器开关次数
	cap_switches: Dict[str, int] = {}
	prev_cap_states: Dict[str, bool] = {}

	# 统计调压器分接头变化次数
	reg_changes: Dict[str, int] = {}
	prev_reg_taps: Dict[str, float] = {}

	for snap in snapshots:
		devices = snap.get("devices", {})

		for name, cd in devices.get("capacitors", {}).items():
			is_on = cd.get("is_on", False)
			if name in prev_cap_states and prev_cap_states[name] != is_on:
				cap_switches[name] = cap_switches.get(name, 0) + 1
			prev_cap_states[name] = is_on

		for name, rd in devices.get("regulators", {}).items():
			tap = rd.get("tap", 0)
			if name in prev_reg_taps and prev_reg_taps[name] != tap:
				reg_changes[name] = reg_changes.get(name, 0) + 1
			prev_reg_taps[name] = tap

	rows: List[List[str]] = []

	for name, switches in cap_switches.items():
		rows.append([name, "Capacitor", f"{switches} switches", "-"])

	for name, changes in reg_changes.items():
		rows.append([name, "Regulator", f"{changes} tap changes", "-"])

	# 电池 SOC 范围
	last = snapshots[-1]
	for name, bd in last.get("devices", {}).get("batteries", {}).items():
		soc = bd.get("soc", 0.0)
		rows.append([name, "Battery", f"Final SOC: {soc:.1%}", "-"])

	# PV 平均削减率
	pv_curtail_totals: Dict[str, List[float]] = {}
	for snap in snapshots:
		for name, pd_data in snap.get("devices", {}).get("pvsystems", {}).items():
			curtail = pd_data.get("curtail_pct", 0.0)
			if name not in pv_curtail_totals:
				pv_curtail_totals[name] = []
			pv_curtail_totals[name].append(curtail)

	for name, curtails in pv_curtail_totals.items():
		avg_curtail = sum(curtails) / len(curtails) if curtails else 0.0
		rows.append([name, "PV System", f"Avg Curtail: {avg_curtail:.1f}%", "-"])

	return rows


def _compute_episode_summary(
	snapshots: List[Dict[str, Any]],
) -> str:
	"""生成 episode 分析摘要 Markdown。"""
	if not snapshots:
		return "*No data*"

	n_steps = len(snapshots)

	# 电压统计
	all_v_min = []
	all_v_max = []
	all_losses = []
	total_violations = 0

	for snap in snapshots:
		circuit = snap.get("circuit", {})
		all_v_min.append(circuit.get("v_min_pu", 1.0))
		all_v_max.append(circuit.get("v_max_pu", 1.0))
		all_losses.append(circuit.get("total_loss_kw", 0.0))

		buses = snap.get("buses", {})
		for bd in buses.values():
			for v in bd.get("v_mag_pu", [1.0]):
				if v < 0.95 or v > 1.05:
					total_violations += 1

	# 奖励统计
	rewards = [snap.get("step_reward", 0.0) for snap in snapshots]
	total_reward = sum(rewards)

	lines = [
		"### Episode Analysis Summary",
		"",
		"| Metric | Value |",
		"|--------|-------|",
		f"| Steps | {n_steps} |",
		f"| Total Reward | {total_reward:.4f} |",
		f"| Avg Step Reward | {total_reward / n_steps:.4f} |",
		f"| Global V Min | {min(all_v_min):.4f} pu |",
		f"| Global V Max | {max(all_v_max):.4f} pu |",
		f"| Avg Loss | {sum(all_losses) / len(all_losses):.2f} kW |",
		f"| Total Loss | {sum(all_losses):.2f} kW |",
		f"| Voltage Violations | {total_violations} |",
	]

	return "\n".join(lines)


def create_tab(shared_states: Dict[str, gr.State]) -> Dict[str, Any]:
	"""创建 Analytics Tab 的 UI 布局和事件绑定。

	Args:
		shared_states: 跨 Tab 共享状态字典

	Returns:
		该 Tab 内关键组件的引用字典
	"""
	# 本地状态
	state_snapshots = gr.State([])

	# === 数据加载 ===
	with gr.Group():
		gr.Markdown("### Load Episode for Analysis")
		with gr.Row():
			file_input = gr.Textbox(
				label="Episode File Path",
				placeholder="e.g., recorded_episodes/vvc_episode_001.npz",
				scale=4,
			)
			load_btn = gr.Button("Analyze", variant="primary", scale=1)

	# === 摘要 ===
	summary_md = gr.Markdown("*No episode loaded*")

	# === 电压统计表 ===
	with gr.Accordion("Voltage Statistics", open=True):
		voltage_table = gr.Dataframe(
			headers=[
				"Step", "V Mean", "V Min", "V Max",
				"V Std", "Loss (kW)", "Violations",
			],
			datatype=["str"] * 7,
			interactive=False,
			wrap=True,
		)

	# === 设备利用率 ===
	with gr.Accordion("Device Utilization", open=True):
		device_table = gr.Dataframe(
			headers=["Device", "Type", "Metric", "Detail"],
			datatype=["str"] * 4,
			interactive=False,
			wrap=True,
		)

	# === 图表 ===
	with gr.Row():
		with gr.Column():
			gr.Markdown("### Voltage Heatmap")
			heatmap_plot = gr.Plot(label="Heatmap")
		with gr.Column():
			gr.Markdown("### Reward Breakdown")
			reward_plot = gr.Plot(label="Reward")

	with gr.Row():
		with gr.Column():
			gr.Markdown("### Device Schedule")
			device_plot = gr.Plot(label="Device Schedule")
		with gr.Column():
			gr.Markdown("### Power Flow (Last Step)")
			power_flow_plot = gr.Plot(label="Power Flow")

	status_box = gr.Textbox(label="Status", interactive=False, lines=1)

	# --- 事件绑定 ---
	def on_analyze(file_path):
		if not file_path:
			return [], "*No file*", [], [], gr.Plot(), gr.Plot(), gr.Plot(), gr.Plot(), "No file"

		try:
			ep = EpisodeReader.load_episode(file_path)
			snaps = ep.snapshots

			summary = _compute_episode_summary(snaps)
			v_stats = _compute_voltage_stats(snaps)
			d_util = _compute_device_utilization(snaps)

			heatmap = create_voltage_heatmap(snaps)
			reward = create_reward_breakdown(snaps)
			device = create_device_schedule(snaps)

			pf = gr.Plot()
			if snaps:
				pf = create_power_flow_diagram(snaps[-1])

			status = f"Analyzed: {len(snaps)} steps | Reward: {ep.total_reward:.4f}"
			return snaps, summary, v_stats, d_util, heatmap, reward, device, pf, status

		except Exception as exc:
			logger.error(f"Analysis failed: {exc}", exc_info=True)
			return [], f"*Error: {exc}*", [], [], gr.Plot(), gr.Plot(), gr.Plot(), gr.Plot(), f"Error: {exc}"

	load_btn.click(
		fn=on_analyze,
		inputs=[file_input],
		outputs=[
			state_snapshots, summary_md,
			voltage_table, device_table,
			heatmap_plot, reward_plot, device_plot, power_flow_plot,
			status_box,
		],
	)

	return {
		"heatmap_plot": heatmap_plot,
		"reward_plot": reward_plot,
		"device_plot": device_plot,
		"power_flow_plot": power_flow_plot,
		"status_box": status_box,
	}
