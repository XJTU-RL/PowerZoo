# -*- coding: utf-8 -*-
"""
出版级静态报告图表

生成四种高质量静态图表：
- 电压分布图（所有母线 24h 电压轨迹）
- 功率平衡饼图（Load/PV/Storage/EV/Loss 占比）
- 奖励分解图（6 分量堆叠 per agent）
- 损耗分布图（Line/Transformer/NoLoad 饼图 + 时序）
每个函数返回 (fig, ax) tuple。
所有图表文本使用英文标注。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from envs.district_dispatch.render.assets.bus_coordinates import (
	BUS_TO_ZONE,
	ZONE_BUSES,
)
from envs.district_dispatch.render.utils.color_scales import voltage_to_color
from envs.district_dispatch.render.viz.theme import (
	COLORS,
	ZONE_COLORS,
	ZONE_NAMES,
	apply_matplotlib_theme,
)

logger = logging.getLogger(__name__)


def _get_timesteps(snapshots: List[Dict[str, Any]]) -> np.ndarray:
	"""提取时间步序列（小时）。

	Args:
		snapshots: 快照列表

	Returns:
		np.ndarray: 时间序列 (h)
	"""
	return np.array([
		snap.get("timestamp_h", snap.get("step", i) * 0.25)
		for i, snap in enumerate(snapshots)
	])


def plot_voltage_trajectories(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (14, 7),
) -> Tuple[Figure, plt.Axes]:
	"""绘制所有母线 24h 电压轨迹。

	每条母线一条线，按 Zone 着色。
	虚线标注 0.95/1.05 pu 安全边界。

	Args:
		snapshots: 快照列表
		figsize: 图形尺寸

	Returns:
		Tuple[Figure, Axes]: matplotlib 图形和坐标轴
	"""
	apply_matplotlib_theme()

	fig, ax = plt.subplots(1, 1, figsize=figsize)
	timesteps = _get_timesteps(snapshots)

	# 收集所有母线名
	all_buses: set = set()
	for snap in snapshots:
		buses = snap.get("buses", snap.get("bus_data", {}))
		all_buses.update(buses.keys())

	# 按 Zone 分组绘制
	zone_plotted = {0: False, 1: False, 2: False}

	for bus_name in sorted(all_buses):
		voltages = []
		for snap in snapshots:
			buses = snap.get("buses", snap.get("bus_data", {}))
			v = buses.get(bus_name, {}).get("v_mean", 1.0)
			voltages.append(v)

		zone_id = BUS_TO_ZONE.get(bus_name, -1)
		if 0 <= zone_id < len(ZONE_COLORS):
			color = ZONE_COLORS[zone_id]
		else:
			color = COLORS["text_secondary"]

		label = None
		if 0 <= zone_id <= 2 and not zone_plotted.get(zone_id, True):
			label = ZONE_NAMES[zone_id]
			zone_plotted[zone_id] = True

		ax.plot(
			timesteps, voltages,
			color=color, linewidth=0.8, alpha=0.6, label=label,
		)

	# 安全边界
	ax.axhline(y=0.95, color=COLORS["warning"], linestyle="--", linewidth=1.0, alpha=0.8, label="V_min (0.95)")
	ax.axhline(y=1.05, color=COLORS["warning"], linestyle="--", linewidth=1.0, alpha=0.8, label="V_max (1.05)")

	ax.set_title("Bus Voltage Trajectories (24h)", fontsize=14, fontweight="bold")
	ax.set_xlabel("Time (h)", fontsize=11)
	ax.set_ylabel("Voltage (pu)", fontsize=11)
	ax.set_ylim(0.88, 1.12)
	ax.legend(fontsize=8, loc="lower right", ncol=2)
	ax.grid(True, alpha=0.3)

	fig.tight_layout()
	return fig, ax


def plot_power_balance_pie(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (10, 8),
) -> Tuple[Figure, plt.Axes]:
	"""绘制功率平衡饼图。

	统计 episode 累计的 Load / PV / Storage / EV / Loss 能量占比。

	Args:
		snapshots: 快照列表
		figsize: 图形尺寸

	Returns:
		Tuple[Figure, Axes]: matplotlib 图形和坐标轴
	"""
	apply_matplotlib_theme()

	# 累计各分量 (kWh，假设每步 0.25h)
	dt_h = 0.25
	totals = {
		"Load": 0.0,
		"PV": 0.0,
		"Storage": 0.0,
		"EV": 0.0,
		"Loss": 0.0,
	}

	for snap in snapshots:
		circuit = snap.get("circuit", {})
		totals["Load"] += abs(circuit.get("total_load_kw", 0.0)) * dt_h
		totals["Loss"] += abs(circuit.get("total_loss_kw", 0.0)) * dt_h

		devices = snap.get("devices", {})
		for pv_data in devices.get("pv", {}).values():
			totals["PV"] += abs(pv_data.get("kw_output", 0.0)) * dt_h
		for st_data in devices.get("storage", {}).values():
			kw = st_data.get("kw_output", st_data.get("kw", 0.0))
			totals["Storage"] += abs(kw) * dt_h
		for ev_data in devices.get("ev", {}).values():
			kw = ev_data.get("kw", ev_data.get("kw_demand", 0.0))
			totals["EV"] += abs(kw) * dt_h

	# 过滤零值
	labels = []
	sizes = []
	colors_list = [
		COLORS["text_secondary"], COLORS["warning"],
		COLORS["info"], COLORS["secondary"], COLORS["danger"],
	]
	pie_colors = []

	for i, (label, value) in enumerate(totals.items()):
		if value > 0:
			labels.append(f"{label}\n{value:.0f} kWh")
			sizes.append(value)
			pie_colors.append(colors_list[i])

	fig, ax = plt.subplots(1, 1, figsize=figsize)

	if sizes:
		wedges, texts, autotexts = ax.pie(
			sizes, labels=labels, colors=pie_colors,
			autopct="%1.1f%%", startangle=140,
			textprops={"fontsize": 9, "color": COLORS["text_primary"]},
			pctdistance=0.75, labeldistance=1.12,
			wedgeprops={"edgecolor": COLORS["bg_dark"], "linewidth": 1.5},
		)
		for autotext in autotexts:
			autotext.set_fontsize(8)
			autotext.set_color(COLORS["text_primary"])
	else:
		ax.text(
			0.5, 0.5, "No power data available",
			ha="center", va="center", fontsize=12,
			color=COLORS["text_secondary"],
		)

	ax.set_title("Energy Balance (Episode Total)", fontsize=14, fontweight="bold")

	fig.tight_layout()
	return fig, ax


def plot_reward_decomposition(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (14, 7),
) -> Tuple[Figure, plt.Axes]:
	"""绘制奖励分解堆叠柱状图。

	显示每个 agent 的 6 个奖励分量堆叠。
	分量名从 reward_components 字典的键自动发现。

	Args:
		snapshots: 快照列表
		figsize: 图形尺寸

	Returns:
		Tuple[Figure, Axes]: matplotlib 图形和坐标轴
	"""
	apply_matplotlib_theme()

	# 收集所有分量名
	component_names: List[str] = []
	for snap in snapshots:
		rc = snap.get("reward_components", {})
		for key in rc:
			if key not in component_names:
				component_names.append(key)

	if not component_names:
		# 如果无 reward_components，从 rewards 列表绘制简单柱状图
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		timesteps = _get_timesteps(snapshots)

		# 汇总每步总奖励
		rewards_per_step = []
		for snap in snapshots:
			rw = snap.get("rewards", [])
			rewards_per_step.append(sum(rw) if rw else 0.0)

		ax.bar(
			timesteps, rewards_per_step,
			width=0.2, color=COLORS["primary"], alpha=0.8,
		)
		ax.set_title("Step Rewards (Sum of All Agents)", fontsize=14, fontweight="bold")
		ax.set_xlabel("Time (h)", fontsize=11)
		ax.set_ylabel("Reward", fontsize=11)
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		return fig, ax

	# 按 agent 累计各分量
	# reward_components 格式: {"component_name": [agent0_val, agent1_val, ...]}
	# 或 {"component_name": float} (全局)
	n_agents = 0
	for snap in snapshots:
		rc = snap.get("reward_components", {})
		for key, val in rc.items():
			if isinstance(val, (list, np.ndarray)):
				n_agents = max(n_agents, len(val))
				break

	n_agents = max(n_agents, 1)

	# 累计每个 agent 的各分量
	agent_components: Dict[str, np.ndarray] = {
		name: np.zeros(n_agents) for name in component_names
	}

	for snap in snapshots:
		rc = snap.get("reward_components", {})
		for name in component_names:
			val = rc.get(name, 0.0)
			if isinstance(val, (list, np.ndarray)):
				arr = np.array(val, dtype=float)
				agent_components[name][:len(arr)] += arr
			else:
				agent_components[name] += float(val) / n_agents

	fig, ax = plt.subplots(1, 1, figsize=figsize)
	agent_ids = np.arange(n_agents)

	# 堆叠条形图
	component_colors = [
		COLORS["primary"], COLORS["success"], COLORS["warning"],
		COLORS["danger"], COLORS["info"], COLORS["secondary"],
		"#6B7280", "#F97316", "#06B6D4", "#EC4899",
	]

	bottom_pos = np.zeros(n_agents)
	bottom_neg = np.zeros(n_agents)

	for i, name in enumerate(component_names):
		vals = agent_components[name]
		color = component_colors[i % len(component_colors)]

		pos_vals = np.maximum(vals, 0)
		neg_vals = np.minimum(vals, 0)

		if np.any(pos_vals > 0):
			ax.bar(
				agent_ids, pos_vals, bottom=bottom_pos,
				width=0.6, color=color, label=name, alpha=0.85,
				edgecolor=COLORS["bg_dark"], linewidth=0.5,
			)
			bottom_pos += pos_vals

		if np.any(neg_vals < 0):
			ax.bar(
				agent_ids, neg_vals, bottom=bottom_neg,
				width=0.6, color=color, alpha=0.85,
				edgecolor=COLORS["bg_dark"], linewidth=0.5,
			)
			bottom_neg += neg_vals

	ax.axhline(y=0, color=COLORS["text_secondary"], linewidth=0.8, alpha=0.5)
	ax.set_title("Reward Decomposition by Agent (Episode Total)", fontsize=14, fontweight="bold")
	ax.set_xlabel("Agent ID", fontsize=11)
	ax.set_ylabel("Cumulative Reward", fontsize=11)
	ax.set_xticks(agent_ids)
	ax.set_xticklabels([f"Agent {i}" for i in agent_ids], fontsize=9)
	ax.legend(fontsize=8, loc="best", ncol=2)
	ax.grid(True, alpha=0.3, axis="y")

	fig.tight_layout()
	return fig, ax


def plot_loss_analysis(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (14, 6),
) -> Tuple[Figure, plt.Axes]:
	"""绘制损耗分析图。

	左侧: 损耗类型饼图 (Line/Transformer/NoLoad)
	右侧: 损耗时序曲线

	Args:
		snapshots: 快照列表
		figsize: 图形尺寸

	Returns:
		Tuple[Figure, Tuple[Axes, Axes]]: matplotlib 图形和两个坐标轴
	"""
	apply_matplotlib_theme()

	fig, (ax_pie, ax_line) = plt.subplots(1, 2, figsize=figsize)

	timesteps = _get_timesteps(snapshots)

	# 提取损耗时间序列
	line_losses: List[float] = []
	xfmr_losses: List[float] = []
	total_losses: List[float] = []

	for snap in snapshots:
		# 线路损耗
		lines = snap.get("lines", {})
		line_loss = sum(
			data.get("loss_kw", 0.0) for data in lines.values()
		)
		line_losses.append(line_loss)

		# 变压器损耗
		xfmrs = snap.get("transformers", {})
		xfmr_loss = sum(
			data.get("loss_kw", 0.0) for data in xfmrs.values()
		)
		xfmr_losses.append(xfmr_loss)

		# 总损耗
		circuit = snap.get("circuit", {})
		total_losses.append(circuit.get("total_loss_kw", line_loss + xfmr_loss))

	# 计算空载损耗 (total - line - transformer)
	no_load_losses = [
		max(0.0, t - l - x)
		for t, l, x in zip(total_losses, line_losses, xfmr_losses)
	]

	# 累计损耗 (kWh)
	dt_h = 0.25
	line_total = sum(line_losses) * dt_h
	xfmr_total = sum(xfmr_losses) * dt_h
	noload_total = sum(no_load_losses) * dt_h

	# 左侧饼图
	pie_data = {
		"Line Loss": line_total,
		"Transformer Loss": xfmr_total,
		"No-Load Loss": noload_total,
	}
	pie_colors = [COLORS["warning"], COLORS["danger"], COLORS["text_secondary"]]

	valid_labels = []
	valid_sizes = []
	valid_colors = []
	for (label, val), color in zip(pie_data.items(), pie_colors):
		if val > 0:
			valid_labels.append(f"{label}\n{val:.1f} kWh")
			valid_sizes.append(val)
			valid_colors.append(color)

	if valid_sizes:
		wedges, texts, autotexts = ax_pie.pie(
			valid_sizes, labels=valid_labels, colors=valid_colors,
			autopct="%1.1f%%", startangle=140,
			textprops={"fontsize": 9, "color": COLORS["text_primary"]},
			wedgeprops={"edgecolor": COLORS["bg_dark"], "linewidth": 1.5},
		)
		for at in autotexts:
			at.set_fontsize(8)
			at.set_color(COLORS["text_primary"])
	else:
		ax_pie.text(
			0.5, 0.5, "No loss data",
			ha="center", va="center", fontsize=12,
			color=COLORS["text_secondary"],
		)

	ax_pie.set_title("Loss Distribution (Total)", fontsize=12, fontweight="bold")

	# 右侧时序图
	ax_line.plot(timesteps, total_losses, color=COLORS["danger"], linewidth=1.5, label="Total")
	ax_line.plot(timesteps, line_losses, color=COLORS["warning"], linewidth=1.2, label="Line")
	ax_line.plot(timesteps, xfmr_losses, color=COLORS["secondary"], linewidth=1.2, label="Transformer")
	ax_line.fill_between(
		timesteps, 0, total_losses,
		color=COLORS["danger"], alpha=0.1,
	)

	ax_line.set_title("Loss Over Time", fontsize=12, fontweight="bold")
	ax_line.set_xlabel("Time (h)", fontsize=10)
	ax_line.set_ylabel("Loss (kW)", fontsize=10)
	ax_line.legend(fontsize=8, loc="upper right")
	ax_line.grid(True, alpha=0.3)

	fig.tight_layout()
	return fig, (ax_pie, ax_line)
