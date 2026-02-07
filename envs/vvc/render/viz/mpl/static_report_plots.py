# -*- coding: utf-8 -*-
"""
VVC 静态报告图 (Matplotlib)

为 HTML 报告生成静态图表:
- 电压剖面图 (voltage profile bar chart)
- 奖励曲线 (reward curve)
- 设备状态概要
"""

import io
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from envs.render_common.utils.color_scales import voltage_to_color
from envs.render_common.viz.theme import COLORS, apply_matplotlib_theme

logger = logging.getLogger(__name__)


def render_voltage_profile(
	snapshot: Dict[str, Any],
	figsize: tuple = (10, 4),
) -> plt.Figure:
	"""渲染单步电压剖面图。

	Args:
		snapshot: 快照字典
		figsize: 图像尺寸

	Returns:
		Matplotlib Figure
	"""
	apply_matplotlib_theme()
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	buses = snapshot.get("buses", {})
	if not buses:
		ax.text(0.5, 0.5, "No bus data", ha="center", va="center")
		return fig

	bus_names = sorted(buses.keys())
	avg_voltages = []
	bar_colors = []

	for name in bus_names:
		bd = buses[name]
		v_pu = bd.get("v_mag_pu", [1.0])
		avg_v = sum(v_pu) / len(v_pu) if v_pu else 1.0
		avg_voltages.append(avg_v)
		bar_colors.append(voltage_to_color(avg_v))

	x_pos = np.arange(len(bus_names))
	ax.bar(x_pos, avg_voltages, color=bar_colors, width=0.7, edgecolor="white", linewidth=0.3)

	# 安全区间
	ax.axhline(y=1.05, color=COLORS["warning"], linestyle="--", linewidth=1, alpha=0.7)
	ax.axhline(y=0.95, color=COLORS["warning"], linestyle="--", linewidth=1, alpha=0.7)
	ax.axhline(y=1.00, color=COLORS["text_secondary"], linestyle=":", linewidth=0.8, alpha=0.5)
	ax.axhspan(0.95, 1.05, color=COLORS["success"], alpha=0.05)

	ax.set_xticks(x_pos)
	ax.set_xticklabels(bus_names, rotation=45, ha="right", fontsize=7)
	ax.set_ylabel("Voltage (pu)")
	ax.set_ylim(0.88, 1.12)
	ax.set_title(f"Voltage Profile (Step {snapshot.get('step', '?')})")

	fig.tight_layout()
	return fig


def render_reward_curve(
	snapshots: List[Dict[str, Any]],
	figsize: tuple = (10, 4),
) -> plt.Figure:
	"""渲染奖励曲线。

	Args:
		snapshots: 快照列表
		figsize: 图像尺寸

	Returns:
		Matplotlib Figure
	"""
	apply_matplotlib_theme()
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	steps = []
	step_rewards = []
	cum_rewards = []
	cum = 0.0

	for snap in snapshots:
		step = snap.get("step", 0)
		reward = snap.get("step_reward", 0.0)
		if isinstance(reward, (int, float)):
			steps.append(step)
			step_rewards.append(reward)
			cum += reward
			cum_rewards.append(cum)

	if not steps:
		ax.text(0.5, 0.5, "No reward data", ha="center", va="center")
		return fig

	# 每步奖励
	ax.bar(
		steps, step_rewards,
		color=COLORS["primary"], alpha=0.6,
		width=0.8, label="Step Reward",
	)

	# 累计奖励
	ax2 = ax.twinx()
	ax2.plot(
		steps, cum_rewards,
		color=COLORS["secondary"], linewidth=2,
		label="Cumulative",
	)
	ax2.set_ylabel("Cumulative Reward", color=COLORS["secondary"])

	ax.set_xlabel("Step")
	ax.set_ylabel("Step Reward")
	ax.set_title("Reward Curve")
	ax.legend(loc="upper left")
	ax2.legend(loc="upper right")

	fig.tight_layout()
	return fig


def render_device_summary(
	snapshot: Dict[str, Any],
	figsize: tuple = (10, 3),
) -> plt.Figure:
	"""渲染设备状态概要条形图。

	Args:
		snapshot: 快照字典
		figsize: 图像尺寸

	Returns:
		Matplotlib Figure
	"""
	apply_matplotlib_theme()
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	devices = snapshot.get("devices", {})
	labels: List[str] = []
	values: List[float] = []
	bar_colors: List[str] = []

	# 电容器 ON 数量
	caps = devices.get("capacitors", {})
	n_on = sum(1 for c in caps.values() if c.get("is_on", False))
	labels.append(f"Cap ON ({n_on}/{len(caps)})")
	values.append(n_on)
	bar_colors.append("#22D3EE")

	# 调压器平均 tap
	regs = devices.get("regulators", {})
	if regs:
		avg_tap = sum(r.get("tap", 0) for r in regs.values()) / len(regs)
		labels.append(f"Avg Tap ({avg_tap:.1f})")
		values.append(avg_tap)
		bar_colors.append("#A78BFA")

	# 电池平均 SOC
	bats = devices.get("batteries", {})
	if bats:
		avg_soc = sum(b.get("soc", 0) for b in bats.values()) / len(bats)
		labels.append(f"Avg SOC ({avg_soc:.1%})")
		values.append(avg_soc * 100)
		bar_colors.append("#06B6D4")

	# PV 总输出
	pvs = devices.get("pvsystems", {})
	if pvs:
		total_pv = sum(p.get("kw", 0) for p in pvs.values())
		labels.append(f"PV Total ({total_pv:.0f}kW)")
		values.append(total_pv)
		bar_colors.append("#F59E0B")

	if not labels:
		ax.text(0.5, 0.5, "No device data", ha="center", va="center")
		return fig

	x_pos = np.arange(len(labels))
	ax.barh(x_pos, values, color=bar_colors, height=0.6, edgecolor="white", linewidth=0.3)
	ax.set_yticks(x_pos)
	ax.set_yticklabels(labels)
	ax.set_title(f"Device Summary (Step {snapshot.get('step', '?')})")

	fig.tight_layout()
	return fig


def figure_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
	"""将 Figure 转为 base64 编码的 PNG 字符串。

	Args:
		fig: Matplotlib Figure
		dpi: 分辨率

	Returns:
		base64 编码字符串 (含 data URI 前缀)
	"""
	import base64

	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	buf.seek(0)
	encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
	return f"data:image/png;base64,{encoded}"
