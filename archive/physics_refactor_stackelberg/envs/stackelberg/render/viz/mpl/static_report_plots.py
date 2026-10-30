# -*- coding: utf-8 -*-
"""
静态报告图表

用于 HTML 报告的 Matplotlib 静态图表生成，
包含电压分布、奖励曲线、市场动态等。
"""

import io
import base64
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
	"""将 Matplotlib Figure 转为 base64 PNG 字符串

	Args:
		fig: Matplotlib Figure
		dpi: 输出 DPI

	Returns:
		base64 编码的 PNG 字符串
	"""
	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
	plt.close(fig)
	buf.seek(0)
	return base64.b64encode(buf.read()).decode("utf-8")


def _setup_dark_axes(ax: plt.Axes) -> None:
	"""设置深色主题坐标轴

	Args:
		ax: Matplotlib Axes
	"""
	ax.set_facecolor("#0f0f23")
	ax.tick_params(colors="#a0a0a0", labelsize=8)
	for spine in ax.spines.values():
		spine.set_color("#3a3a5e")
	ax.xaxis.label.set_color("#a0a0a0")
	ax.yaxis.label.set_color("#a0a0a0")
	ax.title.set_color("#e0e0e0")


def generate_voltage_summary_plot(
	snapshots: List[Dict[str, Any]],
) -> str:
	"""生成电压摘要图 (base64 PNG)

	Args:
		snapshots: 快照列表

	Returns:
		base64 PNG 字符串
	"""
	fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
	_setup_dark_axes(ax)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	v_mins = [snap.get("voltage_summary", {}).get("v_min", 1.0) for snap in snapshots]
	v_means = [snap.get("voltage_summary", {}).get("v_mean", 1.0) for snap in snapshots]
	v_maxs = [snap.get("voltage_summary", {}).get("v_max", 1.0) for snap in snapshots]

	ax.fill_between(steps, v_mins, v_maxs, alpha=0.2, color="#F59E0B")
	ax.plot(steps, v_mins, color="#EF4444", linewidth=1, label="V_min")
	ax.plot(steps, v_means, color="#F59E0B", linewidth=2, label="V_mean")
	ax.plot(steps, v_maxs, color="#3B82F6", linewidth=1, label="V_max")

	ax.axhline(y=0.95, color="#EF4444", linestyle="--", alpha=0.5)
	ax.axhline(y=1.05, color="#EF4444", linestyle="--", alpha=0.5)

	ax.set_title("Voltage Summary", fontsize=12)
	ax.set_xlabel("Step")
	ax.set_ylabel("Voltage (pu)")
	ax.legend(
		facecolor="#16213e", edgecolor="#3a3a5e",
		labelcolor="#e0e0e0", fontsize=8,
	)

	return _fig_to_base64(fig)


def generate_reward_curve_plot(
	snapshots: List[Dict[str, Any]],
) -> str:
	"""生成奖励曲线图 (base64 PNG)

	Args:
		snapshots: 快照列表

	Returns:
		base64 PNG 字符串
	"""
	fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a2e")
	_setup_dark_axes(ax)

	steps = []
	uc_rewards = []
	consumer_rewards = []

	for snap in snapshots:
		step = snap.get("step", 0)
		if step == 0:
			continue
		steps.append(step)
		uc_rewards.append(snap.get("uc_reward", 0.0))
		consumer_rewards.append(snap.get("avg_consumer_reward", 0.0))

	ax.bar(steps, uc_rewards, color="#F59E0B", alpha=0.7, label="UC Leader")
	ax.plot(steps, consumer_rewards, color="#3B82F6", linewidth=2, marker="o",
		markersize=4, label="Consumer Avg")

	ax.set_title("UC vs Consumer Rewards", fontsize=12)
	ax.set_xlabel("Step")
	ax.set_ylabel("Reward")
	ax.legend(
		facecolor="#16213e", edgecolor="#3a3a5e",
		labelcolor="#e0e0e0", fontsize=8,
	)

	return _fig_to_base64(fig)


def generate_market_dynamics_plot(
	snapshots: List[Dict[str, Any]],
) -> str:
	"""生成市场动态图 (base64 PNG)

	Args:
		snapshots: 快照列表

	Returns:
		base64 PNG 字符串
	"""
	fig, (ax1, ax2) = plt.subplots(
		2, 1, figsize=(10, 6), facecolor="#1a1a2e",
		gridspec_kw={"hspace": 0.3},
	)

	for ax in [ax1, ax2]:
		_setup_dark_axes(ax)

	steps = []
	tou_prices = []
	effective_prices = []
	dr_signals = []

	for snap in snapshots:
		step = snap.get("step", 0)
		market = snap.get("market_data", {})
		steps.append(step)
		tou_prices.append(market.get("tou_base_price", 0.0))
		uc_act = market.get("uc_actions", {})
		effective_prices.append(uc_act.get("effective_price", 0.0))
		dr_signals.append(uc_act.get("dr_signal_value", 0.0))

	# 电价
	ax1.fill_between(steps, tou_prices, alpha=0.3, color="#6B7280", label="TOU Base")
	ax1.plot(steps, effective_prices, color="#F59E0B", linewidth=2, label="Effective")
	ax1.set_title("Electricity Pricing", fontsize=11)
	ax1.set_ylabel("Price ($/kWh)")
	ax1.legend(facecolor="#16213e", edgecolor="#3a3a5e", labelcolor="#e0e0e0", fontsize=8)

	# DR 信号
	ax2.bar(steps, dr_signals, color="#7C3AED", alpha=0.7)
	ax2.set_title("DR Signal", fontsize=11)
	ax2.set_xlabel("Step")
	ax2.set_ylabel("Signal Value")

	return _fig_to_base64(fig)


def generate_all_report_plots(
	snapshots: List[Dict[str, Any]],
) -> Dict[str, str]:
	"""生成所有报告图表

	Args:
		snapshots: 快照列表

	Returns:
		{图表名: base64 PNG 字符串}
	"""
	plots: Dict[str, str] = {}

	try:
		plots["voltage_summary"] = generate_voltage_summary_plot(snapshots)
	except Exception as exc:
		logger.warning(f"Failed to generate voltage summary plot: {exc}")

	try:
		plots["reward_curve"] = generate_reward_curve_plot(snapshots)
	except Exception as exc:
		logger.warning(f"Failed to generate reward curve plot: {exc}")

	try:
		plots["market_dynamics"] = generate_market_dynamics_plot(snapshots)
	except Exception as exc:
		logger.warning(f"Failed to generate market dynamics plot: {exc}")

	return plots
