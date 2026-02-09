# -*- coding: utf-8 -*-
"""
Episode 动画

使用 Matplotlib 生成包含多面板的 episode 动画：
电压分布、设备调度、奖励曲线、市场数据。
"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


def create_episode_animation(
	snapshots: List[Dict[str, Any]],
	interval_ms: int = 500,
	output_path: Optional[str] = None,
) -> Optional[FuncAnimation]:
	"""创建 Episode 多面板动画

	4 面板：
	1. 电压分布柱状图
	2. 奖励曲线 (UC vs Consumer)
	3. 市场电价
	4. ESS SOC

	Args:
		snapshots: 快照列表
		interval_ms: 帧间隔 (毫秒)
		output_path: 输出文件路径

	Returns:
		FuncAnimation 对象
	"""
	fig, axes = plt.subplots(
		2, 2, figsize=(14, 10), facecolor="#1a1a2e",
	)
	for ax in axes.flat:
		ax.set_facecolor("#0f0f23")
		ax.tick_params(colors="#a0a0a0")
		for spine in ax.spines.values():
			spine.set_color("#3a3a5e")

	fig.suptitle(
		"Stackelberg Episode Animation",
		color="#F59E0B", fontsize=14, fontweight="bold",
	)

	# 预分配数据
	uc_rewards_acc: List[float] = []
	consumer_rewards_acc: List[float] = []
	prices_acc: List[float] = []
	soc_acc: List[float] = []

	def update(frame: int) -> list:
		snap = snapshots[frame]
		step = snap.get("step", frame)
		artists = []

		# Panel 1: 电压分布
		ax1 = axes[0, 0]
		ax1.clear()
		ax1.set_facecolor("#0f0f23")
		ax1.set_title("Voltage Profile", color="#e0e0e0", fontsize=10)
		vs = snap.get("voltage_summary", {})
		v_min = vs.get("v_min", 1.0)
		v_mean = vs.get("v_mean", 1.0)
		v_max = vs.get("v_max", 1.0)
		bars = ax1.bar(
			["V_min", "V_mean", "V_max"],
			[v_min, v_mean, v_max],
			color=["#EF4444", "#F59E0B", "#3B82F6"],
		)
		ax1.axhline(y=0.95, color="#EF4444", linestyle="--", alpha=0.5)
		ax1.axhline(y=1.05, color="#EF4444", linestyle="--", alpha=0.5)
		ax1.set_ylim(0.90, 1.10)
		ax1.set_ylabel("Voltage (pu)", color="#a0a0a0", fontsize=8)

		# Panel 2: 奖励曲线
		ax2 = axes[0, 1]
		uc_rewards_acc.append(snap.get("uc_reward", 0.0))
		consumer_rewards_acc.append(snap.get("avg_consumer_reward", 0.0))
		ax2.clear()
		ax2.set_facecolor("#0f0f23")
		ax2.set_title("Reward Curves", color="#e0e0e0", fontsize=10)
		x_range = list(range(len(uc_rewards_acc)))
		ax2.plot(x_range, uc_rewards_acc, color="#F59E0B", label="UC Leader")
		ax2.plot(x_range, consumer_rewards_acc, color="#3B82F6", label="Consumer Avg")
		ax2.legend(fontsize=7, facecolor="#16213e", edgecolor="#3a3a5e", labelcolor="#e0e0e0")
		ax2.set_ylabel("Reward", color="#a0a0a0", fontsize=8)

		# Panel 3: 市场电价
		ax3 = axes[1, 0]
		market = snap.get("market_data", {})
		uc_act = market.get("uc_actions", {})
		prices_acc.append(uc_act.get("effective_price", market.get("tou_base_price", 0.0)))
		ax3.clear()
		ax3.set_facecolor("#0f0f23")
		ax3.set_title("Effective Price", color="#e0e0e0", fontsize=10)
		ax3.plot(list(range(len(prices_acc))), prices_acc, color="#F59E0B", linewidth=2)
		ax3.fill_between(
			list(range(len(prices_acc))), prices_acc,
			alpha=0.2, color="#F59E0B",
		)
		ax3.set_ylabel("Price ($/kWh)", color="#a0a0a0", fontsize=8)

		# Panel 4: ESS SOC
		ax4 = axes[1, 1]
		soc = snap.get("ess_soc", 0.5)
		if not isinstance(soc, (int, float)):
			soc = 0.5
		soc_acc.append(soc * 100)
		ax4.clear()
		ax4.set_facecolor("#0f0f23")
		ax4.set_title("ESS SOC", color="#e0e0e0", fontsize=10)
		ax4.plot(list(range(len(soc_acc))), soc_acc, color="#10B981", linewidth=2)
		ax4.fill_between(
			list(range(len(soc_acc))), soc_acc,
			alpha=0.2, color="#10B981",
		)
		ax4.axhline(y=20, color="#EF4444", linestyle="--", alpha=0.5)
		ax4.axhline(y=90, color="#EF4444", linestyle="--", alpha=0.5)
		ax4.set_ylim(0, 100)
		ax4.set_ylabel("SOC (%)", color="#a0a0a0", fontsize=8)
		ax4.set_xlabel("Step", color="#a0a0a0", fontsize=8)

		fig.suptitle(
			f"Stackelberg Episode - Step {step} / Hour {step % 24}",
			color="#F59E0B", fontsize=14, fontweight="bold",
		)

		return artists

	anim = FuncAnimation(
		fig, update, frames=len(snapshots),
		interval=interval_ms, blit=False,
	)

	if output_path:
		try:
			if output_path.endswith(".gif"):
				anim.save(output_path, writer="pillow", fps=1000 // interval_ms)
			else:
				anim.save(output_path, writer="ffmpeg", fps=1000 // interval_ms)
			logger.info(f"Episode animation saved to {output_path}")
			plt.close(fig)
			return None
		except Exception as exc:
			logger.warning(f"Failed to save episode animation: {exc}")

	return anim
