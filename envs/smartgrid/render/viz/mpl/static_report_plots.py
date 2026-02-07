"""
SmartGrid Static Report Plots (Matplotlib)
静态报告图表

为 HTML 报告生成 base64 编码的 Matplotlib 图表，
包含电压分布直方图、设备利用率饼图等。
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from envs.render_common.viz.theme import apply_matplotlib_theme, COLORS, DEVICE_COLORS

logger = logging.getLogger(__name__)


def generate_voltage_histogram(
	snapshots: List[Dict[str, Any]],
	v_min: float = 0.95,
	v_max: float = 1.05,
) -> str:
	"""生成电压分布直方图，返回 base64 编码的 PNG

	Args:
		snapshots: 快照列表
		v_min: 安全下限
		v_max: 安全上限

	Returns:
		base64 编码的 PNG 字符串
	"""
	apply_matplotlib_theme()

	all_v: List[float] = []
	for snap in snapshots:
		buses = snap.get("buses", {})
		for bus_info in buses.values():
			v_pu = bus_info.get("v_mag_pu", [])
			all_v.extend(float(v) for v in v_pu)

	if not all_v:
		return ""

	fig, ax = plt.subplots(figsize=(8, 5))
	arr = np.array(all_v)

	n, bins, patches = ax.hist(arr, bins=50, color=COLORS["primary"], alpha=0.7, edgecolor="white")

	# 着色: 违规部分用红色
	for patch, left_edge in zip(patches, bins[:-1]):
		if left_edge < v_min or left_edge > v_max:
			patch.set_facecolor(COLORS["danger"])
		elif left_edge < v_min + 0.01 or left_edge > v_max - 0.01:
			patch.set_facecolor(COLORS["warning"])

	ax.axvline(v_min, color=COLORS["warning"], linestyle="--", label=f"V_min = {v_min}")
	ax.axvline(v_max, color=COLORS["warning"], linestyle="--", label=f"V_max = {v_max}")
	ax.axvline(1.0, color=COLORS["text_secondary"], linestyle=":", label="Nominal")

	ax.set_xlabel("Voltage (p.u.)")
	ax.set_ylabel("Count")
	ax.set_title("Voltage Distribution (All Steps)")
	ax.legend(fontsize=9)

	return _fig_to_base64(fig)


def generate_device_utilization_chart(
	snapshots: List[Dict[str, Any]],
) -> str:
	"""生成设备利用率汇总图

	Args:
		snapshots: 快照列表

	Returns:
		base64 编码的 PNG 字符串
	"""
	apply_matplotlib_theme()

	# 统计
	cap_on_count = 0
	cap_total = 0
	reg_changes = 0
	bat_avg_soc: List[float] = []

	for snap in snapshots:
		devices = snap.get("devices", {})
		caps = devices.get("capacitors", {})
		for c in caps.values():
			cap_total += 1
			if c.get("status") == 1:
				cap_on_count += 1

		bats = devices.get("batteries", {})
		for b in bats.values():
			bat_avg_soc.append(float(b.get("soc", 0.0)))

	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	# 电容器利用率
	if cap_total > 0:
		rates = [cap_on_count / cap_total, 1 - cap_on_count / cap_total]
		axes[0].pie(
			rates, labels=["ON", "OFF"],
			colors=[DEVICE_COLORS["capacitor_on"], DEVICE_COLORS["capacitor_off"]],
			autopct="%1.1f%%", startangle=90,
		)
		axes[0].set_title("Capacitor Utilization")
	else:
		axes[0].text(0.5, 0.5, "No Capacitors", ha="center", va="center")
		axes[0].set_title("Capacitor Utilization")

	# 电池 SOC 分布
	if bat_avg_soc:
		axes[1].hist(bat_avg_soc, bins=20, color=DEVICE_COLORS["battery_soc"],
					 alpha=0.7, edgecolor="white")
		axes[1].set_xlabel("SOC")
		axes[1].set_ylabel("Count")
		axes[1].set_title("Battery SOC Distribution")
	else:
		axes[1].text(0.5, 0.5, "No Batteries", ha="center", va="center")
		axes[1].set_title("Battery SOC Distribution")

	# 损耗趋势
	loss_vals: List[float] = []
	for snap in snapshots:
		circuit = snap.get("circuit", {})
		loss = circuit.get("total_loss_kw", 0)
		if isinstance(loss, (int, float)):
			loss_vals.append(float(loss))

	if loss_vals:
		axes[2].plot(range(len(loss_vals)), loss_vals,
					 color=DEVICE_COLORS["objective_cost"], linewidth=1.5)
		axes[2].set_xlabel("Day")
		axes[2].set_ylabel("Loss (kW)")
		axes[2].set_title("Power Loss Trend")
	else:
		axes[2].text(0.5, 0.5, "No Loss Data", ha="center", va="center")
		axes[2].set_title("Power Loss Trend")

	fig.tight_layout()
	return _fig_to_base64(fig)


def _fig_to_base64(fig: plt.Figure) -> str:
	"""将 Matplotlib 图转为 base64 PNG 字符串"""
	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	buf.seek(0)
	return base64.b64encode(buf.read()).decode("utf-8")
