"""
DSR Static Report Plots
静态报告图表

为 HTML 报告生成 Matplotlib 静态图表:
- 恢复进度概览
- 电压分布箱线图
- 优先级负荷恢复甘特图
- 网络状态汇总
"""

import io
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def plot_restoration_overview(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (10, 4),
) -> bytes:
	"""恢复进度概览图

	Args:
		snapshots: 快照列表
		figsize: 图片尺寸

	Returns:
		PNG 图片 bytes
	"""
	import matplotlib.pyplot as plt
	from envs.render_common.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	pcts = []
	kws = []

	for snap in snapshots:
		rest = snap.get("restoration_data", {})
		pcts.append(rest.get("restoration_pct", 0.0))
		kws.append(rest.get("total_restored_kw", 0.0))

	# 恢复百分比
	ax1.fill_between(steps, pcts, alpha=0.3, color="#10B981")
	ax1.plot(steps, pcts, "o-", color="#10B981", markersize=5, linewidth=2)
	ax1.axhline(y=100, color="#10B981", linestyle="--", alpha=0.5, label="100% Target")
	ax1.set_xlabel("Step")
	ax1.set_ylabel("Restoration (%)")
	ax1.set_title("Restoration Progress")
	ax1.set_ylim(0, 110)
	ax1.legend()

	# 恢复功率
	ax2.bar(steps, kws, color="#3B82F6", alpha=0.7, width=0.8)
	ax2.set_xlabel("Step")
	ax2.set_ylabel("Restored Power (kW)")
	ax2.set_title("Restored Load Power")

	fig.tight_layout()
	return _fig_to_bytes(fig)


def plot_voltage_boxplot(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (10, 4),
) -> bytes:
	"""电压分布箱线图

	Args:
		snapshots: 快照列表
		figsize: 图片尺寸

	Returns:
		PNG 图片 bytes
	"""
	import matplotlib.pyplot as plt
	from envs.render_common.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	fig, ax = plt.subplots(1, 1, figsize=figsize)

	all_data = []
	labels = []

	for snap in snapshots:
		step = snap.get("step", 0)
		buses = snap.get("buses", {})
		voltages = []
		for bus_data in buses.values():
			if bus_data.get("is_energized", False):
				for v in bus_data.get("v_mag_pu", []):
					if 0.1 < v < 2.0:
						voltages.append(v)
		if voltages:
			all_data.append(voltages)
			labels.append(f"S{step}")

	if all_data:
		bp = ax.boxplot(
			all_data, labels=labels,
			patch_artist=True,
			boxprops=dict(facecolor="rgba(79,70,229,0.3)", edgecolor="#4F46E5"),
			medianprops=dict(color="#F59E0B", linewidth=2),
			whiskerprops=dict(color="#a0a0a0"),
			capprops=dict(color="#a0a0a0"),
			flierprops=dict(marker="o", markerfacecolor="#EF4444", markersize=3),
		)

	# 安全界限
	ax.axhline(y=0.95, color="#EF4444", linestyle="--", alpha=0.5, label="V_min=0.95")
	ax.axhline(y=1.05, color="#EF4444", linestyle="--", alpha=0.5, label="V_max=1.05")

	ax.set_xlabel("Step")
	ax.set_ylabel("Voltage (pu)")
	ax.set_title("Voltage Distribution per Step")
	ax.legend(fontsize=8)

	fig.tight_layout()
	return _fig_to_bytes(fig)


def plot_priority_gantt(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (10, 4),
) -> bytes:
	"""优先级负荷恢复时间线

	Args:
		snapshots: 快照列表
		figsize: 图片尺寸

	Returns:
		PNG 图片 bytes
	"""
	import matplotlib.pyplot as plt
	from envs.render_common.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	fig, ax = plt.subplots(1, 1, figsize=figsize)

	priorities = ["critical", "high", "medium", "low"]
	priority_colors = {
		"critical": "#EF4444",
		"high": "#F59E0B",
		"medium": "#3B82F6",
		"low": "#6B7280",
	}

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	for idx, priority in enumerate(priorities):
		pcts = []
		for snap in snapshots:
			rest = snap.get("restoration_data", {})
			breakdown = rest.get("priority_breakdown", {})
			stats = breakdown.get(priority, {})
			pcts.append(stats.get("pct", 0.0))

		ax.plot(
			steps, pcts, "o-",
			color=priority_colors[priority],
			label=priority.title(),
			markersize=4, linewidth=1.5,
		)

	ax.axhline(y=100, color="white", linestyle="--", alpha=0.3)
	ax.set_xlabel("Step")
	ax.set_ylabel("Restoration (%)")
	ax.set_title("Priority Load Restoration Timeline")
	ax.set_ylim(0, 110)
	ax.legend(loc="lower right", fontsize=9)

	fig.tight_layout()
	return _fig_to_bytes(fig)


def plot_network_summary(
	snapshots: List[Dict[str, Any]],
	figsize: Tuple[float, float] = (10, 4),
) -> bytes:
	"""网络状态汇总

	Args:
		snapshots: 快照列表
		figsize: 图片尺寸

	Returns:
		PNG 图片 bytes
	"""
	import matplotlib.pyplot as plt
	from envs.render_common.viz.theme import apply_matplotlib_theme

	apply_matplotlib_theme()

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	n_en = []
	n_de = []
	n_overload = []

	for snap in snapshots:
		rest = snap.get("restoration_data", {})
		circuit = snap.get("circuit", {})
		n_en.append(len(rest.get("energized_buses", [])))
		n_de.append(len(rest.get("de_energized_buses", [])))
		n_overload.append(circuit.get("n_overloaded_lines", 0))

	# 堆叠面积图
	ax1.fill_between(steps, n_en, alpha=0.5, color="#10B981", label="Energized")
	ax1.fill_between(steps, [e + d for e, d in zip(n_en, n_de)], n_en,
					  alpha=0.5, color="#6B7280", label="De-energized")
	ax1.set_xlabel("Step")
	ax1.set_ylabel("Bus Count")
	ax1.set_title("Bus State Over Time")
	ax1.legend(fontsize=8)

	# 过载折线
	ax2.plot(steps, n_overload, "o-", color="#EF4444", markersize=4, linewidth=1.5)
	ax2.fill_between(steps, n_overload, alpha=0.2, color="#EF4444")
	ax2.set_xlabel("Step")
	ax2.set_ylabel("Overloaded Lines")
	ax2.set_title("Line Overloads Over Time")

	fig.tight_layout()
	return _fig_to_bytes(fig)


def _fig_to_bytes(fig) -> bytes:
	"""将 matplotlib Figure 转换为 PNG bytes"""
	import matplotlib.pyplot as plt

	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	buf.seek(0)
	return buf.getvalue()
