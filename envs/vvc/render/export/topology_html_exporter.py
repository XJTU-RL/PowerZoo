# -*- coding: utf-8 -*-
"""
VVC 拓扑 HTML 报告生成器

继承 BaseReportGenerator，生成包含拓扑图、电压热图、
设备时序和奖励分解的综合 HTML 报告。
"""

import logging
from typing import Any, Dict, List, Optional

from envs.render_common.export.base_report_generator import BaseReportGenerator

logger = logging.getLogger(__name__)


class VVCReportGenerator(BaseReportGenerator):
	"""VVC HTML 报告生成器

	报告包含:
	1. 系统概要 (母线数、设备数、配置摘要)
	2. 电压统计表格
	3. 设备状态概要
	4. 逐步性能指标表
	5. 嵌入 Plotly 图表 (拓扑图、热图、奖励分解)
	"""

	def __init__(self):
		super().__init__(env_name="VVC", accent_color="#4F46E5")

	def _generate_sections(
		self,
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> List[str]:
		"""生成报告各 section 的 HTML 片段。

		Args:
			snapshots: 快照列表
			metadata: 报告元数据

		Returns:
			HTML 片段列表
		"""
		sections: List[str] = []

		# Section 1: 系统概要
		sections.append(self._section_summary(snapshots, metadata))

		# Section 2: 电压统计
		sections.append(self._section_voltage_stats(snapshots))

		# Section 3: 设备状态
		sections.append(self._section_device_summary(snapshots))

		# Section 4: 性能指标时间线
		sections.append(self._section_metrics_timeline(snapshots))

		# Section 5: 嵌入图表
		sections.append(self._section_embedded_charts(snapshots))

		return sections

	@staticmethod
	def _section_summary(
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> str:
		"""系统概要 section。"""
		n_steps = len(snapshots)

		# 统计母线和设备数
		n_buses = 0
		n_caps = 0
		n_regs = 0
		n_bats = 0
		n_pvs = 0

		if snapshots:
			last = snapshots[-1]
			n_buses = len(last.get("buses", {}))
			devices = last.get("devices", {})
			n_caps = len(devices.get("capacitors", {}))
			n_regs = len(devices.get("regulators", {}))
			n_bats = len(devices.get("batteries", {}))
			n_pvs = len(devices.get("pvsystems", {}))

		total_reward = metadata.get("total_reward", 0.0)

		return f"""
<div class="section">
<h2>System Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Episode Length</td><td>{n_steps} steps</td></tr>
<tr><td>Total Reward</td><td>{total_reward:.4f}</td></tr>
<tr><td>Buses</td><td>{n_buses}</td></tr>
<tr><td>Capacitors</td><td>{n_caps}</td></tr>
<tr><td>Regulators</td><td>{n_regs}</td></tr>
<tr><td>Batteries</td><td>{n_bats}</td></tr>
<tr><td>PV Systems</td><td>{n_pvs}</td></tr>
</table>
</div>
"""

	@staticmethod
	def _section_voltage_stats(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""电压统计 section。"""
		rows_html = ""

		for snap in snapshots:
			step = snap.get("step", 0)
			circuit = snap.get("circuit", {})
			v_mean = circuit.get("v_mean_pu", 1.0)
			v_min = circuit.get("v_min_pu", 1.0)
			v_max = circuit.get("v_max_pu", 1.0)
			loss = circuit.get("total_loss_kw", 0.0)

			# 电压越限高亮
			style = ""
			if v_min < 0.95 or v_max > 1.05:
				style = ' style="background: rgba(239,68,68,0.15);"'

			rows_html += (
				f"<tr{style}>"
				f"<td>{step}</td>"
				f"<td>{v_mean:.4f}</td>"
				f"<td>{v_min:.4f}</td>"
				f"<td>{v_max:.4f}</td>"
				f"<td>{loss:.2f}</td>"
				f"</tr>\n"
			)

		return f"""
<div class="section">
<h2>Voltage Statistics</h2>
<table>
<tr>
<th>Step</th><th>V Mean (pu)</th><th>V Min (pu)</th>
<th>V Max (pu)</th><th>Loss (kW)</th>
</tr>
{rows_html}
</table>
</div>
"""

	@staticmethod
	def _section_device_summary(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""设备状态概要 section。"""
		if not snapshots:
			return '<div class="section"><h2>Device Summary</h2><p>No data</p></div>'

		# 统计最后一步的设备状态
		last = snapshots[-1]
		devices = last.get("devices", {})

		cap_rows = ""
		for name, cd in devices.get("capacitors", {}).items():
			state = "ON" if cd.get("is_on", False) else "OFF"
			cap_rows += f"<tr><td>{name}</td><td>{state}</td><td>{cd.get('kvar', 0):.0f} kvar</td></tr>\n"

		reg_rows = ""
		for name, rd in devices.get("regulators", {}).items():
			reg_rows += f"<tr><td>{name}</td><td>Tap {rd.get('tap', 0)}</td><td>{rd.get('forward_vreg', 0):.1f}V</td></tr>\n"

		bat_rows = ""
		for name, bd in devices.get("batteries", {}).items():
			bat_rows += f"<tr><td>{name}</td><td>SOC {bd.get('soc', 0):.1%}</td><td>{bd.get('kw', 0):.1f} kW</td></tr>\n"

		pv_rows = ""
		for name, pd_data in devices.get("pvsystems", {}).items():
			pv_rows += f"<tr><td>{name}</td><td>{pd_data.get('kw', 0):.1f} kW</td><td>Curtail {pd_data.get('curtail_pct', 0):.1f}%</td></tr>\n"

		return f"""
<div class="section">
<h2>Device Summary (Final Step)</h2>
<h3>Capacitors</h3>
<table><tr><th>Name</th><th>State</th><th>Rating</th></tr>{cap_rows}</table>
<h3>Regulators</h3>
<table><tr><th>Name</th><th>Tap</th><th>Vreg</th></tr>{reg_rows}</table>
<h3>Batteries</h3>
<table><tr><th>Name</th><th>SOC</th><th>Power</th></tr>{bat_rows}</table>
<h3>PV Systems</h3>
<table><tr><th>Name</th><th>Output</th><th>Curtailment</th></tr>{pv_rows}</table>
</div>
"""

	@staticmethod
	def _section_metrics_timeline(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""性能指标时间线 section。"""
		rows_html = ""
		for snap in snapshots:
			step = snap.get("step", 0)
			reward = snap.get("step_reward", 0.0)
			cum_reward = snap.get("cumulative_reward", 0.0)
			converged = snap.get("circuit", {}).get("converged", False)

			rows_html += (
				f"<tr>"
				f"<td>{step}</td>"
				f"<td>{reward:.4f}</td>"
				f"<td>{cum_reward:.4f}</td>"
				f"<td>{'Yes' if converged else 'No'}</td>"
				f"</tr>\n"
			)

		return f"""
<div class="section">
<h2>Performance Timeline</h2>
<table>
<tr><th>Step</th><th>Step Reward</th><th>Cumulative</th><th>Converged</th></tr>
{rows_html}
</table>
</div>
"""

	@staticmethod
	def _section_embedded_charts(
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""嵌入 Plotly 图表 section (via JS CDN)。"""
		try:
			from envs.vvc.render.viz.plotly.reward_breakdown import create_reward_breakdown
			from envs.vvc.render.viz.plotly.voltage_heatmap import create_voltage_heatmap

			heatmap_fig = create_voltage_heatmap(snapshots)
			reward_fig = create_reward_breakdown(snapshots)

			heatmap_html = heatmap_fig.to_html(
				full_html=False, include_plotlyjs="cdn"
			)
			reward_html = reward_fig.to_html(
				full_html=False, include_plotlyjs=False
			)

			return f"""
<div class="section">
<h2>Interactive Charts</h2>
<div class="chart-container">{heatmap_html}</div>
<div class="chart-container">{reward_html}</div>
</div>
"""
		except Exception as exc:
			logger.warning(f"Chart embedding failed: {exc}")
			return f"""
<div class="section">
<h2>Interactive Charts</h2>
<p>Chart generation failed: {exc}</p>
</div>
"""
