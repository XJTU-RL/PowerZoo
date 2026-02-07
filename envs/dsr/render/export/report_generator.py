# -*- coding: utf-8 -*-
"""
DSR HTML Report Generator
DSR HTML 报告生成器

继承 BaseReportGenerator，生成暗色主题的 episode 报告，
包含 DSR 特有的恢复进度分析和故障统计部分。
"""

import base64
import logging
from typing import Any, Dict, List

import numpy as np

from envs.render_common.export.base_report_generator import BaseReportGenerator
from envs.dsr.render.viz.mpl.static_report_plots import (
	plot_restoration_overview,
	plot_voltage_boxplot,
	plot_priority_gantt,
	plot_network_summary,
)

logger = logging.getLogger(__name__)


class DSRReportGenerator(BaseReportGenerator):
	"""DSR HTML 报告生成器

	强调色: #DC2626 (red)，与 DSR Plotly 主题一致。
	"""

	def __init__(self):
		super().__init__(
			env_name="DSR",
			accent_color="#DC2626",
		)

	def _generate_sections(
		self,
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> List[str]:
		"""生成 DSR 报告各 section 的 HTML 片段。

		Args:
			snapshots: 快照列表
			metadata: 报告元数据

		Returns:
			HTML 片段列表
		"""
		sections: List[str] = []

		# 1. Episode 概览
		sections.append(self._section_overview(snapshots, metadata))

		# 2. 恢复进度分析
		sections.append(self._section_restoration(snapshots))

		# 3. 电压分析
		sections.append(self._section_voltage(snapshots))

		# 4. 故障与网络状态
		sections.append(self._section_fault_network(snapshots))

		# 5. 优先级负荷时间线
		sections.append(self._section_priority_timeline(snapshots))

		return sections

	def _section_overview(
		self,
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> str:
		"""Episode 概览 section"""
		n_steps = len(snapshots)
		total_reward = metadata.get("total_reward", 0.0)
		system_name = metadata.get("system_name", "Unknown")

		# 最终恢复率
		final_pct = "N/A"
		if snapshots:
			last = snapshots[-1]
			rest = last.get("restoration_data", {})
			pct = rest.get("restoration_pct")
			if pct is not None:
				final_pct = f"{float(pct):.1f}%"

		return f"""
<div class="section">
<h2>Episode Overview</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>System</td><td>{system_name}</td></tr>
<tr><td>Total Steps</td><td>{n_steps}</td></tr>
<tr><td>Total Reward</td><td>{total_reward:.4f}</td></tr>
<tr><td>Final Restoration</td><td>{final_pct}</td></tr>
</table>
</div>"""

	def _section_restoration(self, snapshots: List[Dict[str, Any]]) -> str:
		"""恢复进度分析 section"""
		img_html = ""
		try:
			png_bytes = plot_restoration_overview(snapshots)
			b64 = base64.b64encode(png_bytes).decode("ascii")
			img_html = (
				f'<div class="chart-container">'
				f'<img src="data:image/png;base64,{b64}" '
				f'style="max-width:100%;" alt="Restoration Overview">'
				f'</div>'
			)
		except Exception as e:
			logger.warning(f"Failed to generate restoration overview: {e}")

		# 统计
		rows = ""
		if snapshots:
			last_rest = snapshots[-1].get("restoration_data", {})
			rows += f"<tr><td>Final Restoration (%)</td><td>{last_rest.get('restoration_pct', 0):.1f}</td></tr>\n"
			rows += f"<tr><td>Total Restored (kW)</td><td>{last_rest.get('total_restored_kw', 0):.1f}</td></tr>\n"
			rows += f"<tr><td>Energized Buses</td><td>{len(last_rest.get('energized_buses', []))}</td></tr>\n"
			rows += f"<tr><td>De-energized Buses</td><td>{len(last_rest.get('de_energized_buses', []))}</td></tr>\n"

		return f"""
<div class="section">
<h2>Restoration Progress</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{rows}
</table>
{img_html}
</div>"""

	def _section_voltage(self, snapshots: List[Dict[str, Any]]) -> str:
		"""电压分析 section"""
		img_html = ""
		try:
			png_bytes = plot_voltage_boxplot(snapshots)
			b64 = base64.b64encode(png_bytes).decode("ascii")
			img_html = (
				f'<div class="chart-container">'
				f'<img src="data:image/png;base64,{b64}" '
				f'style="max-width:100%;" alt="Voltage Distribution">'
				f'</div>'
			)
		except Exception as e:
			logger.warning(f"Failed to generate voltage boxplot: {e}")

		# 电压统计
		v_violations = 0
		v_total = 0
		for snap in snapshots:
			buses = snap.get("buses", {})
			for bus_info in buses.values():
				if not bus_info.get("is_energized", False):
					continue
				v_pu = bus_info.get("v_mag_pu", [])
				for v in v_pu:
					v_total += 1
					fv = float(v)
					if fv < 0.95 or fv > 1.05:
						v_violations += 1

		viol_pct = (v_violations / v_total * 100) if v_total > 0 else 0

		return f"""
<div class="section">
<h2>Voltage Analysis</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Voltage Samples (Energized)</td><td>{v_total}</td></tr>
<tr><td>Violations (out of [0.95, 1.05])</td><td>{v_violations}</td></tr>
<tr><td>Violation Rate</td><td>{viol_pct:.2f}%</td></tr>
</table>
{img_html}
</div>"""

	def _section_fault_network(self, snapshots: List[Dict[str, Any]]) -> str:
		"""故障与网络状态 section"""
		img_html = ""
		try:
			png_bytes = plot_network_summary(snapshots)
			b64 = base64.b64encode(png_bytes).decode("ascii")
			img_html = (
				f'<div class="chart-container">'
				f'<img src="data:image/png;base64,{b64}" '
				f'style="max-width:100%;" alt="Network Summary">'
				f'</div>'
			)
		except Exception as e:
			logger.warning(f"Failed to generate network summary: {e}")

		# 故障统计
		fault_lines_seen: set = set()
		for snap in snapshots:
			rest = snap.get("restoration_data", {})
			for fl in rest.get("fault_lines", []):
				fault_lines_seen.add(fl)

		return f"""
<div class="section">
<h2>Fault & Network State</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Distinct Fault Lines</td><td>{len(fault_lines_seen)}</td></tr>
<tr><td>Fault Lines</td><td>{', '.join(sorted(fault_lines_seen)) or 'None'}</td></tr>
</table>
{img_html}
</div>"""

	def _section_priority_timeline(self, snapshots: List[Dict[str, Any]]) -> str:
		"""优先级负荷时间线 section"""
		img_html = ""
		try:
			png_bytes = plot_priority_gantt(snapshots)
			b64 = base64.b64encode(png_bytes).decode("ascii")
			img_html = (
				f'<div class="chart-container">'
				f'<img src="data:image/png;base64,{b64}" '
				f'style="max-width:100%;" alt="Priority Load Timeline">'
				f'</div>'
			)
		except Exception as e:
			logger.warning(f"Failed to generate priority timeline: {e}")

		return f"""
<div class="section">
<h2>Priority Load Restoration Timeline</h2>
{img_html}
</div>"""
