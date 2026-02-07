# -*- coding: utf-8 -*-
"""
SmartGrid HTML Report Generator
SmartGrid HTML 报告生成器

继承 BaseReportGenerator，生成暗色主题的 episode 报告，
包含 CMDP 特有的 Lagrangian 分析部分。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.export.base_report_generator import BaseReportGenerator
from envs.smartgrid.render.viz.mpl.static_report_plots import (
	generate_voltage_histogram,
	generate_device_utilization_chart,
)

logger = logging.getLogger(__name__)


class SmartGridReportGenerator(BaseReportGenerator):
	"""SmartGrid HTML 报告生成器

	强调色: #10B981 (emerald)，与 SmartGrid Plotly 主题一致。
	"""

	def __init__(self):
		super().__init__(
			env_name="SmartGrid",
			accent_color="#10B981",
		)

	def _generate_sections(
		self,
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> List[str]:
		"""生成 SmartGrid 报告各 section 的 HTML 片段。

		Args:
			snapshots: 快照列表
			metadata: 报告元数据

		Returns:
			HTML 片段列表
		"""
		sections: List[str] = []

		# 1. Episode 概览
		sections.append(self._section_overview(snapshots, metadata))

		# 2. 电压分析
		sections.append(self._section_voltage(snapshots))

		# 3. 设备利用率
		sections.append(self._section_devices(snapshots))

		# 4. CMDP / Lagrangian 分析
		sections.append(self._section_lagrangian(snapshots))

		# 5. 损耗与效率
		sections.append(self._section_losses(snapshots))

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

		# 最终 lambda
		final_lambda = "N/A"
		if snapshots:
			last = snapshots[-1]
			lagrangian = last.get("lagrangian", {})
			info = last.get("info", {})
			lmbda = lagrangian.get("lmbda", info.get("lambda"))
			if lmbda is not None:
				final_lambda = f"{float(lmbda):.4f}"

		return f"""
<div class="section">
<h2>Episode Overview</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>System</td><td>{system_name}</td></tr>
<tr><td>Total Steps</td><td>{n_steps}</td></tr>
<tr><td>Total Reward</td><td>{total_reward:.4f}</td></tr>
<tr><td>Final Lambda</td><td>{final_lambda}</td></tr>
</table>
</div>"""

	def _section_voltage(self, snapshots: List[Dict[str, Any]]) -> str:
		"""电压分析 section"""
		hist_b64 = generate_voltage_histogram(snapshots)

		# 统计
		v_violations = 0
		v_total = 0
		for snap in snapshots:
			buses = snap.get("buses", {})
			for bus_info in buses.values():
				v_pu = bus_info.get("v_mag_pu", [])
				for v in v_pu:
					v_total += 1
					fv = float(v)
					if fv < 0.95 or fv > 1.05:
						v_violations += 1

		viol_pct = (v_violations / v_total * 100) if v_total > 0 else 0

		img_html = ""
		if hist_b64:
			img_html = (
				f'<div class="chart-container">'
				f'<img src="data:image/png;base64,{hist_b64}" '
				f'style="max-width:100%;" alt="Voltage Distribution">'
				f'</div>'
			)

		return f"""
<div class="section">
<h2>Voltage Analysis</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Voltage Samples</td><td>{v_total}</td></tr>
<tr><td>Violations (out of [0.95, 1.05])</td><td>{v_violations}</td></tr>
<tr><td>Violation Rate</td><td>{viol_pct:.2f}%</td></tr>
</table>
{img_html}
</div>"""

	def _section_devices(self, snapshots: List[Dict[str, Any]]) -> str:
		"""设备利用率 section"""
		chart_b64 = generate_device_utilization_chart(snapshots)

		img_html = ""
		if chart_b64:
			img_html = (
				f'<div class="chart-container">'
				f'<img src="data:image/png;base64,{chart_b64}" '
				f'style="max-width:100%;" alt="Device Utilization">'
				f'</div>'
			)

		return f"""
<div class="section">
<h2>Device Utilization</h2>
{img_html}
</div>"""

	def _section_lagrangian(self, snapshots: List[Dict[str, Any]]) -> str:
		"""CMDP / Lagrangian 分析 section"""
		lambda_vals: List[float] = []
		cost_vals: List[float] = []

		for snap in snapshots:
			info = snap.get("info", {})
			rc = snap.get("reward_components", {})
			lagrangian = snap.get("lagrangian", {})
			merged = {**rc, **info, **lagrangian}

			lmbda = merged.get("lambda", merged.get("lmbda"))
			if lmbda is not None:
				lambda_vals.append(float(lmbda))

			cost = merged.get("cost_voltage")
			if cost is not None:
				cost_vals.append(float(cost))

		rows = ""
		if lambda_vals:
			rows += f"<tr><td>Lambda Range</td><td>[{min(lambda_vals):.4f}, {max(lambda_vals):.4f}]</td></tr>\n"
			rows += f"<tr><td>Final Lambda</td><td>{lambda_vals[-1]:.4f}</td></tr>\n"

		if cost_vals:
			avg_cost = float(np.mean(cost_vals))
			rows += f"<tr><td>Avg Cost (Voltage)</td><td>{avg_cost:.6f}</td></tr>\n"
			rows += f"<tr><td>Max Cost (Voltage)</td><td>{max(cost_vals):.6f}</td></tr>\n"

		if not rows:
			rows = "<tr><td colspan='2'>No CMDP data available</td></tr>"

		return f"""
<div class="section">
<h2>CMDP / Lagrangian Analysis</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{rows}
</table>
</div>"""

	def _section_losses(self, snapshots: List[Dict[str, Any]]) -> str:
		"""损耗与效率 section"""
		loss_vals: List[float] = []

		for snap in snapshots:
			circuit = snap.get("circuit", {})
			loss = circuit.get("total_loss_kw")
			if isinstance(loss, (int, float)):
				loss_vals.append(float(loss))

		if loss_vals:
			avg_loss = float(np.mean(loss_vals))
			total_loss = float(np.sum(loss_vals))
			max_loss = float(np.max(loss_vals))
			rows = f"""
<tr><td>Avg Loss (kW)</td><td>{avg_loss:.2f}</td></tr>
<tr><td>Max Loss (kW)</td><td>{max_loss:.2f}</td></tr>
<tr><td>Total Loss (kWh)</td><td>{total_loss:.2f}</td></tr>"""
		else:
			rows = "<tr><td colspan='2'>No loss data available</td></tr>"

		return f"""
<div class="section">
<h2>Power Loss & Efficiency</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{rows}
</table>
</div>"""
