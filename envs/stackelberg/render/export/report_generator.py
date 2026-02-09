# -*- coding: utf-8 -*-
"""
Stackelberg HTML 报告生成器

继承 BaseReportGenerator，生成包含市场动态、
Leader-Follower 交互和电力系统运行状态的 HTML 报告。
"""

from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.export.base_report_generator import BaseReportGenerator


class StackelbergReportGenerator(BaseReportGenerator):
	"""Stackelberg HTML 报告生成器

	琥珀色主题，包含 Stackelberg 博弈特有的报告章节。
	"""

	def __init__(self):
		super().__init__(
			env_name="Stackelberg",
			accent_color="#F59E0B",
		)

	def _generate_sections(
		self,
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> List[str]:
		"""生成 Stackelberg 报告各 section

		Args:
			snapshots: 快照列表
			metadata: 报告元数据

		Returns:
			HTML 片段列表
		"""
		sections = []

		sections.append(self._section_overview(snapshots, metadata))
		sections.append(self._section_voltage_summary(snapshots))
		sections.append(self._section_market_dynamics(snapshots))
		sections.append(self._section_agent_performance(snapshots))
		sections.append(self._section_system_losses(snapshots))

		# 尝试嵌入 Matplotlib 静态图
		try:
			from envs.stackelberg.render.viz.mpl.static_report_plots import (
				generate_all_report_plots,
			)
			plots = generate_all_report_plots(snapshots)
			if plots:
				sections.append(self._section_charts(plots))
		except ImportError:
			pass

		return sections

	def _section_overview(
		self,
		snapshots: List[Dict[str, Any]],
		metadata: Dict[str, Any],
	) -> str:
		"""概览 section"""
		n_steps = len(snapshots)
		total_reward = sum(
			snap.get("step_reward", 0.0) for snap in snapshots
		)

		# Agent 信息
		n_agents = 0
		n_consumers = 0
		if snapshots:
			n_agents = snapshots[0].get("n_agents", 0)
			n_consumers = snapshots[0].get("n_consumers", 0)

		return f"""
<div class="section">
<h2>Episode Overview</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Episode Length</td><td>{n_steps} steps</td></tr>
<tr><td>Total Reward</td><td>{total_reward:.4f}</td></tr>
<tr><td>Total Agents</td><td>{n_agents} (1 UC Leader + {n_consumers} Consumers)</td></tr>
<tr><td>System</td><td>{metadata.get('system_name', 'Unknown')}</td></tr>
<tr><td>Seed</td><td>{metadata.get('seed', '-')}</td></tr>
</table>
</div>"""

	def _section_voltage_summary(
		self,
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""电压摘要 section"""
		v_mins = []
		v_means = []
		v_maxs = []
		violations = 0

		for snap in snapshots:
			vs = snap.get("voltage_summary", {})
			v_min = vs.get("v_min", 1.0)
			v_mean = vs.get("v_mean", 1.0)
			v_max = vs.get("v_max", 1.0)
			v_mins.append(v_min)
			v_means.append(v_mean)
			v_maxs.append(v_max)
			if v_min < 0.95 or v_max > 1.05:
				violations += 1

		if not v_mins:
			return '<div class="section"><h2>Voltage Summary</h2><p>No data</p></div>'

		return f"""
<div class="section">
<h2>Voltage Summary</h2>
<table>
<tr><th>Metric</th><th>Min</th><th>Mean</th><th>Max</th></tr>
<tr><td>V_min (pu)</td><td>{min(v_mins):.4f}</td><td>{np.mean(v_mins):.4f}</td><td>{max(v_mins):.4f}</td></tr>
<tr><td>V_mean (pu)</td><td>{min(v_means):.4f}</td><td>{np.mean(v_means):.4f}</td><td>{max(v_means):.4f}</td></tr>
<tr><td>V_max (pu)</td><td>{min(v_maxs):.4f}</td><td>{np.mean(v_maxs):.4f}</td><td>{max(v_maxs):.4f}</td></tr>
</table>
<p>Voltage violations (V &lt; 0.95 or V &gt; 1.05): <b>{violations}</b> / {len(snapshots)} steps</p>
</div>"""

	def _section_market_dynamics(
		self,
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""市场动态 section"""
		prices = []
		dr_signals = []

		for snap in snapshots:
			market = snap.get("market_data", {})
			uc_act = market.get("uc_actions", {})
			ep = uc_act.get("effective_price")
			if ep is not None:
				prices.append(ep)
			dr = uc_act.get("dr_signal_value")
			if dr is not None:
				dr_signals.append(dr)

		rows = ""
		if prices:
			rows += f"""
<tr><td>Avg Effective Price</td><td>{np.mean(prices):.4f} $/kWh</td></tr>
<tr><td>Price Range</td><td>{min(prices):.4f} - {max(prices):.4f} $/kWh</td></tr>
<tr><td>Price Volatility (std)</td><td>{np.std(prices):.4f}</td></tr>"""

		if dr_signals:
			rows += f"""
<tr><td>Avg DR Signal</td><td>{np.mean(dr_signals):.4f}</td></tr>
<tr><td>DR Signal Range</td><td>{min(dr_signals):.4f} - {max(dr_signals):.4f}</td></tr>"""

		if not rows:
			return '<div class="section"><h2>Market Dynamics</h2><p>No market data</p></div>'

		return f"""
<div class="section">
<h2>Market Dynamics (Stackelberg Game)</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{rows}
</table>
</div>"""

	def _section_agent_performance(
		self,
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""Agent 性能 section"""
		uc_rewards = []
		consumer_rewards = []

		for snap in snapshots:
			if "uc_reward" in snap:
				uc_rewards.append(snap["uc_reward"])
			if "avg_consumer_reward" in snap:
				consumer_rewards.append(snap["avg_consumer_reward"])

		rows = ""
		if uc_rewards:
			rows += f"""
<tr><td>UC Leader</td>
<td>{np.sum(uc_rewards):.4f}</td>
<td>{np.mean(uc_rewards):.4f}</td>
<td>{min(uc_rewards):.4f}</td>
<td>{max(uc_rewards):.4f}</td></tr>"""

		if consumer_rewards:
			rows += f"""
<tr><td>Consumer Avg</td>
<td>{np.sum(consumer_rewards):.4f}</td>
<td>{np.mean(consumer_rewards):.4f}</td>
<td>{min(consumer_rewards):.4f}</td>
<td>{max(consumer_rewards):.4f}</td></tr>"""

		if not rows:
			return '<div class="section"><h2>Agent Performance</h2><p>No data</p></div>'

		return f"""
<div class="section">
<h2>Agent Performance</h2>
<table>
<tr><th>Agent</th><th>Total</th><th>Mean</th><th>Min</th><th>Max</th></tr>
{rows}
</table>
</div>"""

	def _section_system_losses(
		self,
		snapshots: List[Dict[str, Any]],
	) -> str:
		"""系统损耗 section"""
		losses = []
		for snap in snapshots:
			circuit = snap.get("circuit", {})
			loss = circuit.get("total_loss_kw", 0.0)
			losses.append(loss)

		if not losses or all(l == 0 for l in losses):
			return '<div class="section"><h2>System Losses</h2><p>No loss data</p></div>'

		return f"""
<div class="section">
<h2>System Losses</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Loss (sum)</td><td>{sum(losses):.2f} kWh</td></tr>
<tr><td>Avg Loss per Step</td><td>{np.mean(losses):.2f} kW</td></tr>
<tr><td>Peak Loss</td><td>{max(losses):.2f} kW</td></tr>
</table>
</div>"""

	def _section_charts(
		self,
		plots: Dict[str, str],
	) -> str:
		"""嵌入 Matplotlib 图表 section"""
		html_parts = ['<div class="section"><h2>Charts</h2>']

		for plot_name, b64_data in plots.items():
			title = plot_name.replace("_", " ").title()
			html_parts.append(f"""
<div class="chart-container">
<h3 style="color: #F59E0B; margin-bottom: 8px;">{title}</h3>
<img src="data:image/png;base64,{b64_data}" style="max-width: 100%; border-radius: 4px;">
</div>""")

		html_parts.append("</div>")
		return "\n".join(html_parts)
