# -*- coding: utf-8 -*-
"""
Stackelberg 独有：电价信号图

展示 TOU 基础电价 + UC Leader 定价调整 + 有效电价时间序列，
包含峰/平/谷时段着色背景和 DR 信号叠加。
"""

from typing import Any, Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from envs.render_common.viz.theme import get_plotly_layout
from envs.stackelberg.render.data.market_data_extractor import (
	TOU_BASE_PRICES,
	TOU_PERIODS,
	MarketDataExtractor,
)

# 时段颜色
PERIOD_COLORS: Dict[str, str] = {
	"off_peak": "rgba(16,185,129,0.08)",    # 绿色 (谷时段)
	"mid_peak": "rgba(245,158,11,0.08)",    # 琥珀 (平时段)
	"on_peak": "rgba(239,68,68,0.08)",      # 红色 (峰时段)
}

PERIOD_LABELS: Dict[str, str] = {
	"off_peak": "Off-Peak",
	"mid_peak": "Mid-Peak",
	"on_peak": "On-Peak",
}


def _add_tou_background(fig: go.Figure, row: int = 1) -> None:
	"""添加 TOU 时段着色背景

	Args:
		fig: Plotly Figure
		row: 子图行号
	"""
	for period_name, info in TOU_PERIODS.items():
		hours = info["hours"]
		color = PERIOD_COLORS.get(period_name, "rgba(128,128,128,0.05)")

		# 找连续区间
		if not hours:
			continue

		sorted_hours = sorted(hours)
		ranges = []
		start = sorted_hours[0]
		prev = start

		for h in sorted_hours[1:]:
			if h == prev + 1:
				prev = h
			else:
				ranges.append((start, prev))
				start = h
				prev = h
		ranges.append((start, prev))

		for h_start, h_end in ranges:
			fig.add_vrect(
				x0=h_start - 0.5,
				x1=h_end + 0.5,
				fillcolor=color,
				layer="below",
				line_width=0,
				row=row, col=1,
			)


def create_price_signal_chart(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建电价信号时间序列图

	2 行子图：
	1. TOU 基础电价 + 有效电价 + UC 价格调整
	2. DR 信号 + ESS 充放电

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Price Signal (No Data)", env_name="stackelberg"
		))
		return fig

	extractor = MarketDataExtractor()
	market_series = extractor.extract_timeseries(snapshots)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]
	hours = [s % 24 for s in steps]

	fig = make_subplots(
		rows=2, cols=1,
		shared_xaxes=True,
		vertical_spacing=0.10,
		subplot_titles=(
			"Electricity Pricing",
			"Demand Response & ESS",
		),
	)

	# TOU 时段背景 (两个子图都加)
	_add_tou_background(fig, row=1)
	_add_tou_background(fig, row=2)

	# Row 1: 电价
	# TOU 基础电价 (阶梯线)
	fig.add_trace(go.Scatter(
		x=steps,
		y=market_series["tou_base_price"],
		mode="lines",
		name="TOU Base Price",
		line=dict(color="#6B7280", width=2, dash="dash"),
		fill="tozeroy",
		fillcolor="rgba(107,114,128,0.05)",
	), row=1, col=1)

	# 有效电价 (UC 调整后)
	fig.add_trace(go.Scatter(
		x=steps,
		y=market_series["effective_price"],
		mode="lines+markers",
		name="Effective Price",
		line=dict(color="#F59E0B", width=2),
		marker=dict(size=6),
	), row=1, col=1)

	# Row 2: DR + ESS
	# DR 信号
	fig.add_trace(go.Scatter(
		x=steps,
		y=market_series["dr_signal"],
		mode="lines+markers",
		name="DR Signal",
		line=dict(color="#7C3AED", width=2),
		fill="tozeroy",
		fillcolor="rgba(124,58,237,0.1)",
	), row=2, col=1)

	# ESS 充电
	fig.add_trace(go.Bar(
		x=steps,
		y=market_series["ess_charge"],
		name="ESS Charge",
		marker_color="rgba(59,130,246,0.7)",
	), row=2, col=1)

	# ESS 放电 (负值)
	fig.add_trace(go.Bar(
		x=steps,
		y=[-v for v in market_series["ess_discharge"]],
		name="ESS Discharge",
		marker_color="rgba(239,68,68,0.7)",
	), row=2, col=1)

	# 时段标注
	fig.add_annotation(
		x=0.02, y=1.08, xref="paper", yref="paper",
		text=(
			"<span style='color:#10B981'>Off-Peak</span> | "
			"<span style='color:#F59E0B'>Mid-Peak</span> | "
			"<span style='color:#EF4444'>On-Peak</span>"
		),
		showarrow=False,
		font=dict(size=10),
	)

	layout = get_plotly_layout(
		title="Price Signal & Market Dynamics",
		height=600,
		env_name="stackelberg",
	)
	layout.update(
		hovermode="x unified",
		barmode="relative",
	)
	fig.update_layout(**layout)

	fig.update_yaxes(title_text="Price ($/kWh)", row=1, col=1)
	fig.update_yaxes(title_text="Signal / Power", row=2, col=1)
	fig.update_xaxes(title_text="Step (Hour)", row=2, col=1)

	return fig


def create_price_comparison(
	snapshots: List[Dict[str, Any]],
) -> go.Figure:
	"""创建 TOU 基础电价 vs 有效电价 对比面积图

	Args:
		snapshots: 快照列表

	Returns:
		Plotly Figure
	"""
	if not snapshots:
		fig = go.Figure()
		fig.update_layout(**get_plotly_layout(
			"Price Comparison (No Data)", env_name="stackelberg"
		))
		return fig

	extractor = MarketDataExtractor()
	series = extractor.extract_timeseries(snapshots)

	steps = [snap.get("step", i) for i, snap in enumerate(snapshots)]

	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x=steps,
		y=series["tou_base_price"],
		mode="lines",
		name="TOU Base",
		line=dict(color="#6B7280", width=1),
		fill="tonexty",
	))

	fig.add_trace(go.Scatter(
		x=steps,
		y=series["effective_price"],
		mode="lines+markers",
		name="Effective (UC Adjusted)",
		line=dict(color="#F59E0B", width=2),
		fill="tozeroy",
		fillcolor="rgba(245,158,11,0.15)",
	))

	_add_tou_background(fig)

	layout = get_plotly_layout(
		title="TOU Base vs UC Effective Price",
		height=350,
		env_name="stackelberg",
	)
	layout.update(
		xaxis=dict(title="Step (Hour)"),
		yaxis=dict(title="Price ($/kWh)"),
		hovermode="x unified",
	)
	fig.update_layout(**layout)

	return fig
