"""
Report Generator
HTML 分析报告生成器 -- 生成自包含的 HTML 分析报告

使用 Plotly.js CDN 生成交互式图表。当 viz/plotly/ 模块不可用时，
回退到纯表格报告。
"""

import html as html_lib
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_plotly_available = False
try:
	import plotly.graph_objects as go
	_plotly_available = True
except ImportError:
	logger.info("plotly not available, will generate table-only report")

# 暗色主题 Plotly layout 公共参数
_DARK_LAYOUT = dict(
	template="plotly_dark", paper_bgcolor="#0f0f23",
	plot_bgcolor="#0f0f23", height=400,
)

_CSS = (
	"body{font-family:'Inter',sans-serif;background:#1a1a2e;color:#e0e0e0;"
	"margin:0;padding:20px}"
	".container{max-width:1200px;margin:0 auto}"
	"h1,h2{color:#4F46E5;border-bottom:2px solid #3a3a5a;padding-bottom:8px}"
	"h1{font-size:24px}h2{font-size:18px;margin-top:32px}"
	"table{border-collapse:collapse;width:100%;margin:16px 0}"
	"th,td{border:1px solid #3a3a5a;padding:8px 12px;text-align:left}"
	"th{background:#16213e;color:#7C3AED;font-weight:600}"
	"td{background:#0f0f23}"
	".cc{margin:16px 0;background:#0f0f23;border:1px solid #3a3a5a;"
	"border-radius:8px;padding:8px}"
	".mg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
	"gap:12px;margin:16px 0}"
	".mc{background:#16213e;border:1px solid #3a3a5a;border-radius:8px;"
	"padding:16px;text-align:center}"
	".mv{font-size:24px;font-weight:700;color:#10B981}"
	".ml{font-size:12px;color:#a0a0a0;margin-top:4px}"
	".ts{color:#a0a0a0;font-size:12px;text-align:right;margin-top:32px}"
)


def generate_report(
	snapshots: List[Dict[str, Any]],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
) -> str:
	"""生成自包含 HTML 分析报告

	Args:
		snapshots: 快照列表
		bus_coords: 母线坐标字典
		output_path: 输出 .html 文件路径

	Returns:
		输出文件的绝对路径
	"""
	sections = [
		_build_summary_section(snapshots),
		_build_voltage_section(snapshots),
		_build_power_section(snapshots),
		_build_device_section(snapshots),
		_build_reward_section(snapshots),
	]
	body = "\n".join(sections)

	plotly_cdn = ""
	if _plotly_available:
		plotly_cdn = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
	ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	full_html = (
		f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
		f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
		f'<title>PowerZoo Episode Analysis Report</title>'
		f'{plotly_cdn}<style>{_CSS}</style></head><body>'
		f'<div class="container"><h1>PowerZoo Episode Analysis Report</h1>'
		f'{body}<p class="ts">Generated: {ts}</p></div></body></html>'
	)

	with open(output_path, "w", encoding="utf-8") as f:
		f.write(full_html)
	return output_path


def _build_summary_section(snapshots: List[Dict[str, Any]]) -> str:
	"""构建 Episode 概要卡片"""
	n = len(snapshots)
	total_reward, total_loss = 0.0, 0.0
	v_means: List[float] = []
	for snap in snapshots:
		rewards = snap.get("rewards")
		if isinstance(rewards, list):
			total_reward += sum(r for r in rewards if isinstance(r, (int, float)))
		cir = snap.get("circuit", {})
		loss = cir.get("total_loss_kw")
		if isinstance(loss, (int, float)):
			total_loss += loss
		vm = cir.get("v_mean_pu")
		if isinstance(vm, (int, float)):
			v_means.append(vm)
	avg_v = sum(v_means) / len(v_means) if v_means else 0.0
	avg_loss = total_loss / n if n > 0 else 0.0

	cards = [
		(n, "Total Steps"), (f"{total_reward:.2f}", "Total Reward"),
		(f"{avg_v:.4f}", "Avg Voltage (pu)"), (f"{avg_loss:.2f}", "Avg Loss (kW)"),
	]
	items = "".join(
		f'<div class="mc"><div class="mv">{v}</div><div class="ml">{l}</div></div>'
		for v, l in cards
	)
	return f'<h2>Episode Summary</h2><div class="mg">{items}</div>'


def _build_voltage_section(snapshots: List[Dict[str, Any]]) -> str:
	"""构建电压分析节"""
	steps, v_means, v_mins, v_maxs = [], [], [], []
	viol_low, viol_high = 0, 0
	for snap in snapshots:
		steps.append(snap.get("step", 0))
		cir = snap.get("circuit", {})
		v_means.append(cir.get("v_mean_pu", 1.0))
		v_mins.append(cir.get("v_min_pu", 1.0))
		v_maxs.append(cir.get("v_max_pu", 1.0))
		lc = cir.get("v_below_095_count", 0)
		hc = cir.get("v_above_105_count", 0)
		if isinstance(lc, (int, float)):
			viol_low += int(lc)
		if isinstance(hc, (int, float)):
			viol_high += int(hc)

	if _plotly_available:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=steps, y=v_means, name="V_mean", line=dict(color="#10B981", width=2)))
		fig.add_trace(go.Scatter(x=steps, y=v_mins, name="V_min", line=dict(color="#3B82F6", width=1, dash="dash")))
		fig.add_trace(go.Scatter(x=steps, y=v_maxs, name="V_max", line=dict(color="#EF4444", width=1, dash="dash")))
		fig.add_hline(y=0.95, line_dash="dot", line_color="#F59E0B", annotation_text="0.95 pu")
		fig.add_hline(y=1.05, line_dash="dot", line_color="#F59E0B", annotation_text="1.05 pu")
		fig.update_layout(title="Voltage Profile", xaxis_title="Step", yaxis_title="Voltage (pu)", **_DARK_LAYOUT)
		chart = f'<div class="cc">{fig.to_html(include_plotlyjs=False, full_html=False)}</div>'
	else:
		chart = _table_from_columns(
			["Step", "V_mean", "V_min", "V_max"],
			list(zip(steps, v_means, v_mins, v_maxs)), max_rows=24,
		)

	return (
		f'<h2>Voltage Analysis</h2>{chart}'
		f'<table><tr><th>Metric</th><th>Value</th></tr>'
		f'<tr><td>Under-voltage events (V &lt; 0.95)</td><td>{viol_low}</td></tr>'
		f'<tr><td>Over-voltage events (V &gt; 1.05)</td><td>{viol_high}</td></tr></table>'
	)


def _build_power_section(snapshots: List[Dict[str, Any]]) -> str:
	"""构建功率分析节"""
	steps, gen, load, loss, pv, stor = [], [], [], [], [], []
	for snap in snapshots:
		steps.append(snap.get("step", 0))
		cir = snap.get("circuit", {})
		gen.append(cir.get("total_gen_kw", 0))
		load.append(cir.get("total_load_kw", 0))
		loss.append(cir.get("total_loss_kw", 0))
		pv.append(cir.get("total_pv_kw", 0))
		stor.append(cir.get("total_storage_kw", 0))

	if _plotly_available:
		fig = go.Figure()
		traces = [
			("Generation", gen, "#10B981", 2, None),
			("Load", load, "#6B7280", 2, None),
			("PV", pv, "#F59E0B", 2, None),
			("Storage", stor, "#3B82F6", 2, None),
		]
		for name, vals, color, w, _ in traces:
			fig.add_trace(go.Scatter(x=steps, y=vals, name=name, line=dict(color=color, width=w)))
		fig.add_trace(go.Scatter(
			x=steps, y=loss, name="Loss", line=dict(color="#EF4444", width=1, dash="dash"),
			fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
		))
		fig.update_layout(title="Power Balance", xaxis_title="Step", yaxis_title="Power (kW)", **_DARK_LAYOUT)
		chart = f'<div class="cc">{fig.to_html(include_plotlyjs=False, full_html=False)}</div>'
	else:
		chart = _table_from_columns(
			["Step", "Gen", "Load", "PV", "Storage", "Loss"],
			list(zip(steps, gen, load, pv, stor, loss)), max_rows=24,
		)

	t_loss = sum(loss)
	t_gen = sum(gen)
	ratio = (t_loss / t_gen * 100) if t_gen > 0 else 0
	return (
		f'<h2>Power Analysis</h2>{chart}'
		f'<table><tr><th>Metric</th><th>Value</th></tr>'
		f'<tr><td>Total energy loss (kWh)</td><td>{t_loss * 0.25:.2f}</td></tr>'
		f'<tr><td>Total generation (kWh)</td><td>{t_gen * 0.25:.2f}</td></tr>'
		f'<tr><td>Loss ratio</td><td>{ratio:.2f}%</td></tr></table>'
	)


def _build_device_section(snapshots: List[Dict[str, Any]]) -> str:
	"""构建设备调度时序图"""
	steps: List[int] = []
	pv_n: set = set()
	st_n: set = set()
	ev_n: set = set()
	for snap in snapshots:
		steps.append(snap.get("step", 0))
		dev = snap.get("devices", {})
		pv_n.update(dev.get("pv", {}).keys())
		st_n.update(dev.get("storage", {}).keys())
		ev_n.update(dev.get("ev", {}).keys())

	pv_s = sorted(pv_n)
	st_s = sorted(st_n)
	ev_s = sorted(ev_n)
	pv_d: Dict[str, List[float]] = {n: [] for n in pv_s}
	st_d: Dict[str, List[float]] = {n: [] for n in st_s}
	ev_d: Dict[str, List[float]] = {n: [] for n in ev_s}

	for snap in snapshots:
		dev = snap.get("devices", {})
		for n in pv_s:
			pv_d[n].append(dev.get("pv", {}).get(n, {}).get("kw_output", 0))
		for n in st_s:
			st_d[n].append(dev.get("storage", {}).get(n, {}).get("kw_output", 0))
		for n in ev_s:
			ev_d[n].append(dev.get("ev", {}).get(n, {}).get("kw", 0))

	if _plotly_available:
		fig = go.Figure()
		for n in pv_s:
			fig.add_trace(go.Scatter(x=steps, y=pv_d[n], name=f"PV:{n}", line=dict(width=1.5)))
		for n in st_s:
			fig.add_trace(go.Scatter(x=steps, y=st_d[n], name=f"ESS:{n}", line=dict(width=1.5, dash="dash")))
		for n in ev_s:
			fig.add_trace(go.Scatter(x=steps, y=ev_d[n], name=f"EV:{n}", line=dict(width=1, dash="dot")))
		fig.update_layout(
			title="Device Dispatch", xaxis_title="Step", yaxis_title="Power (kW)",
			legend=dict(font=dict(size=9)), **_DARK_LAYOUT,
		)
		chart = f'<div class="cc">{fig.to_html(include_plotlyjs=False, full_html=False)}</div>'
	else:
		rows = []
		for n in pv_s:
			v = pv_d[n]
			rows.append((f"PV:{n}", f"{sum(v)/len(v):.2f}" if v else "0", f"{max(v):.2f}" if v else "0"))
		for n in st_s:
			v = st_d[n]
			rows.append((f"ESS:{n}", f"{sum(v)/len(v):.2f}" if v else "0", f"{max(v):.2f}" if v else "0"))
		for n in ev_s:
			v = ev_d[n]
			rows.append((f"EV:{n}", f"{sum(v)/len(v):.2f}" if v else "0", f"{max(v):.2f}" if v else "0"))
		chart = _table_from_columns(["Device", "Avg kW", "Max kW"], rows)

	return f'<h2>Device Dispatch</h2>{chart}'


def _build_reward_section(snapshots: List[Dict[str, Any]]) -> str:
	"""构建奖励分析节"""
	steps: List[int] = []
	totals: List[float] = []
	comp_keys: List[str] = []
	seen: set = set()
	for snap in snapshots:
		rc = snap.get("reward_components", {})
		if rc:
			for k in rc:
				if k not in seen:
					seen.add(k)
					comp_keys.append(k)

	comp_data: Dict[str, List[float]] = {k: [] for k in comp_keys}
	for snap in snapshots:
		steps.append(snap.get("step", 0))
		rewards = snap.get("rewards")
		tr = sum(r for r in rewards if isinstance(r, (int, float))) if isinstance(rewards, list) else 0.0
		totals.append(tr)
		rc = snap.get("reward_components", {}) or {}
		for k in comp_keys:
			v = rc.get(k, 0)
			if isinstance(v, list):
				v = sum(v) / len(v) if v else 0
			comp_data[k].append(v if isinstance(v, (int, float)) else 0)

	if _plotly_available:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=steps, y=totals, name="Total Reward", line=dict(color="#4F46E5", width=2)))
		colors = ["#10B981", "#F59E0B", "#EF4444", "#3B82F6", "#7C3AED", "#F97316", "#06B6D4", "#EC4899"]
		for i, k in enumerate(comp_keys):
			fig.add_trace(go.Scatter(
				x=steps, y=comp_data[k], name=k,
				line=dict(color=colors[i % len(colors)], width=1, dash="dash"),
			))
		fig.update_layout(title="Reward Analysis", xaxis_title="Step", yaxis_title="Reward", **_DARK_LAYOUT)
		chart = f'<div class="cc">{fig.to_html(include_plotlyjs=False, full_html=False)}</div>'
	else:
		chart = _table_from_columns(
			["Step", "Total"] + comp_keys,
			[
				(s, f"{r:.4f}") + tuple(f"{comp_data[k][i]:.4f}" for k in comp_keys)
				for i, (s, r) in enumerate(zip(steps, totals))
			],
			max_rows=24,
		)

	return f'<h2>Reward Analysis</h2>{chart}'


def _table_from_columns(headers: List[str], rows: List[tuple], max_rows: int = 0) -> str:
	"""生成 HTML 表格"""
	display = rows[:max_rows] if max_rows > 0 and len(rows) > max_rows else rows
	th = "".join(f"<th>{html_lib.escape(str(h))}</th>" for h in headers)
	tr = "".join(
		"<tr>" + "".join(f"<td>{html_lib.escape(str(c))}</td>" for c in row) + "</tr>"
		for row in display
	)
	note = ""
	if max_rows > 0 and len(rows) > max_rows:
		note = f'<p style="color:#a0a0a0;font-size:12px;">Showing {max_rows} of {len(rows)} rows</p>'
	return f"<table><tr>{th}</tr>{tr}</table>{note}"
