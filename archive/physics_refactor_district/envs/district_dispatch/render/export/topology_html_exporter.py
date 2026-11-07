"""
Topology HTML Exporter
拓扑 HTML 导出器 -- 导出独立交互式拓扑 HTML 文件

生成可在浏览器中直接打开的自包含 HTML，嵌入 Plotly 拓扑图表。
当 viz/plotly/topology_graph 模块不可用时，使用内置简化版拓扑渲染。
"""

import html as html_lib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 延迟导入 plotly 拓扑模块
_topology_module = None
_plotly_available = False

try:
	import plotly.graph_objects as go
	_plotly_available = True
except ImportError:
	logger.info("plotly not available, topology export will use basic HTML")

try:
	from envs.district_dispatch.render.viz.plotly import topology_graph
	_topology_module = topology_graph
except ImportError:
	logger.info(
		"viz.plotly.topology_graph not available, "
		"using built-in topology renderer"
	)


def export_topology_html(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
	output_path: str,
) -> str:
	"""导出独立交互式拓扑 HTML

	生成可在浏览器中打开的自包含 HTML 文件。
	如果 Plotly 可用，生成交互式拓扑图；否则生成静态 SVG 拓扑。

	Args:
		snapshot: 单个快照字典，包含 buses, lines, devices 等
		bus_coords: 母线坐标字典 {bus_name: (x, y)}
		output_path: 输出 .html 文件路径

	Returns:
		输出文件的绝对路径
	"""
	step = snapshot.get("step", 0)
	timestamp_h = snapshot.get("timestamp_h", step * 0.25)

	if _plotly_available:
		chart_html = _build_plotly_topology(snapshot, bus_coords)
	else:
		chart_html = _build_svg_topology(snapshot, bus_coords)

	# 系统状态摘要表格
	info_html = _build_info_table(snapshot)

	full_html = _wrap_topology_html(
		chart_html=chart_html,
		info_html=info_html,
		step=step,
		timestamp_h=timestamp_h,
	)

	with open(output_path, "w", encoding="utf-8") as f:
		f.write(full_html)

	return output_path


# ======================================================================
# Plotly 拓扑构建
# ======================================================================

def _build_plotly_topology(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> str:
	"""使用 Plotly 构建交互式拓扑图

	Args:
		snapshot: 快照字典
		bus_coords: 母线坐标

	Returns:
		Plotly 图表的 HTML 片段
	"""
	# 优先使用 viz/plotly/topology_graph 模块
	if _topology_module is not None:
		try:
			fig = _topology_module.create_topology_figure(
				snapshot, bus_coords
			)
			return fig.to_html(include_plotlyjs="cdn", full_html=False)
		except Exception as exc:
			logger.warning(
				f"topology_graph module failed: {exc}, "
				"falling back to built-in renderer"
			)

	# 内置简化版拓扑
	return _build_builtin_plotly_topology(snapshot, bus_coords)


def _build_builtin_plotly_topology(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> str:
	"""内置简化版 Plotly 拓扑渲染

	Args:
		snapshot: 快照字典
		bus_coords: 母线坐标

	Returns:
		Plotly 图表的 HTML 片段
	"""
	buses = snapshot.get("buses", {})
	lines = snapshot.get("lines", {})

	fig = go.Figure()

	# 绘制线路 (edges)
	for line_name, line_data in lines.items():
		from_bus = line_data.get("from_bus", "")
		to_bus = line_data.get("to_bus", "")

		if from_bus in bus_coords and to_bus in bus_coords:
			x0, y0 = bus_coords[from_bus]
			x1, y1 = bus_coords[to_bus]
			loading = line_data.get("loading_pct", 0)

			# 根据负载率着色
			if loading > 100:
				color = "#EF4444"
			elif loading > 75:
				color = "#F59E0B"
			else:
				color = "#4B5563"

			fig.add_trace(go.Scatter(
				x=[x0, x1, None],
				y=[y0, y1, None],
				mode="lines",
				line=dict(color=color, width=max(1, loading / 50)),
				hoverinfo="text",
				text=f"{line_name}<br>Loading: {loading:.1f}%",
				showlegend=False,
			))

	# 绘制母线 (nodes)
	bus_x = []
	bus_y = []
	bus_colors = []
	bus_texts = []
	bus_sizes = []

	for bus_name, coord in bus_coords.items():
		bus_x.append(coord[0])
		bus_y.append(coord[1])

		bus_data = buses.get(bus_name, {})
		v_mean = bus_data.get("v_mean", 1.0)
		bus_colors.append(v_mean)

		n_phases = bus_data.get("n_phases", 3)
		bus_sizes.append(8 + n_phases * 2)

		bus_texts.append(
			f"{bus_name}<br>"
			f"V: {v_mean:.4f} pu<br>"
			f"Phases: {n_phases}"
		)

	if bus_x:
		fig.add_trace(go.Scatter(
			x=bus_x,
			y=bus_y,
			mode="markers",
			marker=dict(
				size=bus_sizes,
				color=bus_colors,
				colorscale=[
					[0.0, "#EF4444"],
					[0.25, "#F59E0B"],
					[0.5, "#10B981"],
					[0.75, "#F59E0B"],
					[1.0, "#EF4444"],
				],
				cmin=0.90,
				cmax=1.10,
				colorbar=dict(
					title="Voltage (pu)",
					thickness=15,
				),
				line=dict(color="white", width=1),
			),
			text=bus_texts,
			hoverinfo="text",
			showlegend=False,
		))

	# 标注设备位置
	_annotate_devices(fig, snapshot, bus_coords)

	fig.update_layout(
		title="Network Topology",
		template="plotly_dark",
		paper_bgcolor="#0f0f23",
		plot_bgcolor="#0f0f23",
		xaxis=dict(
			showgrid=False, zeroline=False,
			showticklabels=False, title="",
		),
		yaxis=dict(
			showgrid=False, zeroline=False,
			showticklabels=False, title="",
			scaleanchor="x", scaleratio=1,
		),
		height=600,
		margin=dict(l=20, r=20, t=50, b=20),
	)

	return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _annotate_devices(fig: Any, snapshot: Dict[str, Any], bus_coords: Dict[str, Tuple[float, float]]) -> None:
	"""在拓扑图上标注设备位置 (PV=三角, Storage=方形, EV=菱形)"""
	devices = snapshot.get("devices", {})
	# (设备类型, 数据键, 功率键, 符号, 颜色, 偏移)
	configs = [
		("pv", "kw_output", "triangle-up", "#F59E0B", (0, 0.002)),
		("storage", "kw_output", "square", "#3B82F6", (0, -0.002)),
		("ev", "kw", "diamond", "#7C3AED", (0.002, 0)),
	]
	for dev_type, kw_key, symbol, color, (dx, dy) in configs:
		for name, data in devices.get(dev_type, {}).items():
			bus = data.get("bus", "")
			if bus not in bus_coords:
				continue
			x, y = bus_coords[bus]
			kw = data.get(kw_key, 0)
			extra = ""
			if dev_type == "storage":
				extra = f", SOC={data.get('soc_pct', 0):.1f}%"
			fig.add_trace(go.Scatter(
				x=[x + dx], y=[y + dy], mode="markers",
				marker=dict(symbol=symbol, size=10, color=color, line=dict(color="white", width=1)),
				text=f"{dev_type.upper()}: {name}<br>{kw:.1f} kW{extra}",
				hoverinfo="text", showlegend=False,
			))


# ======================================================================
# SVG 回退拓扑
# ======================================================================

def _build_svg_topology(
	snapshot: Dict[str, Any],
	bus_coords: Dict[str, Tuple[float, float]],
) -> str:
	"""使用纯 SVG 构建静态拓扑图（Plotly 不可用时的回退方案）

	Args:
		snapshot: 快照字典
		bus_coords: 母线坐标

	Returns:
		SVG HTML 字符串
	"""
	if not bus_coords:
		return '<p style="color:#a0a0a0;">No bus coordinates available</p>'

	# 计算坐标范围
	xs = [c[0] for c in bus_coords.values()]
	ys = [c[1] for c in bus_coords.values()]
	x_min, x_max = min(xs), max(xs)
	y_min, y_max = min(ys), max(ys)

	# SVG 视口
	width = 800
	height = 600
	padding = 40
	x_range = x_max - x_min or 1
	y_range = y_max - y_min or 1

	def to_svg_x(x: float) -> float:
		return padding + (x - x_min) / x_range * (width - 2 * padding)

	def to_svg_y(y: float) -> float:
		# SVG y 轴翻转
		return height - padding - (y - y_min) / y_range * (height - 2 * padding)

	elements: List[str] = []

	# 线路
	lines = snapshot.get("lines", {})
	for line_name, line_data in lines.items():
		from_bus = line_data.get("from_bus", "")
		to_bus = line_data.get("to_bus", "")
		if from_bus in bus_coords and to_bus in bus_coords:
			x0, y0 = to_svg_x(bus_coords[from_bus][0]), to_svg_y(bus_coords[from_bus][1])
			x1, y1 = to_svg_x(bus_coords[to_bus][0]), to_svg_y(bus_coords[to_bus][1])
			elements.append(
				f'<line x1="{x0:.1f}" y1="{y0:.1f}" '
				f'x2="{x1:.1f}" y2="{y1:.1f}" '
				f'stroke="#4B5563" stroke-width="1.5"/>'
			)

	# 母线
	buses = snapshot.get("buses", {})
	for bus_name, coord in bus_coords.items():
		sx = to_svg_x(coord[0])
		sy = to_svg_y(coord[1])
		bus_data = buses.get(bus_name, {})
		v_mean = bus_data.get("v_mean", 1.0)

		# 根据电压着色
		if v_mean < 0.95 or v_mean > 1.05:
			fill = "#EF4444"
		elif v_mean < 0.97 or v_mean > 1.03:
			fill = "#F59E0B"
		else:
			fill = "#10B981"

		name_escaped = html_lib.escape(bus_name)
		elements.append(
			f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" '
			f'fill="{fill}" stroke="white" stroke-width="0.5">'
			f'<title>{name_escaped}: {v_mean:.4f} pu</title></circle>'
		)

	svg = (
		f'<svg width="{width}" height="{height}" '
		f'xmlns="http://www.w3.org/2000/svg" '
		f'style="background:#0f0f23;border:1px solid #3a3a5a;'
		f'border-radius:8px;">'
		+ "\n".join(elements)
		+ "</svg>"
	)

	return svg


# ======================================================================
# 信息表格
# ======================================================================

def _build_info_table(snapshot: Dict[str, Any]) -> str:
	"""构建系统状态摘要表格

	Args:
		snapshot: 快照字典

	Returns:
		HTML 表格字符串
	"""
	circuit = snapshot.get("circuit", {})
	devices = snapshot.get("devices", {})

	n_pv = len(devices.get("pv", {}))
	n_storage = len(devices.get("storage", {}))
	n_ev = len(devices.get("ev", {}))

	rows = [
		("Total Loss (kW)", f"{circuit.get('total_loss_kw', 0):.2f}"),
		("Total Load (kW)", f"{circuit.get('total_load_kw', 0):.2f}"),
		("Total Generation (kW)", f"{circuit.get('total_gen_kw', 0):.2f}"),
		("Total PV (kW)", f"{circuit.get('total_pv_kw', 0):.2f}"),
		("Total Storage (kW)", f"{circuit.get('total_storage_kw', 0):.2f}"),
		("Buses", f"{circuit.get('n_buses', 0)}"),
		("PV Systems", f"{n_pv}"),
		("Storage Units", f"{n_storage}"),
		("EV Chargers", f"{n_ev}"),
		("Converged", f"{circuit.get('converged', 'N/A')}"),
	]

	# 电压统计
	v_mean = circuit.get("v_mean_pu")
	v_min = circuit.get("v_min_pu")
	v_max = circuit.get("v_max_pu")
	if isinstance(v_mean, (int, float)):
		rows.append(("V_mean (pu)", f"{v_mean:.4f}"))
	if isinstance(v_min, (int, float)):
		rows.append(("V_min (pu)", f"{v_min:.4f}"))
	if isinstance(v_max, (int, float)):
		rows.append(("V_max (pu)", f"{v_max:.4f}"))

	tr = "".join(
		f"<tr><td>{html_lib.escape(k)}</td><td>{html_lib.escape(v)}</td></tr>"
		for k, v in rows
	)

	return f"""
<table>
<tr><th>Metric</th><th>Value</th></tr>
{tr}
</table>"""


# ======================================================================
# HTML 模板
# ======================================================================

def _wrap_topology_html(
	chart_html: str,
	info_html: str,
	step: int,
	timestamp_h: float,
) -> str:
	"""包装拓扑 HTML 页面

	Args:
		chart_html: 拓扑图表 HTML
		info_html: 信息表格 HTML
		step: 时间步
		timestamp_h: 仿真时间 (h)

	Returns:
		完整的自包含 HTML 字符串
	"""
	css = """
body {
	font-family: 'Inter', -apple-system, sans-serif;
	background: #1a1a2e;
	color: #e0e0e0;
	margin: 0;
	padding: 20px;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 { color: #4F46E5; font-size: 22px; }
h2 { color: #7C3AED; font-size: 16px; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #3a3a5a; padding: 6px 10px; text-align: left; }
th { background: #16213e; color: #7C3AED; }
td { background: #0f0f23; }
.chart { margin: 16px 0; }
.subtitle { color: #a0a0a0; font-size: 13px; }
"""

	return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PowerZoo Topology - Step {step}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<h1>Network Topology</h1>
<p class="subtitle">Step {step} | t = {timestamp_h:.2f} h</p>
<div class="chart">{chart_html}</div>
<h2>System Status</h2>
{info_html}
</div>
</body>
</html>"""
