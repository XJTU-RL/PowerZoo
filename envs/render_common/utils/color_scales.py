"""
Color Scales (Common)
渲染系统颜色常量与色彩映射工具

提供 PowerZoo 品牌色、Plotly/Matplotlib 配色方案，
以及电压/负载/SOC/恢复率值到颜色的连续映射函数。
从 District Dispatch 提取，完全可复用。
"""

from typing import Any, Dict, List, Tuple


# -- PowerZoo 品牌色 --

COLORS: Dict[str, str] = {
	"primary": "#4F46E5",
	"secondary": "#7C3AED",
	"success": "#10B981",
	"warning": "#F59E0B",
	"danger": "#EF4444",
	"info": "#3B82F6",
	"bg_dark": "#1a1a2e",
	"bg_card": "#16213e",
	"text_primary": "#e0e0e0",
	"text_secondary": "#a0a0a0",
}

# -- 分区颜色 --

ZONE_COLORS: List[str] = ["#4F46E5", "#10B981", "#F59E0B"]

# -- Plotly 连续色标 --

VOLTAGE_COLORSCALE: List[List[Any]] = [
	[0, "#EF4444"],
	[0.475, "#F59E0B"],
	[0.5, "#10B981"],
	[0.525, "#F59E0B"],
	[1, "#EF4444"],
]

LOADING_COLORSCALE: List[List[Any]] = [
	[0, "#10B981"],
	[0.7, "#F59E0B"],
	[1, "#EF4444"],
]

SOC_COLORSCALE: List[List[Any]] = [
	[0, "#EF4444"],
	[0.2, "#F59E0B"],
	[0.5, "#10B981"],
	[0.8, "#F59E0B"],
	[1, "#EF4444"],
]

RESTORATION_COLORSCALE: List[List[Any]] = [
	[0, "#EF4444"],
	[0.5, "#F59E0B"],
	[1, "#10B981"],
]


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
	"""将十六进制颜色转换为 RGB 元组。

	Args:
		hex_color: 十六进制颜色字符串，如 "#EF4444"

	Returns:
		(R, G, B) 各分量 0-255
	"""
	h = hex_color.lstrip("#")
	return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
	"""将 RGB 元组转换为十六进制颜色字符串。

	Args:
		r: 红色分量 0-255
		g: 绿色分量 0-255
		b: 蓝色分量 0-255

	Returns:
		十六进制颜色字符串
	"""
	return f"#{r:02X}{g:02X}{b:02X}"


def _interpolate_colorscale(value: float, colorscale: List[List[Any]]) -> str:
	"""在色标上线性插值，将 [0, 1] 范围的值映射为颜色。

	Args:
		value: 归一化值，clamp 到 [0, 1]
		colorscale: Plotly 格式色标 [[position, color], ...]

	Returns:
		插值后的十六进制颜色
	"""
	value = max(0.0, min(1.0, value))

	for i in range(len(colorscale) - 1):
		pos_lo, color_lo = colorscale[i][0], colorscale[i][1]
		pos_hi, color_hi = colorscale[i + 1][0], colorscale[i + 1][1]

		if pos_lo <= value <= pos_hi:
			if pos_hi == pos_lo:
				t = 0.0
			else:
				t = (value - pos_lo) / (pos_hi - pos_lo)

			r_lo, g_lo, b_lo = _hex_to_rgb(color_lo)
			r_hi, g_hi, b_hi = _hex_to_rgb(color_hi)

			r = int(r_lo + t * (r_hi - r_lo))
			g = int(g_lo + t * (g_hi - g_lo))
			b = int(b_lo + t * (b_hi - b_lo))
			return _rgb_to_hex(r, g, b)

	return colorscale[-1][1]


def voltage_to_color(
	v_pu: float, v_min: float = 0.95, v_max: float = 1.05
) -> str:
	"""将标幺值电压映射为颜色。

	Args:
		v_pu: 母线电压标幺值
		v_min: 电压下限 (默认 0.95 pu)
		v_max: 电压上限 (默认 1.05 pu)

	Returns:
		十六进制颜色字符串
	"""
	v_range_lo = v_min - (v_max - v_min)
	v_range_hi = v_max + (v_max - v_min)
	normalized = (v_pu - v_range_lo) / (v_range_hi - v_range_lo)
	return _interpolate_colorscale(normalized, VOLTAGE_COLORSCALE)


def loading_to_color(loading_pct: float) -> str:
	"""将负载率百分比映射为颜色。

	Args:
		loading_pct: 负载率百分比 (0-100+)

	Returns:
		十六进制颜色字符串
	"""
	normalized = loading_pct / 100.0
	return _interpolate_colorscale(normalized, LOADING_COLORSCALE)


def soc_to_color(
	soc: float, soc_min: float = 0.2, soc_max: float = 0.8
) -> str:
	"""将储能 SOC 映射为颜色。

	Args:
		soc: 荷电状态 (0.0 - 1.0)
		soc_min: SOC 下限警告阈值
		soc_max: SOC 上限警告阈值

	Returns:
		十六进制颜色字符串
	"""
	return _interpolate_colorscale(soc, SOC_COLORSCALE)


def restoration_to_color(progress: float) -> str:
	"""将恢复进度映射为颜色 (DSR 环境专用)。

	Args:
		progress: 恢复进度 (0.0 - 1.0)

	Returns:
		十六进制颜色字符串
	"""
	return _interpolate_colorscale(progress, RESTORATION_COLORSCALE)


def get_plotly_layout_defaults() -> Dict[str, Any]:
	"""获取 Plotly 图表的默认 layout 配置，暗色主题风格。

	Returns:
		可直接传入 plotly.graph_objects.Layout 的参数字典
	"""
	return {
		"template": "plotly_dark",
		"paper_bgcolor": COLORS["bg_dark"],
		"plot_bgcolor": COLORS["bg_card"],
		"font": {
			"family": "Inter, -apple-system, sans-serif",
			"color": COLORS["text_primary"],
			"size": 12,
		},
		"title": {
			"font": {"size": 16, "color": COLORS["text_primary"]},
			"x": 0.5,
			"xanchor": "center",
		},
		"xaxis": {
			"gridcolor": "rgba(255,255,255,0.1)",
			"zerolinecolor": "rgba(255,255,255,0.2)",
		},
		"yaxis": {
			"gridcolor": "rgba(255,255,255,0.1)",
			"zerolinecolor": "rgba(255,255,255,0.2)",
		},
		"legend": {
			"bgcolor": "rgba(0,0,0,0.3)",
			"font": {"color": COLORS["text_secondary"]},
		},
		"margin": {"l": 60, "r": 30, "t": 50, "b": 50},
	}


def get_matplotlib_style() -> Dict[str, Any]:
	"""获取 Matplotlib rcParams 配置字典。

	Returns:
		matplotlib rcParams 字典
	"""
	return {
		"figure.facecolor": COLORS["bg_dark"],
		"axes.facecolor": COLORS["bg_card"],
		"text.color": COLORS["text_primary"],
		"axes.labelcolor": COLORS["text_primary"],
		"xtick.color": COLORS["text_secondary"],
		"ytick.color": COLORS["text_secondary"],
		"axes.grid": True,
		"grid.color": "rgba(255,255,255,0.1)",
		"grid.alpha": 0.3,
		"grid.linestyle": "--",
		"axes.edgecolor": "rgba(255,255,255,0.2)",
		"axes.linewidth": 0.8,
		"font.family": "sans-serif",
		"font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica", "Arial"],
		"font.size": 11,
		"axes.titlesize": 14,
		"axes.labelsize": 12,
		"legend.facecolor": COLORS["bg_card"],
		"legend.edgecolor": "rgba(255,255,255,0.2)",
		"legend.fontsize": 10,
		"lines.linewidth": 1.5,
		"lines.antialiased": True,
		"figure.figsize": [10, 6],
		"figure.dpi": 100,
		"savefig.dpi": 150,
		"savefig.bbox": "tight",
	}
