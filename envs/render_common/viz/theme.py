"""
Render System Visual Theme (Common)
渲染系统统一视觉主题

定义 Plotly 和 Matplotlib 的统一样式，确保所有环境图表风格一致。
扩展 District Dispatch 配色，增加 VVC/SmartGrid/Stackelberg/DSR 设备色。
所有图表文本使用英文标注。
"""

from typing import Dict, Optional


# === Brand Colors ===
COLORS: Dict[str, str] = {
	"primary": "#4F46E5",
	"secondary": "#7C3AED",
	"success": "#10B981",
	"warning": "#F59E0B",
	"danger": "#EF4444",
	"info": "#3B82F6",
	"bg_dark": "#1a1a2e",
	"bg_card": "#16213e",
	"bg_plot": "#0f0f23",
	"text_primary": "#e0e0e0",
	"text_secondary": "#a0a0a0",
	"grid": "#2a2a4a",
	"border": "#3a3a5a",
}

# === Environment Accent Colors ===
ENV_COLORS: Dict[str, str] = {
	"vvc": "#4F46E5",           # Indigo
	"smartgrid": "#10B981",     # Emerald
	"stackelberg": "#F59E0B",   # Amber
	"dsr": "#EF4444",           # Red
	"district_dispatch": "#7C3AED",  # Violet
}

# === Zone Colors (Upstream -> Middle -> Downstream) ===
ZONE_COLORS = ["#4F46E5", "#10B981", "#F59E0B"]
ZONE_NAMES = ["Zone 0 (Upstream)", "Zone 1 (Middle)", "Zone 2 (Downstream)"]

# === Device Colors (共享 + 环境特有) ===
DEVICE_COLORS: Dict[str, str] = {
	# 通用设备
	"pv": "#F59E0B",
	"storage_charge": "#3B82F6",
	"storage_discharge": "#EF4444",
	"load": "#6B7280",
	"ev": "#7C3AED",
	"exchange_in": "#10B981",
	"exchange_out": "#F97316",
	# VVC 特有
	"capacitor_on": "#22D3EE",
	"capacitor_off": "#475569",
	"regulator_tap": "#A78BFA",
	"battery_soc": "#06B6D4",
	# SmartGrid 特有
	"lagrangian_lambda": "#EC4899",
	"constraint_violation": "#F43F5E",
	"objective_cost": "#8B5CF6",
	# Stackelberg 特有
	"uc_leader": "#F59E0B",
	"consumer_follower": "#3B82F6",
	"price_signal": "#F97316",
	"dr_signal": "#EC4899",
	"ess_power": "#06B6D4",
	# DSR 特有
	"switch_open": "#EF4444",
	"switch_closed": "#10B981",
	"fault_line": "#DC2626",
	"restored_load": "#22D3EE",
	"priority_critical": "#EF4444",
	"priority_high": "#F59E0B",
	"priority_medium": "#3B82F6",
	"priority_low": "#6B7280",
}

# === Agent Role Colors (异质环境) ===
AGENT_ROLE_COLORS: Dict[str, str] = {
	"uc_leader": "#F59E0B",
	"consumer": "#3B82F6",
	"switch_agent": "#EF4444",
	"pv_agent": "#F59E0B",
	"load_agent": "#10B981",
}

# === Voltage Color Scale (0.90 pu ~ 1.10 pu) ===
VOLTAGE_COLORSCALE = [
	[0.0, "#EF4444"],     # 0.90 pu (deep violation)
	[0.25, "#F59E0B"],    # 0.9375 pu (near violation)
	[0.5, "#10B981"],     # 1.00 pu (nominal)
	[0.75, "#F59E0B"],    # 1.0375 pu (near violation)
	[1.0, "#EF4444"],     # 1.10 pu (deep violation)
]

# === Loading Color Scale (0% ~ 150%) ===
LOADING_COLORSCALE = [
	[0.0, "#10B981"],     # 0% (no load)
	[0.5, "#F59E0B"],     # 75% (moderate)
	[0.67, "#EF4444"],    # 100% (overloaded)
	[1.0, "#991B1B"],     # 150% (critical)
]

# === SOC Color Scale (0% ~ 100%) ===
SOC_COLORSCALE = [
	[0.0, "#EF4444"],     # 0% (empty)
	[0.2, "#F59E0B"],     # 20% (low)
	[0.5, "#10B981"],     # 50% (healthy)
	[0.8, "#F59E0B"],     # 80% (high)
	[1.0, "#EF4444"],     # 100% (full)
]

# === Restoration Progress Color Scale (DSR) ===
RESTORATION_COLORSCALE = [
	[0.0, "#EF4444"],     # 0% (no restoration)
	[0.5, "#F59E0B"],     # 50% (partial)
	[1.0, "#10B981"],     # 100% (fully restored)
]


def get_plotly_layout(
	title: str = "",
	height: int = 500,
	width: Optional[int] = None,
	env_name: Optional[str] = None,
) -> dict:
	"""获取 Plotly 图表默认布局

	返回与暗色主题一致的 Plotly layout 参数字典，
	可直接解包传入 fig.update_layout(**layout)。

	Args:
		title: 图表标题（英文）
		height: 图表高度（像素）
		width: 图表宽度（像素），None 则自适应
		env_name: 环境名称，用于设置主色调

	Returns:
		dict，适用于 fig.update_layout(**layout)
	"""
	accent = ENV_COLORS.get(env_name, COLORS["primary"])

	layout = {
		"template": "plotly_dark",
		"paper_bgcolor": COLORS["bg_plot"],
		"plot_bgcolor": COLORS["bg_plot"],
		"font": {
			"color": COLORS["text_primary"],
			"family": "Inter, sans-serif",
		},
		"title": {
			"text": title,
			"font": {"size": 16, "color": accent},
		},
		"height": height,
		"margin": {"l": 60, "r": 30, "t": 50, "b": 50},
		"xaxis": {
			"gridcolor": COLORS["grid"],
			"zerolinecolor": COLORS["grid"],
		},
		"yaxis": {
			"gridcolor": COLORS["grid"],
			"zerolinecolor": COLORS["grid"],
		},
		"legend": {
			"bgcolor": "rgba(0,0,0,0.3)",
			"bordercolor": COLORS["border"],
		},
	}
	if width is not None:
		layout["width"] = width
	return layout


def get_matplotlib_rcparams() -> Dict[str, object]:
	"""获取 Matplotlib 统一样式参数

	返回可直接传入 matplotlib.rcParams.update() 的参数字典。

	Returns:
		rcParams 参数字典
	"""
	return {
		"figure.facecolor": COLORS["bg_dark"],
		"axes.facecolor": COLORS["bg_plot"],
		"axes.edgecolor": COLORS["border"],
		"axes.labelcolor": COLORS["text_primary"],
		"text.color": COLORS["text_primary"],
		"xtick.color": COLORS["text_secondary"],
		"ytick.color": COLORS["text_secondary"],
		"grid.color": COLORS["grid"],
		"grid.alpha": 0.3,
		"legend.facecolor": COLORS["bg_card"],
		"legend.edgecolor": COLORS["border"],
		"font.family": "sans-serif",
		"font.sans-serif": ["Inter", "DejaVu Sans", "Helvetica", "Arial"],
		"font.size": 10,
		"figure.dpi": 150,
		"savefig.dpi": 300,
		"savefig.bbox": "tight",
	}


def apply_matplotlib_theme() -> None:
	"""应用 Matplotlib 主题到全局 rcParams"""
	import matplotlib
	matplotlib.rcParams.update(get_matplotlib_rcparams())
