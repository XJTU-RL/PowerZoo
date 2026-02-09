"""
PowerZoo SmartGrid 智能电网: 交互式 CMDP 环境演示
HuggingFace Spaces 应用，基于 Gradio + Plotly。

5 个标签页: 概览 | 电压热图 | 拉格朗日轨迹 | 组件状态 | 训练仪表盘

SmartGrid 是一个基于约束马尔可夫决策过程(CMDP)与拉格朗日松弛的模块化光伏集成环境。
360步年度回合（1步 = 1天），同构智能体控制电容器、调压器、电池和光伏系统。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gradio as gr

# === Monkey-patch: 修复 Gradio 与 Plotly 的 additionalProperties schema 错误 ===
_original_plot_init = gr.Plot.__init__


def _patched_plot_init(self, *args, **kwargs):
	_original_plot_init(self, *args, **kwargs)
	if hasattr(self, "schema") and isinstance(self.schema, dict):
		self.schema.pop("additionalProperties", None)


gr.Plot.__init__ = _patched_plot_init


# === 调色板 ===
COLORS = {
	"primary": "#10B981",      # 翡翠绿
	"secondary": "#059669",    # 深翡翠绿
	"accent": "#06B6D4",       # 青色
	"warning": "#F59E0B",      # 琥珀色
	"danger": "#EF4444",       # 红色
	"bg_card": "rgba(16, 185, 129, 0.05)",
	"grid": "rgba(255, 255, 255, 0.08)",
	"text": "#E2E8F0",
	"text_dim": "#94A3B8",
}

PLOTLY_LAYOUT_DEFAULTS = dict(
	template="plotly_dark",
	paper_bgcolor="rgba(0,0,0,0)",
	plot_bgcolor="rgba(0,0,0,0)",
	font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
	margin=dict(l=60, r=30, t=50, b=50),
	xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
	yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)

# 确定性随机数生成器，用于可复现的演示数据
RNG = np.random.default_rng(seed=42)


# ============================================================
# 演示数据生成器
# ============================================================

BUS_NAMES_13 = [
	"650", "632", "633", "634", "645", "646",
	"671", "680", "684", "611", "652", "692", "675",
]


def generate_voltage_heatmap_data() -> np.ndarray:
	"""生成 360x13 电压矩阵，包含季节性光伏模式。

	夏季光伏注入导致电压偏高；冬季负荷增大导致电压偏低。

	Returns:
		np.ndarray: 形状 (360, 13)，电压单位为标幺值(p.u.)。
	"""
	days = np.arange(360)
	n_buses = len(BUS_NAMES_13)

	# 基础电压曲线：年度正弦模式
	# 夏季峰值（约第180天）因光伏发电推高电压
	seasonal = 0.025 * np.sin(2 * np.pi * (days - 90) / 360)

	# 各母线基准偏移（部分母线天然偏高/偏低）
	bus_offset = RNG.uniform(-0.015, 0.015, size=n_buses)

	# 构建矩阵
	voltage = np.ones((360, n_buses))
	for b in range(n_buses):
		voltage[:, b] += seasonal + bus_offset[b]

	# 添加日波动噪声
	noise = RNG.normal(0, 0.008, size=(360, n_buses))
	voltage += noise

	# 注入真实越限场景：
	# 夏季下游母线（索引6-12）光伏过电压
	for b in range(6, n_buses):
		summer_mask = (days >= 120) & (days <= 240)
		voltage[summer_mask, b] += RNG.uniform(0.02, 0.045, size=summer_mask.sum())

	# 冬季重负荷母线（索引3, 10, 12）欠电压
	for b in [3, 10, 12]:
		winter_mask = (days <= 60) | (days >= 300)
		voltage[winter_mask, b] -= RNG.uniform(0.01, 0.035, size=winter_mask.sum())

	return np.clip(voltage, 0.90, 1.12)


def generate_lagrangian_data(n_episodes: int = 500) -> dict[str, np.ndarray]:
	"""生成拉格朗日乘子与约束违反率的CMDP收敛曲线。

	Lambda 从约10开始收敛到约2。
	违反率从约15%降至2%以下。

	Returns:
		dict，键: episodes, lambda_values, violation_rate
	"""
	episodes = np.arange(n_episodes)

	# Lambda 收敛：指数衰减 + 噪声
	lambda_base = 2.0 + 8.0 * np.exp(-episodes / 80)
	lambda_noise = RNG.normal(0, 0.3, size=n_episodes) * np.exp(-episodes / 200)
	lambda_values = np.clip(lambda_base + lambda_noise, 0.5, 12.0)

	# 约束违反率：类Sigmoid递减
	violation_base = 0.15 / (1 + np.exp((episodes - 100) / 40))
	violation_noise = RNG.uniform(-0.005, 0.005, size=n_episodes)
	violation_rate = np.clip(violation_base + violation_noise + 0.012, 0.0, 0.20)
	# 最后阶段稳定在2%以下
	violation_rate[400:] = np.clip(violation_rate[400:] * 0.6, 0.005, 0.02)

	return {
		"episodes": episodes,
		"lambda_values": lambda_values,
		"violation_rate": violation_rate * 100,  # 百分比
	}


def generate_component_data(day_of_year: int) -> dict[str, np.ndarray]:
	"""生成指定日期的24小时组件运行数据。

	季节变化影响光伏出力和电池充放电循环。

	Args:
		day_of_year: 0索引日期（0=1月1日, 90=4月1日, 181=7月1日, 272=10月1日）。

	Returns:
		dict，键: hours, cap_kvar, reg_tap, battery_soc, pv_kw, pv_curtail
	"""
	hours = np.arange(24)

	# 季节因子（0=冬季, 1=夏季峰值）
	season_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 90) / 360)

	# 电容器无功功率：根据负荷/电压切换
	# 午后偏高，夜间偏低
	load_pattern = np.array([
		0.3, 0.25, 0.2, 0.2, 0.25, 0.35, 0.5, 0.65,
		0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9,
		0.85, 0.9, 0.95, 0.85, 0.7, 0.55, 0.45, 0.35,
	])
	cap_kvar = load_pattern * 300 + RNG.normal(0, 15, size=24)
	cap_kvar = np.clip(cap_kvar, 0, 400)

	# 调压器分接头位置：-16 到 +16，跟踪电压偏差
	reg_base = np.array([
		2, 2, 3, 3, 2, 1, 0, -1,
		-2, -3, -4, -5, -6, -7, -6, -5,
		-4, -3, -2, -1, 0, 1, 2, 2,
	])
	# 夏季光伏推动分接头下调
	reg_tap = reg_base - int(season_factor * 4) + RNG.integers(-1, 2, size=24)
	reg_tap = np.clip(reg_tap, -16, 16)

	# 电池荷电状态：正午光伏充电，傍晚高峰放电
	soc = np.zeros(24)
	soc[0] = 50 + season_factor * 10  # 初始SOC
	for h in range(1, 24):
		if 9 <= h <= 15:  # 光伏充电
			soc[h] = soc[h - 1] + (3.0 + season_factor * 2.0) + RNG.normal(0, 0.5)
		elif 17 <= h <= 21:  # 傍晚放电
			soc[h] = soc[h - 1] - (4.0 + season_factor * 1.5) + RNG.normal(0, 0.5)
		else:
			soc[h] = soc[h - 1] + RNG.normal(0, 0.3)
	soc = np.clip(soc, 10, 95)

	# 光伏输出：以正午为中心的钟形曲线，随季节缩放
	pv_peak = 200 + season_factor * 300  # kW
	pv_raw = pv_peak * np.exp(-0.5 * ((hours - 12) / 2.8) ** 2)
	pv_raw[:6] = 0
	pv_raw[19:] = 0
	pv_noise = RNG.normal(0, 10, size=24) * (pv_raw > 0)
	pv_kw = np.clip(pv_raw + pv_noise, 0, 600)

	# 光伏弃光：仅在夏季正午存在过电压风险时发生
	pv_curtail = np.zeros(24)
	if season_factor > 0.6:
		curtail_hours = (hours >= 10) & (hours <= 15)
		pv_curtail[curtail_hours] = pv_kw[curtail_hours] * RNG.uniform(0.05, 0.20, size=curtail_hours.sum())

	return {
		"hours": hours,
		"cap_kvar": cap_kvar,
		"reg_tap": reg_tap.astype(float),
		"battery_soc": soc,
		"pv_kw": pv_kw,
		"pv_curtail": pv_curtail,
	}


def generate_training_data(n_episodes: int = 500) -> dict[str, np.ndarray]:
	"""生成CMDP训练曲线：奖励、拉格朗日目标函数分解、约束满足率。

	Returns:
		dict，键: episodes, rewards, primal_obj, dual_penalty,
		lagrangian_obj, constraint_satisfaction
	"""
	episodes = np.arange(n_episodes)

	# 回合奖励：从低值起步，逐步改善并伴有噪声
	reward_base = -50 + 45 * (1 - np.exp(-episodes / 120))
	reward_noise = RNG.normal(0, 3.0, size=n_episodes) * np.exp(-episodes / 300)
	rewards = reward_base + reward_noise

	# 原始目标 J(pi)：CMDP中的奖励部分
	primal_obj = rewards.copy()

	# 拉格朗日乘子（与lagrangian数据轨迹一致）
	lambda_vals = 2.0 + 8.0 * np.exp(-episodes / 80)
	lambda_noise = RNG.normal(0, 0.2, size=n_episodes) * np.exp(-episodes / 200)
	lambda_vals = np.clip(lambda_vals + lambda_noise, 0.5, 12.0)

	# 约束代价 g(pi)：违反幅度
	g_pi = 0.12 / (1 + np.exp((episodes - 100) / 40))
	g_noise = RNG.uniform(-0.003, 0.003, size=n_episodes)
	g_pi = np.clip(g_pi + g_noise + 0.008, 0.001, 0.2)
	g_pi[400:] = np.clip(g_pi[400:] * 0.5, 0.002, 0.015)

	# 对偶惩罚项：lambda * g(pi)
	dual_penalty = lambda_vals * g_pi * 100  # 放大以便观察

	# 拉格朗日目标函数：J(pi) - lambda * g(pi)
	lagrangian_obj = primal_obj - dual_penalty

	# 约束满足率：1 - 违反率
	violation_rate = g_pi / 0.12  # 归一化
	constraint_satisfaction = np.clip((1 - violation_rate) * 100, 50, 100)
	constraint_satisfaction[350:] = np.clip(
		constraint_satisfaction[350:] + RNG.uniform(0, 2, size=150), 96, 100
	)

	return {
		"episodes": episodes,
		"rewards": rewards,
		"primal_obj": primal_obj,
		"dual_penalty": dual_penalty,
		"lagrangian_obj": lagrangian_obj,
		"constraint_satisfaction": constraint_satisfaction,
	}


# 模块加载时预生成所有演示数据
VOLTAGE_DATA = generate_voltage_heatmap_data()
LAGRANGIAN_DATA = generate_lagrangian_data()
TRAINING_DATA = generate_training_data()


# ============================================================
# 图表工厂函数
# ============================================================

def plot_voltage_heatmap(time_window: str = "全年") -> go.Figure:
	"""创建电压热图，y轴为母线名称，x轴为天数。

	Args:
		time_window: '全年', '第一季度', '第二季度', '第三季度', '第四季度' 之一。

	Returns:
		带标注的Plotly热图。
	"""
	# 按季度切片
	slices = {
		"全年": (0, 360),
		"第一季度 (1-3月)": (0, 90),
		"第二季度 (4-6月)": (90, 180),
		"第三季度 (7-9月)": (180, 270),
		"第四季度 (10-12月)": (270, 360),
	}
	start, end = slices.get(time_window, (0, 360))
	data = VOLTAGE_DATA[start:end, :].T  # 形状: (n_buses, n_days)
	days = list(range(start, end))

	# 自定义色阶：蓝色(低) -> 绿色(正常) -> 红色(高)
	colorscale = [
		[0.0, "#2563EB"],    # 蓝色：严重欠电压
		[0.25, "#3B82F6"],   # 蓝色：欠电压
		[0.40, "#10B981"],   # 绿色：进入安全区
		[0.50, "#059669"],   # 深绿：额定 1.0 p.u.
		[0.60, "#10B981"],   # 绿色：离开安全区
		[0.75, "#F59E0B"],   # 琥珀：过电压警告
		[1.0, "#EF4444"],    # 红色：严重过电压
	]

	fig = go.Figure(data=go.Heatmap(
		z=data,
		x=days,
		y=BUS_NAMES_13,
		colorscale=colorscale,
		zmin=0.92,
		zmax=1.10,
		colorbar=dict(
			title=dict(text="电压 (标幺值)", side="right"),
			tickvals=[0.93, 0.95, 1.00, 1.05, 1.08],
			ticktext=["0.93", "0.95", "1.00", "1.05", "1.08"],
		),
		hovertemplate=(
			"天数: %{x}<br>"
			"母线: %{y}<br>"
			"电压: %{z:.4f} 标幺值"
			"<extra></extra>"
		),
	))

	# 添加越限边界注释
	fig.add_hline(y=None)  # hlines 不适用于热图 y 轴
	# 使用文字标注电压限值
	fig.add_annotation(
		text="V_min=0.95 | V_max=1.05",
		xref="paper", yref="paper",
		x=1.0, y=1.05,
		showarrow=False,
		font=dict(size=11, color=COLORS["warning"]),
		xanchor="right",
	)

	layout_overrides = {**PLOTLY_LAYOUT_DEFAULTS}
	layout_overrides["yaxis"] = dict(
		gridcolor=COLORS["grid"],
		zerolinecolor=COLORS["grid"],
		type="category",
		title_text="母线名称",
	)
	fig.update_layout(
		**layout_overrides,
		height=520,
		title=f"母线电压分布 - {time_window}",
		xaxis_title="年度天数",
	)
	return fig


def plot_lagrangian_trajectory() -> go.Figure:
	"""创建双子图：拉格朗日乘子收敛 + 约束违反率随回合变化。

	Returns:
		1x2 子图的 Plotly Figure。
	"""
	data = LAGRANGIAN_DATA
	ep = data["episodes"]

	fig = make_subplots(
		rows=1, cols=2,
		subplot_titles=(
			"拉格朗日乘子 (lambda) 收敛曲线",
			"电压约束违反率",
		),
		horizontal_spacing=0.12,
	)

	# Lambda 收敛
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["lambda_values"],
			mode="lines",
			name="lambda",
			line=dict(color=COLORS["primary"], width=1.8),
			opacity=0.7,
		),
		row=1, col=1,
	)
	# 平滑后的 lambda（滑动平均）
	window = 20
	lambda_smooth = np.convolve(data["lambda_values"], np.ones(window) / window, mode="valid")
	fig.add_trace(
		go.Scatter(
			x=ep[window - 1:], y=lambda_smooth,
			mode="lines",
			name="lambda (平滑)",
			line=dict(color=COLORS["warning"], width=2.5),
		),
		row=1, col=1,
	)
	# 目标 lambda 参考线
	fig.add_hline(
		y=2.0, line_dash="dot", line_color=COLORS["text_dim"],
		annotation_text="收敛值 ~2.0",
		annotation_position="bottom right",
		row=1, col=1,
	)

	# 约束违反率
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["violation_rate"],
			mode="lines",
			name="违反率 %",
			line=dict(color=COLORS["danger"], width=1.8),
			opacity=0.7,
		),
		row=1, col=2,
	)
	# 平滑后的违反率
	viol_smooth = np.convolve(data["violation_rate"], np.ones(window) / window, mode="valid")
	fig.add_trace(
		go.Scatter(
			x=ep[window - 1:], y=viol_smooth,
			mode="lines",
			name="违反率 % (平滑)",
			line=dict(color=COLORS["accent"], width=2.5),
		),
		row=1, col=2,
	)
	# 目标违反率阈值
	fig.add_hline(
		y=2.0, line_dash="dot", line_color=COLORS["text_dim"],
		annotation_text="目标 < 2%",
		annotation_position="bottom right",
		row=1, col=2,
	)

	fig.update_xaxes(title_text="训练回合", row=1, col=1)
	fig.update_xaxes(title_text="训练回合", row=1, col=2)
	fig.update_yaxes(title_text="拉格朗日乘子 (lambda)", row=1, col=1)
	fig.update_yaxes(title_text="约束违反率 (%)", row=1, col=2)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=450,
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
	)
	return fig


def plot_component_status(season: str = "第1天 (冬季)") -> go.Figure:
	"""创建分组条形/折线图，展示24小时组件运行状态。

	Args:
		season: 季节日期选项之一。

	Returns:
		包含4种组件类型的 2x2 子图 Plotly Figure。
	"""
	day_map = {
		"第1天 (冬季)": 1,
		"第91天 (春季)": 91,
		"第182天 (夏季)": 182,
		"第273天 (秋季)": 273,
	}
	day = day_map.get(season, 1)
	data = generate_component_data(day)
	hours = data["hours"]

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"电容器无功功率",
			"调压器分接头位置",
			"电池荷电状态",
			"光伏输出与弃光",
		),
		vertical_spacing=0.15,
		horizontal_spacing=0.10,
	)

	# 电容器 kVar
	fig.add_trace(
		go.Bar(
			x=hours, y=data["cap_kvar"],
			name="电容器 kVar",
			marker_color=COLORS["primary"],
			opacity=0.85,
		),
		row=1, col=1,
	)

	# 调压器分接头
	fig.add_trace(
		go.Scatter(
			x=hours, y=data["reg_tap"],
			mode="lines+markers",
			name="调压器分接头",
			line=dict(color=COLORS["accent"], width=2),
			marker=dict(size=6),
		),
		row=1, col=2,
	)
	fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_dim"], row=1, col=2)

	# 电池 SOC
	fig.add_trace(
		go.Scatter(
			x=hours, y=data["battery_soc"],
			mode="lines+markers",
			name="荷电状态 %",
			line=dict(color=COLORS["warning"], width=2.5),
			marker=dict(size=5),
			fill="tozeroy",
			fillcolor="rgba(245, 158, 11, 0.1)",
		),
		row=2, col=1,
	)
	# SOC 上下限
	fig.add_hline(y=20, line_dash="dot", line_color=COLORS["danger"], row=2, col=1)
	fig.add_hline(y=90, line_dash="dot", line_color=COLORS["danger"], row=2, col=1)

	# 光伏输出 + 弃光堆叠
	fig.add_trace(
		go.Bar(
			x=hours, y=data["pv_kw"] - data["pv_curtail"],
			name="光伏实际输出 (kW)",
			marker_color=COLORS["secondary"],
		),
		row=2, col=2,
	)
	fig.add_trace(
		go.Bar(
			x=hours, y=data["pv_curtail"],
			name="光伏弃光 (kW)",
			marker_color=COLORS["danger"],
			opacity=0.7,
		),
		row=2, col=2,
	)

	# 轴标签
	for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
		fig.update_xaxes(title_text="小时", row=row, col=col)
	fig.update_yaxes(title_text="无功功率 (kVar)", row=1, col=1)
	fig.update_yaxes(title_text="分接头位置", row=1, col=2)
	fig.update_yaxes(title_text="荷电状态 (%)", row=2, col=1)
	fig.update_yaxes(title_text="功率 (kW)", row=2, col=2)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=600,
		barmode="stack",
		title=f"组件运行状态 - {season}（第{day}天）",
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
	)
	return fig


def plot_training_rewards() -> go.Figure:
	"""绘制CMDP训练过程中的回合奖励曲线。

	Returns:
		包含原始与平滑奖励曲线的 Plotly Figure。
	"""
	data = TRAINING_DATA
	ep = data["episodes"]
	window = 20

	fig = go.Figure()

	# 原始奖励
	fig.add_trace(go.Scatter(
		x=ep, y=data["rewards"],
		mode="lines",
		name="回合奖励 (原始)",
		line=dict(color=COLORS["primary"], width=1),
		opacity=0.4,
	))

	# 平滑奖励
	smooth = np.convolve(data["rewards"], np.ones(window) / window, mode="valid")
	fig.add_trace(go.Scatter(
		x=ep[window - 1:], y=smooth,
		mode="lines",
		name="回合奖励 (平滑)",
		line=dict(color=COLORS["primary"], width=2.5),
	))

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title="约束马尔可夫决策过程(CMDP) 回合奖励 (SmartGrid 34母线光伏系统)",
		xaxis_title="训练回合",
		yaxis_title="回合奖励",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_lagrangian_decomposition() -> go.Figure:
	"""绘制拉格朗日目标函数分解：J(pi) - lambda*g(pi)。

	展示原始目标、对偶惩罚项和组合拉格朗日目标函数。

	Returns:
		包含三条叠加曲线的 Plotly Figure。
	"""
	data = TRAINING_DATA
	ep = data["episodes"]
	window = 25

	fig = go.Figure()

	# 辅助函数：平滑曲线
	def _smooth(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		s = np.convolve(arr, np.ones(window) / window, mode="valid")
		return ep[window - 1:], s

	# 原始目标 J(pi)
	x, y = _smooth(data["primal_obj"])
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode="lines",
		name="J(pi) 原始目标",
		line=dict(color=COLORS["accent"], width=2.2),
	))

	# 对偶惩罚 lambda*g(pi)
	x, y = _smooth(data["dual_penalty"])
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode="lines",
		name="lambda * g(pi) 对偶惩罚",
		line=dict(color=COLORS["danger"], width=2.2, dash="dash"),
	))

	# 拉格朗日目标 L = J(pi) - lambda*g(pi)
	x, y = _smooth(data["lagrangian_obj"])
	fig.add_trace(go.Scatter(
		x=x, y=y,
		mode="lines",
		name="L(pi, lambda) 拉格朗日目标",
		line=dict(color=COLORS["warning"], width=2.8),
	))

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title="拉格朗日目标函数分解: L = J(pi) - lambda * g(pi)",
		xaxis_title="训练回合",
		yaxis_title="目标函数值",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
	)
	return fig


def plot_constraint_satisfaction() -> go.Figure:
	"""绘制训练过程中的约束满足率。

	Returns:
		包含满足率和95%目标线的 Plotly Figure。
	"""
	data = TRAINING_DATA
	ep = data["episodes"]
	window = 20

	fig = go.Figure()

	# 原始
	fig.add_trace(go.Scatter(
		x=ep, y=data["constraint_satisfaction"],
		mode="lines",
		name="约束满足率 (原始)",
		line=dict(color=COLORS["secondary"], width=1),
		opacity=0.35,
	))

	# 平滑
	smooth = np.convolve(data["constraint_satisfaction"], np.ones(window) / window, mode="valid")
	fig.add_trace(go.Scatter(
		x=ep[window - 1:], y=smooth,
		mode="lines",
		name="约束满足率 (平滑)",
		line=dict(color=COLORS["secondary"], width=2.5),
	))

	# 95% 目标线
	fig.add_hline(
		y=95, line_dash="dot", line_color=COLORS["warning"],
		annotation_text="95% 目标",
		annotation_position="bottom right",
	)
	# 98% 优秀线
	fig.add_hline(
		y=98, line_dash="dot", line_color=COLORS["primary"],
		annotation_text="98% 优秀",
		annotation_position="top right",
	)

	layout_overrides = {**PLOTLY_LAYOUT_DEFAULTS}
	layout_overrides["yaxis"] = dict(
		range=[50, 102],
		gridcolor=COLORS["grid"],
		zerolinecolor=COLORS["grid"],
		title_text="满足率 (%)",
	)
	fig.update_layout(
		**layout_overrides,
		height=350,
		title="电压约束满足率",
		xaxis_title="训练回合",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


# ============================================================
# 构建 Gradio 应用
# ============================================================

def build_app() -> gr.Blocks:
	"""构建包含5个标签页的 Gradio Blocks 应用。"""
	with gr.Blocks(
		title="PowerZoo SmartGrid 智能电网: CMDP 环境演示",
		theme=gr.themes.Soft(primary_hue="emerald"),
	) as app:
		# 页眉
		gr.Markdown(
			"""
			# PowerZoo SmartGrid 智能电网：模块化光伏集成与约束马尔可夫决策过程(CMDP)
			**约束马尔可夫决策过程(CMDP)** | **拉格朗日松弛** | **360步年度回合** | **同构智能体**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# 标签页 1: 概览
			# --------------------------------------------------------
			with gr.Tab("概览"):
				gr.Markdown(
					"""
					## SmartGrid 智能电网环境

					SmartGrid 是一个基于**约束马尔可夫决策过程(CMDP)**框架与拉格朗日松弛的**模块化光伏集成环境**。
					与将电压约束折叠进奖励函数的标准MDP不同，SmartGrid 将**经济目标**（网损最小化、控制平滑性）
					与**安全约束**（电压调节在 0.95-1.05 标幺值范围内）分离处理。

					### 核心规格

					| 属性 | 值 |
					|------|-----|
					| **智能体类型** | 同构（共享策略） |
					| **回合长度** | 360步（1步 = 1天，年度循环） |
					| **框架** | 约束马尔可夫决策过程(CMDP) + 拉格朗日乘子 |
					| **被控设备** | 电容器、调压器、电池、光伏系统 |
					| **电压限值** | 0.95 - 1.05 标幺值 (ANSI C84.1) |
					| **奖励** | 网损 + 控制代价 + 光伏利用率 |
					| **约束** | 电压越限率（平方铰链损失） |

					### SmartGrid 的独特之处

					- **CMDP 建模**：拉格朗日乘子 `lambda` 自动平衡经济性能与电压安全，无需手动调节惩罚系数。
					- **面向对象的电路建模**：基于组件的架构（电容器、调压器、电池、光伏），SmartGrid 自有的 `Circuit` 类封装 OpenDSS。
					- **年度回合**：360步回合捕捉季节性负荷/光伏变化，使智能体能学习长期策略。
					- **课程学习**：三阶段训练（探索 -> 优化 -> 精细化），逐步收紧约束。

					### 支持的 IEEE 标准测试系统

					| 系统 | 母线数 | 智能体数 | 复杂度 |
					|------|--------|----------|--------|
					| **13母线** | 13 | 2-4 | 快速原型验证 |
					| **34母线_PV** | 34 | 6-9 | 光伏集成研究 |
					| **123母线** | 123 | 12-20 | 中等规模验证 |
					| **8500节点** | 8500 | 50+ | 大规模压力测试 |

					光伏变体可选：保守型、优化型、激进型渗透水平。

					### CMDP 优化目标

					智能体优化拉格朗日函数：

					**L(pi, lambda) = J(pi) - lambda * g(pi)**

					其中 `J(pi)` 是原始奖励（经济目标），`g(pi)` 是约束代价（电压越限），
					`lambda` 是通过梯度上升更新的对偶变量。
					"""
				)

			# --------------------------------------------------------
			# 标签页 2: 电压热图
			# --------------------------------------------------------
			with gr.Tab("电压热图"):
				gr.Markdown(
					"""
					## 母线电压热图（IEEE 13母线演示）
					可视化全年所有母线的电压分布。
					**蓝色** = 欠电压 (<0.95)，**绿色** = 正常 (0.95-1.05)，**红色** = 过电压 (>1.05)。
					夏季光伏注入导致下游母线过电压；冬季负荷拉低电压。
					"""
				)
				time_dropdown = gr.Dropdown(
					choices=["全年", "第一季度 (1-3月)", "第二季度 (4-6月)", "第三季度 (7-9月)", "第四季度 (10-12月)"],
					value="全年",
					label="时间窗口",
				)
				heatmap_plot = gr.Plot(value=plot_voltage_heatmap("全年"))

				time_dropdown.change(
					fn=plot_voltage_heatmap,
					inputs=time_dropdown,
					outputs=heatmap_plot,
				)

			# --------------------------------------------------------
			# 标签页 3: 拉格朗日轨迹
			# --------------------------------------------------------
			with gr.Tab("拉格朗日轨迹"):
				gr.Markdown(
					"""
					## CMDP 拉格朗日收敛过程
					拉格朗日乘子 `lambda` 从高值（约10）开始以强制严格电压约束，
					随后收敛到约2，因为策略已学会自然满足约束。
					约束违反率从约15%降至2%以下。

					这种对偶收敛是基于拉格朗日松弛的CMDP训练的标志性行为。
					"""
				)
				lagrangian_plot = gr.Plot(value=plot_lagrangian_trajectory())

			# --------------------------------------------------------
			# 标签页 4: 组件状态
			# --------------------------------------------------------
			with gr.Tab("组件状态"):
				gr.Markdown(
					"""
					## 24小时组件运行状态
					查看电容器、调压器、电池和光伏系统在一整天中的运行情况。
					选择不同季节，观察运行模式如何随负荷和光伏出力变化。
					"""
				)
				season_radio = gr.Radio(
					choices=["第1天 (冬季)", "第91天 (春季)", "第182天 (夏季)", "第273天 (秋季)"],
					value="第182天 (夏季)",
					label="选择年度日期",
				)
				component_plot = gr.Plot(value=plot_component_status("第182天 (夏季)"))

				season_radio.change(
					fn=plot_component_status,
					inputs=season_radio,
					outputs=component_plot,
				)

			# --------------------------------------------------------
			# 标签页 5: 训练仪表盘
			# --------------------------------------------------------
			with gr.Tab("训练仪表盘"):
				gr.Markdown(
					"""
					## 约束马尔可夫决策过程(CMDP) 训练仪表盘 (SmartGrid 34母线光伏系统)
					训练曲线同时展示原始目标（奖励）和对偶目标（约束）的收敛过程。
					拉格朗日目标函数分解为 J(pi) 和惩罚项 lambda * g(pi)。
					"""
				)

				gr.Markdown("### 回合奖励")
				reward_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### 拉格朗日目标函数分解")
				decomp_plot = gr.Plot(value=plot_lagrangian_decomposition())

				gr.Markdown("### 约束满足率")
				constraint_plot = gr.Plot(value=plot_constraint_satisfaction())

		# 页脚
		gr.Markdown(
			"""
			---
			**PowerZoo** · MIT 许可证 · [XJTU-RL](https://github.com/XJTU-RL) · IEEE TSG 2025
			"""
		)

	return app


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(server_name="0.0.0.0", server_port=7860, share=False)
