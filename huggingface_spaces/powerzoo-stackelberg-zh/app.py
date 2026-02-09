"""
PowerZoo Stackelberg 电力市场博弈: HuggingFace Space 交互式演示

5 个标签页: 概览 | 电价信号 | 主从博弈动态 | 市场均衡 | 训练仪表盘

独立的 Gradio + Plotly 应用。所有演示数据在代码中直接生成。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gradio as gr

# === Monkey-patch: fix Gradio additionalProperties schema error with Plotly ===
_original_plot_init = gr.Plot.__init__


def _patched_plot_init(self, *args, **kwargs):
	_original_plot_init(self, *args, **kwargs)
	if hasattr(self, "schema") and isinstance(self.schema, dict):
		self.schema.pop("additionalProperties", None)


gr.Plot.__init__ = _patched_plot_init


# ============================================================
# Color Palette & Theming
# ============================================================

COLORS = {
	"primary": "#F59E0B",       # amber
	"secondary": "#D97706",     # darker amber
	"accent": "#EF4444",        # red
	"uc": "#F59E0B",            # UC (leader) color
	"consumer": "#8B5CF6",      # consumer (follower) color
	"bg_card": "rgba(245, 158, 11, 0.05)",
	"grid": "rgba(255, 255, 255, 0.08)",
	"text": "#E5E7EB",
	"text_dim": "#9CA3AF",
	"cost_colors": ["#F59E0B", "#8B5CF6", "#3B82F6", "#EF4444"],
}

PLOTLY_LAYOUT_DEFAULTS = dict(
	template="plotly_dark",
	paper_bgcolor="rgba(0,0,0,0)",
	plot_bgcolor="rgba(17,24,39,0.6)",
	font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
	margin=dict(l=60, r=60, t=50, b=50),
	hovermode="x unified",
)

HOURS = list(range(24))


# ============================================================
# Demo Data Generation
# ============================================================

def _seed() -> np.random.Generator:
	"""确定性随机数生成器，用于生成可复现的演示数据。"""
	return np.random.default_rng(42)


def gen_tou_price() -> np.ndarray:
	"""分时电价基准时间表 ($/kWh)，24小时。"""
	tou = np.zeros(24)
	for h in range(24):
		if 0 <= h < 7 or 22 <= h <= 23:
			tou[h] = 0.05   # 谷时
		elif 11 <= h < 17:
			tou[h] = 0.15   # 峰时
		else:
			tou[h] = 0.10   # 平时
	return tou


def gen_uc_dynamic_price(tou: np.ndarray) -> np.ndarray:
	"""电力公司学习得到的动态电价，围绕分时基准进行自适应调整。"""
	rng = _seed()
	# 电力公司学习在峰时略微降价、在谷时提价、平滑过渡
	delta = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			delta[h] = rng.uniform(-0.03, -0.01)   # 峰时略微折扣
		elif 7 <= h < 11:
			delta[h] = rng.uniform(0.005, 0.02)     # 峰前提价
		elif 17 <= h < 22:
			delta[h] = rng.uniform(0.0, 0.015)      # 晚间溢价
		else:
			delta[h] = rng.uniform(-0.005, 0.01)    # 谷时噪声
	# 移动平均平滑
	kernel = np.array([0.15, 0.25, 0.35, 0.25])
	delta_smooth = np.convolve(delta, kernel / kernel.sum(), mode="same")
	return np.clip(tou + delta_smooth, 0.03, 0.20)


def gen_dr_incentive() -> np.ndarray:
	"""电力公司提供的需求响应激励信号 ($/kWh)。"""
	rng = _seed()
	dr = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			dr[h] = rng.uniform(0.03, 0.08)    # 峰时高激励
		elif 7 <= h < 11 or 17 <= h < 22:
			dr[h] = rng.uniform(0.01, 0.04)     # 肩时中等激励
		else:
			dr[h] = rng.uniform(0.0, 0.01)      # 谷时最低激励
	return dr


def gen_uc_actions() -> np.ndarray:
	"""电力公司5维动作，24步: energy_price, dr_incentive, ess_charge, ess_discharge, reserve_margin。"""
	rng = _seed()
	tou = gen_tou_price()
	actions = np.zeros((24, 5))

	# energy_price: 围绕分时电价归一化
	actions[:, 0] = (gen_uc_dynamic_price(tou) - 0.03) / 0.17

	# dr_incentive: 归一化 [0,1]
	actions[:, 1] = gen_dr_incentive() / 0.10

	# ess_charge: 谷时充电更多
	for h in range(24):
		if h < 7 or h >= 22:
			actions[h, 2] = rng.uniform(0.4, 0.8)
		elif 11 <= h < 17:
			actions[h, 2] = rng.uniform(0.0, 0.15)
		else:
			actions[h, 2] = rng.uniform(0.1, 0.3)

	# ess_discharge: 峰时放电更多
	for h in range(24):
		if 11 <= h < 17:
			actions[h, 3] = rng.uniform(0.5, 0.9)
		elif 7 <= h < 11 or 17 <= h < 22:
			actions[h, 3] = rng.uniform(0.1, 0.35)
		else:
			actions[h, 3] = rng.uniform(0.0, 0.1)

	# reserve_margin: 相对稳定
	actions[:, 4] = rng.uniform(0.3, 0.6, size=24)

	return actions


def gen_consumer_actions() -> np.ndarray:
	"""平均消费者3维动作，24步: load_shift, der_output, flexibility_bid。"""
	rng = _seed()
	actions = np.zeros((24, 3))

	for h in range(24):
		# load_shift: 消费者在峰时转移负荷
		if 11 <= h < 17:
			actions[h, 0] = rng.uniform(-0.6, -0.2)  # 削减峰荷
		elif h < 7 or h >= 22:
			actions[h, 0] = rng.uniform(0.1, 0.5)    # 吸收转移负荷
		else:
			actions[h, 0] = rng.uniform(-0.15, 0.15)

		# der_output: 大致跟随太阳能曲线
		solar_factor = max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
		actions[h, 1] = solar_factor * rng.uniform(0.5, 0.95)

		# flexibility_bid: 需求响应激励越高出价越高
		if 11 <= h < 17:
			actions[h, 2] = rng.uniform(0.5, 0.85)
		else:
			actions[h, 2] = rng.uniform(0.1, 0.4)

	return actions


def gen_reward_curves() -> tuple[np.ndarray, np.ndarray]:
	"""单回合内电力公司收益和消费者总效用，24步。"""
	rng = _seed()
	uc_reward = np.zeros(24)
	consumer_reward = np.zeros(24)

	for h in range(24):
		# 电力公司收益: 售电收入 - 储能成本 - 需求响应成本
		base_rev = rng.uniform(2.0, 5.0)
		if 11 <= h < 17:
			base_rev += rng.uniform(1.0, 3.0)  # 峰时收入
		storage_cost = rng.uniform(0.3, 0.8)
		dr_cost = rng.uniform(0.2, 0.6) if 11 <= h < 17 else rng.uniform(0.0, 0.2)
		uc_reward[h] = base_rev - storage_cost - dr_cost

		# 消费者效用: 用电效用 - 电费支出 + 需求响应收益
		utility = rng.uniform(1.5, 4.0)
		elec_cost = rng.uniform(0.8, 2.5)
		if 11 <= h < 17:
			elec_cost += rng.uniform(0.5, 1.5)
		dr_benefit = rng.uniform(0.3, 1.0) if 11 <= h < 17 else rng.uniform(0.0, 0.3)
		consumer_reward[h] = utility - elec_cost + dr_benefit

	return uc_reward, consumer_reward


def gen_cost_breakdown() -> dict[str, np.ndarray]:
	"""系统成本分解: 发电、需求响应、储能、惩罚，24步。"""
	rng = _seed()
	costs = {}

	# 发电成本跟随负荷曲线
	gen_base = np.array([
		3.2, 2.8, 2.5, 2.3, 2.2, 2.4, 3.0, 4.5,
		5.8, 6.2, 6.5, 7.8, 8.5, 8.2, 7.9, 7.5,
		7.0, 7.8, 7.2, 6.0, 5.2, 4.5, 3.8, 3.4,
	])
	costs["generation_cost"] = gen_base + rng.uniform(-0.3, 0.3, size=24)

	# 需求响应成本: 电力公司为需求响应支付的费用
	dr_cost = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			dr_cost[h] = rng.uniform(0.8, 2.0)
		elif 7 <= h < 11 or 17 <= h < 22:
			dr_cost[h] = rng.uniform(0.2, 0.6)
		else:
			dr_cost[h] = rng.uniform(0.0, 0.1)
	costs["dr_cost"] = dr_cost

	# 储能成本: 充放电损耗
	storage_cost = np.zeros(24)
	for h in range(24):
		if h < 7 or h >= 22:
			storage_cost[h] = rng.uniform(0.3, 0.7)   # 充电成本
		elif 11 <= h < 17:
			storage_cost[h] = rng.uniform(0.2, 0.5)   # 放电损耗
		else:
			storage_cost[h] = rng.uniform(0.05, 0.2)
	costs["storage_cost"] = storage_cost

	# 惩罚: 电压越限与可靠性
	penalty = np.zeros(24)
	for h in range(24):
		if 11 <= h < 17:
			penalty[h] = rng.uniform(0.1, 0.6)   # 峰时压力
		else:
			penalty[h] = rng.uniform(0.0, 0.15)
	costs["penalty"] = penalty

	return costs


def gen_equilibrium_scatter(n_episodes: int = 200) -> pd.DataFrame:
	"""训练轨迹: 各回合的电力公司利润 vs 平均消费者效用。"""
	rng = _seed()
	episodes = np.arange(n_episodes)

	# 训练早期: 方差大，远离均衡
	# 训练后期: 收敛至帕累托前沿
	progress = episodes / n_episodes

	# 电力公司利润先提升后稳定
	uc_base = -5.0 + 18.0 * (1.0 - np.exp(-3.0 * progress))
	uc_noise = rng.normal(0.0, 2.5 * (1.0 - 0.7 * progress), size=n_episodes)
	uc_profit = uc_base + uc_noise

	# 消费者效用: 提升较慢，与电力公司存在天然博弈张力
	consumer_base = -2.0 + 10.0 * (1.0 - np.exp(-2.5 * progress))
	consumer_noise = rng.normal(0.0, 1.8 * (1.0 - 0.6 * progress), size=n_episodes)
	consumer_utility = consumer_base + consumer_noise

	# 训练早期加入负相关（竞争），后期减弱（合作）
	competition_factor = 0.3 * (1.0 - progress)
	consumer_utility -= competition_factor * uc_noise

	return pd.DataFrame({
		"episode": episodes,
		"uc_profit": uc_profit,
		"consumer_utility": consumer_utility,
		"phase": np.where(progress < 0.3, "探索期", np.where(progress < 0.7, "学习期", "收敛期")),
	})


def gen_training_curves(n_episodes: int = 1000) -> dict[str, np.ndarray]:
	"""长期训练曲线: 电力公司收益、消费者收益、社会福利、电价稳定性。"""
	rng = _seed()
	episodes = np.arange(n_episodes)
	progress = episodes / n_episodes

	# 电力公司收益: 从负值开始，逐步攀升并稳定
	uc_raw = -15.0 + 40.0 * (1.0 - np.exp(-4.0 * progress))
	uc_noise = rng.normal(0.0, 3.0, size=n_episodes) * (1.0 - 0.5 * progress)
	uc_reward = uc_raw + uc_noise

	# 消费者收益（单个消费者，共4个）: 从负值开始，逐步改善
	c_raw = -8.0 + 20.0 * (1.0 - np.exp(-3.0 * progress))
	c_noise = rng.normal(0.0, 2.0, size=n_episodes) * (1.0 - 0.5 * progress)
	consumer_per = c_raw + c_noise

	# 社会福利 = 电力公司收益 + 4个消费者收益之和
	social_welfare = uc_reward + 4.0 * consumer_per

	# 电价稳定性: 24小时动态电价标准差，随电力公司学习而下降
	price_vol_raw = 0.06 - 0.035 * (1.0 - np.exp(-3.5 * progress))
	price_vol_noise = rng.uniform(-0.005, 0.005, size=n_episodes) * (1.0 - 0.6 * progress)
	price_stability = np.clip(price_vol_raw + price_vol_noise, 0.005, 0.08)

	return {
		"episodes": episodes,
		"uc_reward": uc_reward,
		"consumer_reward": consumer_per,
		"social_welfare": social_welfare,
		"price_stability": price_stability,
	}


# ============================================================
# Plot Factory Functions
# ============================================================

def plot_price_signal() -> go.Figure:
	"""标签页2: 24小时分时基准电价、电力公司动态电价和需求响应激励。"""
	tou = gen_tou_price()
	dynamic = gen_uc_dynamic_price(tou)
	dr = gen_dr_incentive()

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# 分时基准电价（阶梯函数）
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=tou.tolist(),
			mode="lines",
			name="分时电价基准",
			line=dict(color="#6B7280", width=2.5, shape="hv", dash="dot"),
			hovertemplate="时刻 %{x}<br>分时电价: $%{y:.3f}/kWh<extra></extra>",
		),
		secondary_y=False,
	)

	# 电力公司动态电价（平滑曲线）
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=dynamic.tolist(),
			mode="lines+markers",
			name="动态电价",
			line=dict(color=COLORS["uc"], width=3),
			marker=dict(size=6, symbol="diamond"),
			hovertemplate="时刻 %{x}<br>动态电价: $%{y:.3f}/kWh<extra></extra>",
		),
		secondary_y=False,
	)

	# 需求响应激励（柱状图）
	fig.add_trace(
		go.Bar(
			x=HOURS, y=dr.tolist(),
			name="需求响应激励",
			marker_color=COLORS["consumer"],
			opacity=0.6,
			hovertemplate="时刻 %{x}<br>需求响应: $%{y:.3f}/kWh<extra></extra>",
		),
		secondary_y=True,
	)

	# 时段标注
	fig.add_vrect(x0=-0.5, x1=6.5, fillcolor="rgba(59,130,246,0.06)", line_width=0,
		annotation_text="谷时", annotation_position="top left",
		annotation=dict(font=dict(size=10, color=COLORS["text_dim"])))
	fig.add_vrect(x0=10.5, x1=16.5, fillcolor="rgba(239,68,68,0.06)", line_width=0,
		annotation_text="峰时", annotation_position="top left",
		annotation=dict(font=dict(size=10, color=COLORS["text_dim"])))

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=520,
		title=dict(text="24小时电价信号与需求响应激励", font=dict(size=16)),
		xaxis=dict(title="时刻", dtick=2, gridcolor=COLORS["grid"]),
		yaxis=dict(title="电价 ($/kWh)", gridcolor=COLORS["grid"],
			range=[0.0, 0.22]),
		yaxis2=dict(title="需求响应激励 ($/kWh)", gridcolor=COLORS["grid"],
			range=[0.0, 0.10], overlaying="y", side="right"),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
		barmode="overlay",
	)

	return fig


def plot_leader_follower() -> go.Figure:
	"""标签页3: 2x2子图展示主从博弈动态。"""
	uc_actions = gen_uc_actions()
	consumer_actions = gen_consumer_actions()
	uc_reward, consumer_reward = gen_reward_curves()
	costs = gen_cost_breakdown()

	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"电力公司动作 (5维)", "消费者动作 (3维, 4个智能体平均)",
			"电力公司 vs 消费者收益", "系统成本分解",
		),
		vertical_spacing=0.14,
		horizontal_spacing=0.12,
		specs=[
			[{"type": "heatmap"}, {"type": "heatmap"}],
			[{"type": "xy"}, {"type": "xy"}],
		],
	)

	# --- (1,1) 电力公司动作热力图 ---
	uc_labels = ["energy_price", "dr_incentive", "ess_charge", "ess_discharge", "reserve_margin"]
	fig.add_trace(
		go.Heatmap(
			z=uc_actions.T.tolist(),
			x=HOURS,
			y=uc_labels,
			colorscale=[[0, "#1E1B4B"], [0.5, "#F59E0B"], [1, "#FEF3C7"]],
			showscale=False,
			hovertemplate="时刻 %{x}<br>%{y}: %{z:.3f}<extra></extra>",
		),
		row=1, col=1,
	)

	# --- (1,2) 消费者动作热力图 ---
	consumer_labels = ["load_shift", "der_output", "flexibility_bid"]
	fig.add_trace(
		go.Heatmap(
			z=consumer_actions.T.tolist(),
			x=HOURS,
			y=consumer_labels,
			colorscale=[[0, "#1E1B4B"], [0.5, "#8B5CF6"], [1, "#EDE9FE"]],
			showscale=False,
			hovertemplate="时刻 %{x}<br>%{y}: %{z:.3f}<extra></extra>",
		),
		row=1, col=2,
	)

	# --- (2,1) 收益曲线 ---
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=uc_reward.tolist(),
			mode="lines+markers",
			name="电力公司收益",
			line=dict(color=COLORS["uc"], width=2.5),
			marker=dict(size=5),
		),
		row=2, col=1,
	)
	fig.add_trace(
		go.Scatter(
			x=HOURS, y=consumer_reward.tolist(),
			mode="lines+markers",
			name="消费者效用 (总计)",
			line=dict(color=COLORS["consumer"], width=2.5),
			marker=dict(size=5),
		),
		row=2, col=1,
	)

	# --- (2,2) 堆叠面积图: 成本分解 ---
	cost_names = ["generation_cost", "dr_cost", "storage_cost", "penalty"]
	display_names = ["发电成本", "需求响应成本", "储能成本", "惩罚成本"]
	# 将十六进制颜色转换为rgba以实现填充透明度
	_fill_rgba = [
		"rgba(245, 158, 11, 0.5)",   # #F59E0B
		"rgba(139, 92, 246, 0.5)",   # #8B5CF6
		"rgba(59, 130, 246, 0.5)",   # #3B82F6
		"rgba(239, 68, 68, 0.5)",    # #EF4444
	]
	for i, (key, dname) in enumerate(zip(cost_names, display_names)):
		fig.add_trace(
			go.Scatter(
				x=HOURS, y=costs[key].tolist(),
				mode="lines",
				name=dname,
				stackgroup="costs",
				line=dict(color=COLORS["cost_colors"][i], width=0.5),
				fillcolor=_fill_rgba[i],
			),
			row=2, col=2,
		)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=700,
		title=dict(text="主从博弈动态 (单回合)", font=dict(size=16)),
		showlegend=True,
		legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
	)

	# 坐标轴标签
	fig.update_xaxes(title_text="时刻", row=1, col=1, gridcolor=COLORS["grid"])
	fig.update_xaxes(title_text="时刻", row=1, col=2, gridcolor=COLORS["grid"])
	fig.update_xaxes(title_text="时刻", row=2, col=1, gridcolor=COLORS["grid"])
	fig.update_xaxes(title_text="时刻", row=2, col=2, gridcolor=COLORS["grid"])
	fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=1)
	fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=2)
	fig.update_yaxes(title_text="收益", gridcolor=COLORS["grid"], row=2, col=1)
	fig.update_yaxes(title_text="成本 ($)", gridcolor=COLORS["grid"], row=2, col=2)

	return fig


def plot_market_equilibrium() -> go.Figure:
	"""标签页4: 电力公司利润 vs 消费者效用散点图，含帕累托前沿。"""
	df = gen_equilibrium_scatter(200)

	# 计算帕累托前沿
	sorted_df = df.sort_values("uc_profit", ascending=False).reset_index(drop=True)
	pareto_uc: list[float] = []
	pareto_cu: list[float] = []
	max_cu = -np.inf
	for _, row in sorted_df.iterrows():
		if row["consumer_utility"] > max_cu:
			max_cu = row["consumer_utility"]
			pareto_uc.append(row["uc_profit"])
			pareto_cu.append(row["consumer_utility"])
	# 按顺序排列用于绘制线图
	pareto_order = np.argsort(pareto_uc)
	pareto_uc = [pareto_uc[i] for i in pareto_order]
	pareto_cu = [pareto_cu[i] for i in pareto_order]

	fig = go.Figure()

	# 按回合编号着色的散点图
	fig.add_trace(
		go.Scatter(
			x=df["uc_profit"].tolist(),
			y=df["consumer_utility"].tolist(),
			mode="markers",
			marker=dict(
				size=7,
				color=df["episode"].tolist(),
				colorscale=[
					[0.0, "#1E3A5F"],
					[0.3, "#3B82F6"],
					[0.7, "#F59E0B"],
					[1.0, "#FEF3C7"],
				],
				colorbar=dict(title="回合", tickfont=dict(color=COLORS["text"])),
				opacity=0.7,
				line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
			),
			text=[f"回合 {e} ({p})" for e, p in zip(df["episode"], df["phase"])],
			hovertemplate="电力公司利润: %{x:.2f}<br>消费者效用: %{y:.2f}<br>%{text}<extra></extra>",
			name="各回合",
		)
	)

	# 帕累托前沿
	fig.add_trace(
		go.Scatter(
			x=pareto_uc,
			y=pareto_cu,
			mode="lines+markers",
			name="帕累托前沿",
			line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
			marker=dict(size=4, color=COLORS["accent"]),
		)
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=560,
		title=dict(text="市场均衡: 电力公司利润 vs 消费者效用", font=dict(size=16)),
		xaxis=dict(title="电力公司利润 ($/回合)", gridcolor=COLORS["grid"]),
		yaxis=dict(title="平均消费者效用 ($/回合)", gridcolor=COLORS["grid"]),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


def compute_equilibrium_metrics() -> pd.DataFrame:
	"""市场均衡的汇总指标表。"""
	df = gen_equilibrium_scatter(200)
	converged = df[df["episode"] >= 140]
	return pd.DataFrame({
		"指标": [
			"平均电力公司利润 (收敛后)",
			"平均消费者效用 (收敛后)",
			"社会福利 (收敛后)",
			"电价波动率 (最后50回合)",
		],
		"数值": [
			f"${converged['uc_profit'].mean():.2f}",
			f"${converged['consumer_utility'].mean():.2f}",
			f"${(converged['uc_profit'] + 4 * converged['consumer_utility']).mean():.2f}",
			f"{gen_training_curves(1000)['price_stability'][-50:].mean():.4f} $/kWh",
		],
	})


def plot_training_rewards() -> go.Figure:
	"""标签页5: 电力公司和消费者收益曲线，双Y轴。"""
	data = gen_training_curves(1000)
	ep = data["episodes"].tolist()

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# 电力公司收益
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["uc_reward"].tolist(),
			mode="lines",
			name="电力公司收益",
			line=dict(color=COLORS["uc"], width=1.5),
			opacity=0.4,
		),
		secondary_y=False,
	)
	# 电力公司收益平滑（滚动50回合）
	uc_smooth = pd.Series(data["uc_reward"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=uc_smooth,
			mode="lines",
			name="电力公司收益 (平滑)",
			line=dict(color=COLORS["uc"], width=3),
		),
		secondary_y=False,
	)

	# 消费者收益
	fig.add_trace(
		go.Scatter(
			x=ep, y=data["consumer_reward"].tolist(),
			mode="lines",
			name="消费者效用",
			line=dict(color=COLORS["consumer"], width=1.5),
			opacity=0.4,
		),
		secondary_y=True,
	)
	# 消费者收益平滑
	c_smooth = pd.Series(data["consumer_reward"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=c_smooth,
			mode="lines",
			name="消费者效用 (平滑)",
			line=dict(color=COLORS["consumer"], width=3),
		),
		secondary_y=True,
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=480,
		title=dict(text="训练收益曲线: 电力公司 (UC) (领导者) vs 消费者 (跟随者)",
			font=dict(size=16)),
		xaxis=dict(title="回合", gridcolor=COLORS["grid"]),
		yaxis=dict(
			title=dict(text="电力公司收益", font=dict(color=COLORS["uc"])),
			gridcolor=COLORS["grid"],
		),
		yaxis2=dict(
			title=dict(text="消费者效用 (单智能体)", font=dict(color=COLORS["consumer"])),
			gridcolor=COLORS["grid"],
			overlaying="y", side="right",
		),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


def plot_social_welfare() -> go.Figure:
	"""标签页5: 训练过程中的社会福利。"""
	data = gen_training_curves(1000)
	ep = data["episodes"].tolist()

	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=ep, y=data["social_welfare"].tolist(),
			mode="lines",
			name="社会福利 (原始)",
			line=dict(color="#6B7280", width=1),
			opacity=0.3,
		)
	)

	sw_smooth = pd.Series(data["social_welfare"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=sw_smooth,
			mode="lines",
			name="社会福利 (平滑)",
			line=dict(color="#10B981", width=3),
		)
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title=dict(text="社会福利 (电力公司收益 + 4 x 消费者效用)", font=dict(size=14)),
		xaxis=dict(title="回合", gridcolor=COLORS["grid"]),
		yaxis=dict(title="社会福利 ($)", gridcolor=COLORS["grid"]),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


def plot_price_stability() -> go.Figure:
	"""标签页5: 训练过程中的电价稳定性指标。"""
	data = gen_training_curves(1000)
	ep = data["episodes"].tolist()

	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=ep, y=data["price_stability"].tolist(),
			mode="lines",
			name="电价波动率 (原始)",
			line=dict(color="#6B7280", width=1),
			opacity=0.3,
		)
	)

	ps_smooth = pd.Series(data["price_stability"]).rolling(50, min_periods=1).mean().tolist()
	fig.add_trace(
		go.Scatter(
			x=ep, y=ps_smooth,
			mode="lines",
			name="电价波动率 (平滑)",
			line=dict(color=COLORS["uc"], width=3),
		)
	)

	fig.update_layout(
		**PLOTLY_LAYOUT_DEFAULTS,
		height=400,
		title=dict(text="电价稳定性: 每小时电价标准差随训练变化",
			font=dict(size=14)),
		xaxis=dict(title="回合", gridcolor=COLORS["grid"]),
		yaxis=dict(title="电价标准差 ($/kWh)", gridcolor=COLORS["grid"]),
		legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
			bgcolor="rgba(0,0,0,0)"),
	)

	return fig


# ============================================================
# Build Gradio Application
# ============================================================

def build_app() -> gr.Blocks:
	"""构建包含5个标签页的 Gradio Blocks 应用。"""
	with gr.Blocks(
		title="PowerZoo Stackelberg 电力市场博弈",
		theme=gr.themes.Soft(primary_hue="amber"),
	) as app:

		# 页头
		gr.Markdown(
			"""
			# PowerZoo Stackelberg 电力市场博弈
			**异构多智能体强化学习** | **1个电力公司领导者 + N个消费者跟随者** | **Stackelberg-Nash均衡** | **IEEE TSG 2025**
			"""
		)

		with gr.Tabs():
			# ========================================
			# 标签页1: 概览
			# ========================================
			with gr.Tab("概览"):
				gr.Markdown(
					"""
					## Stackelberg 博弈环境

					Stackelberg 环境建模了一个电力配电网中**电力公司 (UC)** 与多个消费者之间的
					**双层非合作博弈**。电力公司作为 **Stackelberg 领导者**，制定电价策略并管理
					储能系统；消费者作为**跟随者**，根据电力公司的策略优化自身的负荷管理和分布式
					能源资源 (DER) 利用。

					该框架捕捉了放开管制电力市场中的根本博弈张力：电力公司通过战略定价和需求响应
					计划追求收入最大化，而消费者则致力于最小化成本、最大化用电效用。

					### 关键规格

					| 属性 | 取值 |
					|---|---|
					| **智能体结构** | 异构: 1个电力公司 (领导者) + 4个消费者 (跟随者) |
					| **回合长度** | 24步 (小时分辨率，一天) |
					| **电力公司动作空间** | 5维连续: `能源定价`, `需求响应激励`, `储能充电`, `储能放电`, `备用容量` |
					| **消费者动作空间** | 3维连续: `负荷转移`, `分布式能源输出`, `灵活性出价` |
					| **观测** | 系统状态 (电压、负荷、光伏) + 市场信号 (电价、分时电价、储能荷电状态) |
					| **奖励** | 电力公司: 收入 - 成本; 消费者: 效用 - 成本 + 需求响应收益 |
					| **博弈均衡** | Stackelberg-Nash: 电力公司先行承诺，消费者最优响应 |

					### 支持的IEEE标准测试系统

					| 系统 | 节点数 | 负荷数 | 应用场景 |
					|---|---|---|---|
					| **13节点** | 13 | 9 | 快速原型验证、算法调试 |
					| **34节点** | 34 | 22 | 标准基准测试、中等复杂度 |
					| **123节点** | 123 | 85 | 可扩展性测试、大规模实验 |

					### 博弈论结构

					该环境实现了一个 **Stackelberg 博弈**，其中：
					1. **电力公司 (领导者)** 公布下一时段的定价方案和需求响应激励
					2. **消费者 (跟随者)** 观测电力公司的策略并独立优化各自的响应
					3. 电力公司在制定策略时预判消费者的反应（预期定价）
					4. 消费者之间形成 **Nash 均衡**：任何单个消费者都无法通过单方面偏离获益

					这构成了一个丰富的学习问题：电力公司需要学会有效地*领导*，消费者需要学会
					最优地*跟随*——最终形成 Stackelberg-Nash 均衡。
					"""
				)

			# ========================================
			# 标签页2: 电价信号
			# ========================================
			with gr.Tab("电价信号"):
				gr.Markdown(
					"""
					### 24小时电力市场信号

					电力公司围绕分时电价 (TOU) 基准制定动态电价。在峰时时段 (11:00-17:00)，
					电力公司提供需求响应激励以缓解系统压力。定价与需求响应信号之间的交互作用
					驱动消费者行为，塑造市场均衡。
					"""
				)
				price_plot = gr.Plot(value=plot_price_signal())

				gr.Markdown(
					"""
					**图表解读**: 灰色虚线阶梯函数是管制分时基准电价。琥珀色曲线是电力公司
					学习得到的动态电价——注意它在峰时略微降价（以留住客户），同时在肩时提价
					（以获取额外收入）。紫色柱状图表示向削减峰荷的消费者提供的需求响应激励补贴。
					"""
				)

			# ========================================
			# 标签页3: 主从博弈动态
			# ========================================
			with gr.Tab("主从博弈动态"):
				gr.Markdown(
					"""
					### 单回合动作与结果分析

					可视化单个24小时回合内的 Stackelberg 交互过程。电力公司的5维动作向量
					和消费者的3维响应揭示了博弈学习中涌现的时序协调模式。
					"""
				)
				dynamics_plot = gr.Plot(value=plot_leader_follower())

				gr.Markdown(
					"""
					**关键观察**:
					- 电力公司在谷时充电储能（`ess_charge` 第3行夜间较暗）并在峰时放电
					  （`ess_discharge` 第4行中午较亮）
					- 消费者在峰时转移负荷（`load_shift` 中午为负值）并增加分布式能源输出
					  以跟随太阳能曲线
					- 收益曲线体现了天然的博弈张力：电力公司在峰时获利更多，但消费者成本
					  也随之增加——博弈寻求均衡点
					"""
				)

			# ========================================
			# 标签页4: 市场均衡
			# ========================================
			with gr.Tab("市场均衡"):
				gr.Markdown(
					"""
					### 向 Stackelberg-Nash 均衡收敛

					每个点代表一个训练回合，以电力公司利润 (x轴) vs 平均消费者效用 (y轴) 绘制。
					早期回合（深蓝色）表现为混沌探索。随着训练推进（琥珀色到白色），两方智能体
					逐步收敛至帕累托前沿——即任一方都无法在不损害另一方的情况下改善自身的边界。
					"""
				)

				with gr.Row():
					with gr.Column(scale=3):
						eq_plot = gr.Plot(value=plot_market_equilibrium())
					with gr.Column(scale=1):
						gr.Markdown("### 均衡指标")
						eq_table = gr.Dataframe(
							value=compute_equilibrium_metrics(),
							headers=["指标", "数值"],
							interactive=False,
						)

			# ========================================
			# 标签页5: 训练仪表盘
			# ========================================
			with gr.Tab("训练仪表盘"):
				gr.Markdown(
					"""
					### HAPPO 在 Stackelberg 13节点系统上的训练

					展示1000回合 HAPPO 实验的训练曲线，电力公司领导者和消费者跟随者同时学习。
					双收益曲线揭示了领导者策略与跟随者策略的共同演化过程。
					"""
				)
				reward_plot = gr.Plot(value=plot_training_rewards())

				with gr.Row():
					with gr.Column():
						sw_plot = gr.Plot(value=plot_social_welfare())
					with gr.Column():
						ps_plot = gr.Plot(value=plot_price_stability())

				gr.Markdown(
					"""
					**解读**: 社会福利（绿色）随着两方智能体在竞争框架内学会合作而提升。
					电价稳定性（琥珀色）随着电力公司学习到一致的定价策略而改善——早期探索
					导致电价高波动，随着策略成熟逐步稳定。
					"""
				)

		# 页脚
		gr.Markdown(
			"""
			---
			**PowerZoo** · MIT 许可证 · [XJTU-RL](https://github.com/XJTU-RL) · IEEE TSG 2025
			"""
		)

	return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(server_name="0.0.0.0", server_port=7860, share=False)
