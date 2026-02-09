"""
PowerZoo: Interactive Web Showcase (Chinese Version)
HuggingFace Spaces application with Gradio + Plotly.

6 Tabs: Project Overview | Power System Explorer | Data Visualization | Training Dashboard | Algorithm Comparison | Architecture Diagrams
"""
import json
import os
from pathlib import Path

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

# === Data Loading ===
DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"

with open(DATA_DIR / "loadshapes.json") as f:
	LOADSHAPES = json.load(f)

with open(DATA_DIR / "pv_sample.json") as f:
	PV_DATA = json.load(f)

with open(DATA_DIR / "environments.json") as f:
	ENVIRONMENTS = json.load(f)

with open(DATA_DIR / "algorithms.json") as f:
	ALGORITHMS = json.load(f)

with open(DATA_DIR / "sample_training.json") as f:
	TRAINING = json.load(f)

# Architecture diagram JSON data (Plotly figures)
ARCH_FIGS = {}
_ARCH_NAMES = [
	"algorithm_hierarchy", "training_pipeline", "runner_algorithm_matrix",
	"happo_family", "mappo_family", "dan_happo",
	"ddpg_family", "hasac", "value_decomposition", "twots_vvc",
]
for fig_name in _ARCH_NAMES:
	fig_path = DATA_DIR / f"{fig_name}.json"
	if fig_path.exists():
		ARCH_FIGS[fig_name] = go.Figure(json.loads(fig_path.read_text()))

# === Color Palette ===
COLORS = {
	"primary": "#1565c0",
	"secondary": "#5e35b1",
	"accent": "#2e7d32",
	"warning": "#e65100",
	"agents": ["#1565c0", "#5e35b1", "#2e7d32", "#e65100", "#c62828", "#00838f"],
	"loadshapes": ["#1565c0", "#e65100", "#2e7d32"],
}

# === Init figures (required for gr.Plot(value=fig) pattern) ===
_INIT_FIG = go.Figure()
_INIT_FIG.update_layout(
	template="plotly_white",
	height=400,
	margin=dict(l=40, r=40, t=40, b=40),
)


# ============================================================
# Plot Factory Functions
# ============================================================

def plot_load_profiles(selected_shapes: list[str]) -> go.Figure:
	"""Plot annual load curves for selected LoadShapes."""
	if not selected_shapes:
		fig = go.Figure()
		fig.update_layout(template="plotly_white", height=450)
		fig.add_annotation(text="请至少选择一个负荷曲线", showarrow=False, font=dict(size=16))
		return fig

	fig = go.Figure()
	for i, name in enumerate(selected_shapes):
		if name in LOADSHAPES:
			data = LOADSHAPES[name]
			# X-axis: approximate hours across the year (730 points, each ~12h apart)
			x = np.linspace(0, 8760, len(data))
			fig.add_trace(go.Scatter(
				x=x, y=data,
				mode="lines",
				name=name,
				line=dict(color=COLORS["loadshapes"][i % 3], width=1.5),
				opacity=0.85,
			))

	fig.update_layout(
		template="plotly_white",
		height=450,
		title="年度负荷曲线 (8760 小时)",
		xaxis_title="年度小时",
		yaxis_title="负荷 (标幺值)",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		hovermode="x unified",
	)
	return fig


def plot_load_daily_stats(shape_name: str) -> go.Figure:
	"""Plot daily mean +/- std band for a single LoadShape (reshape 365x24)."""
	if shape_name not in LOADSHAPES:
		fig = go.Figure()
		fig.update_layout(template="plotly_white", height=450)
		fig.add_annotation(text="请选择一个负荷曲线", showarrow=False, font=dict(size=16))
		return fig

	raw = LOADSHAPES[shape_name]
	# Upsample back to ~8760 via linear interpolation for reshape
	full = np.interp(np.arange(8760), np.linspace(0, 8759, len(raw)), raw)
	daily = full.reshape(365, 24)
	mean = daily.mean(axis=0)
	std = daily.std(axis=0)
	hours = list(range(24))

	fig = go.Figure()
	# Std band
	fig.add_trace(go.Scatter(
		x=hours + hours[::-1],
		y=np.concatenate([mean + std, (mean - std)[::-1]]).tolist(),
		fill="toself",
		fillcolor="rgba(21, 101, 192, 0.15)",
		line=dict(color="rgba(0,0,0,0)"),
		showlegend=True,
		name="标准差范围",
	))
	# Mean line
	fig.add_trace(go.Scatter(
		x=hours, y=mean.tolist(),
		mode="lines+markers",
		name="日均值",
		line=dict(color=COLORS["primary"], width=2.5),
		marker=dict(size=5),
	))

	fig.update_layout(
		template="plotly_white",
		height=450,
		title=f"{shape_name} - 日均负荷模式",
		xaxis_title="小时 (0-23)",
		yaxis_title="负荷 (标幺值)",
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_pv_timeseries() -> go.Figure:
	"""Plot 7-day PV irradiance time series."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=PV_DATA["timestamps"],
		y=PV_DATA["irradiance_wm2"],
		mode="lines",
		name="辐照度",
		line=dict(color=COLORS["warning"], width=1.2),
		fill="tozeroy",
		fillcolor="rgba(230, 81, 0, 0.1)",
	))
	fig.update_layout(
		template="plotly_white",
		height=450,
		title="7天太阳辐照度 (2025年1月, 10分钟分辨率)",
		xaxis_title="时间戳",
		yaxis_title="辐照度 (W/m²)",
	)
	return fig


def plot_pv_scatter() -> go.Figure:
	"""Plot irradiance vs temperature scatter, colored by hour of day."""
	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=PV_DATA["temperature_c"],
		y=PV_DATA["irradiance_wm2"],
		mode="markers",
		marker=dict(
			size=4,
			color=PV_DATA["hours"],
			colorscale="Viridis",
			colorbar=dict(title="小时"),
			opacity=0.6,
		),
		text=[f"小时: {h:.1f}" for h in PV_DATA["hours"]],
		hovertemplate="温度: %{x:.1f}°C<br>辐照度: %{y:.0f} W/m²<br>%{text}<extra></extra>",
	))
	fig.update_layout(
		template="plotly_white",
		height=450,
		title="辐照度 vs 温度 (按时段着色)",
		xaxis_title="温度 (°C)",
		yaxis_title="辐照度 (W/m²)",
	)
	return fig


def plot_training_rewards() -> go.Figure:
	"""Plot episode rewards training curve."""
	if "episode_rewards" not in TRAINING:
		return _INIT_FIG

	data = TRAINING["episode_rewards"]
	steps = [pt[0] for pt in data]
	values = [pt[1] for pt in data]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=steps, y=values,
		mode="lines+markers",
		name="回合奖励",
		line=dict(color=COLORS["primary"], width=2),
		marker=dict(size=4),
	))
	fig.update_layout(
		template="plotly_white",
		height=450,
		title="HAPPO 在 13Bus 系统上的回合奖励",
		xaxis_title="训练步数",
		yaxis_title="总奖励",
	)
	return fig


# Metric name translation mapping
_METRIC_ZH = {
	"policy_loss": "策略损失",
	"dist_entropy": "分布熵",
}


def plot_training_metric(metric_name: str) -> go.Figure:
	"""Plot per-agent comparison for a selected metric."""
	fig = go.Figure()
	for agent_id in range(6):
		key = f"agent{agent_id}_{metric_name}"
		if key in TRAINING:
			data = TRAINING[key]
			steps = [pt[0] for pt in data]
			values = [pt[1] for pt in data]
			fig.add_trace(go.Scatter(
				x=steps, y=values,
				mode="lines+markers",
				name=f"智能体 {agent_id}",
				line=dict(color=COLORS["agents"][agent_id], width=1.8),
				marker=dict(size=4),
			))

	display_name = _METRIC_ZH.get(metric_name, metric_name.replace("_", " ").title())
	fig.update_layout(
		template="plotly_white",
		height=450,
		title=f"各智能体 {display_name}",
		xaxis_title="训练步数",
		yaxis_title=display_name,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	)
	return fig


def plot_power_metrics() -> go.Figure:
	"""Plot 2x2 subplot: power_loss_kw, power_loss_kvar, total_power_kw, total_power_kvar."""
	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			"有功损耗 (kW)", "无功损耗 (kVar)",
			"总有功功率 (kW)", "总无功功率 (kVar)",
		),
		vertical_spacing=0.12,
		horizontal_spacing=0.1,
	)

	metrics = [
		("power_loss_kw", 1, 1, COLORS["primary"]),
		("power_loss_kvar", 1, 2, COLORS["secondary"]),
		("total_power_kw", 2, 1, COLORS["accent"]),
		("total_power_kvar", 2, 2, COLORS["warning"]),
	]

	for key, row, col, color in metrics:
		if key in TRAINING:
			data = TRAINING[key]
			steps = [pt[0] for pt in data]
			values = [pt[1] for pt in data]
			fig.add_trace(
				go.Scatter(
					x=steps, y=values,
					mode="lines+markers",
					line=dict(color=color, width=2),
					marker=dict(size=4),
					showlegend=False,
				),
				row=row, col=col,
			)

	fig.update_layout(
		template="plotly_white",
		height=550,
		title_text="训练过程中的电力系统指标",
	)
	return fig


# ============================================================
# Environment Explorer Helpers
# ============================================================

# Environment name translation mapping
_ENV_ZH = {
	"PowerZoo VVC": "PowerZoo VVC（电压无功控制）",
	"SmartGrid": "SmartGrid（智能电网）",
	"Stackelberg": "Stackelberg（博弈环境）",
	"DSR": "DSR（需求侧响应）",
}


def get_env_names() -> list[str]:
	"""Return list of environment names."""
	return list(ENVIRONMENTS.keys())


def get_env_display_names() -> list[str]:
	"""Return list of environment display names with Chinese descriptions."""
	return [_ENV_ZH.get(name, name) for name in ENVIRONMENTS.keys()]


def _env_display_to_key(display_name: str) -> str:
	"""Convert Chinese display name back to original key."""
	for key, zh in _ENV_ZH.items():
		if zh == display_name:
			return key
	return display_name


def get_system_names(env_display: str) -> gr.Dropdown:
	"""Update system dropdown based on selected environment."""
	env_name = _env_display_to_key(env_display)
	if env_name and env_name in ENVIRONMENTS:
		systems = list(ENVIRONMENTS[env_name]["systems"].keys())
		return gr.Dropdown(choices=systems, value=systems[0] if systems else None)
	return gr.Dropdown(choices=[], value=None)


def get_env_info(env_display: str) -> str:
	"""Return environment description as markdown."""
	env_name = _env_display_to_key(env_display)
	if not env_name or env_name not in ENVIRONMENTS:
		return "请选择一个环境以查看详情。"

	env = ENVIRONMENTS[env_name]
	zh_name = _ENV_ZH.get(env_name, env_name)
	md = f"### {zh_name}\n\n"
	md += f"**描述**: {env['description']}\n\n"
	md += f"**动作空间**: {env['action_space']}\n\n"
	md += f"**观测空间**: {env['observation']}\n\n"
	md += f"**奖励函数**: {env['reward']}\n\n"
	md += "**核心特性**:\n"
	for feat in env["features"]:
		md += f"- {feat}\n"
	return md


def get_system_table(env_display: str, system_name: str) -> pd.DataFrame:
	"""Return system configuration as a DataFrame."""
	env_name = _env_display_to_key(env_display)
	if not env_name or not system_name:
		return pd.DataFrame({"属性": ["请选择环境和系统"], "值": ["-"]})

	env = ENVIRONMENTS.get(env_name, {})
	system = env.get("systems", {}).get(system_name, {})
	if not system:
		return pd.DataFrame({"属性": ["未找到该系统"], "值": ["-"]})

	rows = []
	for key, val in system.items():
		if key == "name":
			continue
		display_key = key.replace("_", " ").title()
		rows.append({"属性": display_key, "值": str(val)})
	return pd.DataFrame(rows)


# ============================================================
# Algorithm Table
# ============================================================

# Algorithm table field translation
_ALGO_TYPE_ZH = {
	"on-policy": "在策略",
	"off-policy": "离策略",
	"value-based": "基于值",
	"special": "特殊方法",
	"two-timescale": "两时间尺度",
}

_ALGO_POLICY_ZH = {
	"stochastic": "随机策略",
	"deterministic": "确定性策略",
	"squashed_gaussian": "压缩高斯策略",
	"value_decomposition": "值分解",
	"mixed": "混合策略",
}

_ALGO_ACTION_ZH = {
	"continuous": "连续",
	"discrete": "离散",
	"both": "连续/离散",
	"continuous/discrete": "连续/离散",
}


def get_algorithm_df() -> pd.DataFrame:
	"""Return algorithm comparison DataFrame."""
	return pd.DataFrame([
		{
			"算法": a["name"],
			"类型": _ALGO_TYPE_ZH.get(a["type"], a["type"]),
			"策略": _ALGO_POLICY_ZH.get(a["policy"], a["policy"]),
			"动作空间": _ALGO_ACTION_ZH.get(a["action_space"], a["action_space"]),
			"核心特性": a["key_feature"],
		}
		for a in ALGORITHMS
	])


# ============================================================
# Build Gradio App
# ============================================================

def build_app() -> gr.Blocks:
	"""Construct the Gradio Blocks application with 6 tabs."""
	with gr.Blocks(
		title="PowerZoo: 电力系统多智能体强化学习平台",
		theme=gr.themes.Soft(
			primary_hue="blue",
			secondary_hue="purple",
		),
	) as app:
		# Header
		gr.Markdown(
			"""
			# ⚡ PowerZoo: 电力系统控制通用多智能体强化学习平台
			**4 个环境** · **15 种算法** · **9 个 IEEE 标准测试系统** · **IEEE TSG 2025**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# Tab 1: Project Overview
			# --------------------------------------------------------
			with gr.Tab("项目概览"):
				gr.Markdown(
					"""
					## 关于 PowerZoo

					PowerZoo 是一个面向电力系统智能控制的综合多智能体强化学习 (MARL) 平台，
					提供统一接口用于在多种电力系统环境中训练和评估 MARL 算法。

					### 核心亮点
					"""
				)

				with gr.Row():
					with gr.Column(scale=1):
						gr.Markdown(
							"""
							**4 个环境**
							- PowerZoo VVC（电压无功控制）
							- SmartGrid（光伏集成）
							- Stackelberg 博弈（市场）
							- DSR（故障恢复）
							"""
						)
					with gr.Column(scale=1):
						gr.Markdown(
							"""
							**15 种 MARL 算法**
							- 在策略: HAPPO, HATRPO, MAPPO 等
							- 离策略: HADDPG, HASAC, MADDPG 等
							- 基于值: QMIX, HAD3QN
							- 特殊: 2TS-VVC, DAN-HAPPO
							"""
						)
					with gr.Column(scale=1):
						gr.Markdown(
							"""
							**9 个 IEEE 标准测试系统**
							- 13节点、34节点、123节点、8500节点
							- 光伏变体（保守型/优化型/激进型）
							- 从快速原型到可扩展性测试
							"""
						)

				# Architecture SVG
				svg_path = ASSETS_DIR / "architecture.svg"
				if svg_path.exists():
					svg_content = svg_path.read_text()
					gr.HTML(
						f'<div style="text-align:center; margin:20px 0; overflow-x:auto;">'
						f'{svg_content}'
						f'</div>'
					)

				gr.Markdown(
					"""
					### 引用

					> PowerZoo: A Universal Multi-Agent Reinforcement Learning Platform for Power System Control.
					> *IEEE Transactions on Smart Grid*, 2025.

					**链接**: [GitHub](https://github.com/XJTU-RL/PowerZoo) ·
					[论文](https://ieeexplore.ieee.org/)
					"""
				)

			# --------------------------------------------------------
			# Tab 2: Power System Explorer
			# --------------------------------------------------------
			with gr.Tab("电力系统浏览器"):
				gr.Markdown("## 探索环境与 IEEE 标准测试系统")

				with gr.Row():
					env_dropdown = gr.Dropdown(
						choices=get_env_display_names(),
						label="选择环境",
						value=get_env_display_names()[0],
					)
					system_dropdown = gr.Dropdown(
						choices=[],
						label="选择 IEEE 系统",
					)

				env_info = gr.Markdown(value="请选择一个环境以查看详情。")
				system_table = gr.Dataframe(
					headers=["属性", "值"],
					label="系统配置",
				)

				# Wire events
				env_dropdown.change(
					fn=get_system_names,
					inputs=env_dropdown,
					outputs=system_dropdown,
				)
				env_dropdown.change(
					fn=get_env_info,
					inputs=env_dropdown,
					outputs=env_info,
				)
				system_dropdown.change(
					fn=get_system_table,
					inputs=[env_dropdown, system_dropdown],
					outputs=system_table,
				)

				# Trigger initial load
				app.load(
					fn=get_system_names,
					inputs=env_dropdown,
					outputs=system_dropdown,
				)
				app.load(
					fn=get_env_info,
					inputs=env_dropdown,
					outputs=env_info,
				)

			# --------------------------------------------------------
			# Tab 3: Data Visualization
			# --------------------------------------------------------
			with gr.Tab("数据可视化"):
				with gr.Tabs():
					# Sub-tab: Load Profiles
					with gr.Tab("负荷曲线"):
						gr.Markdown("### 年度负荷曲线可视化")
						load_checkbox = gr.CheckboxGroup(
							choices=list(LOADSHAPES.keys()),
							value=list(LOADSHAPES.keys()),
							label="选择负荷曲线",
						)
						load_annual_plot = gr.Plot(value=plot_load_profiles(list(LOADSHAPES.keys())))

						gr.Markdown("### 日均负荷模式")
						load_radio = gr.Radio(
							choices=list(LOADSHAPES.keys()),
							value="LoadShape1",
							label="选择负荷曲线进行日分析",
						)
						load_daily_plot = gr.Plot(value=plot_load_daily_stats("LoadShape1"))

						load_checkbox.change(
							fn=plot_load_profiles,
							inputs=load_checkbox,
							outputs=load_annual_plot,
						)
						load_radio.change(
							fn=plot_load_daily_stats,
							inputs=load_radio,
							outputs=load_daily_plot,
						)

					# Sub-tab: PV Data
					with gr.Tab("光伏发电"):
						gr.Markdown("### 太阳能光伏数据 (2025年1月, 前7天)")
						pv_ts_plot = gr.Plot(value=plot_pv_timeseries())
						gr.Markdown("### 辐照度-温度相关性")
						pv_scatter_plot = gr.Plot(value=plot_pv_scatter())

			# --------------------------------------------------------
			# Tab 4: Training Dashboard
			# --------------------------------------------------------
			with gr.Tab("训练仪表盘"):
				gr.Markdown(
					"""
					## HAPPO 在 IEEE 13节点系统上的训练
					来自 PowerZoo VVC 环境上 HAPPO 实验的示例训练指标。
					"""
				)

				rewards_plot = gr.Plot(value=plot_training_rewards())

				gr.Markdown("### 各智能体指标")
				metric_dropdown = gr.Dropdown(
					choices=["policy_loss", "dist_entropy"],
					value="policy_loss",
					label="选择指标",
				)
				agent_plot = gr.Plot(value=plot_training_metric("policy_loss"))

				metric_dropdown.change(
					fn=plot_training_metric,
					inputs=metric_dropdown,
					outputs=agent_plot,
				)

				gr.Markdown("### 电力系统指标")
				power_plot = gr.Plot(value=plot_power_metrics())

			# --------------------------------------------------------
			# Tab 5: Algorithm Comparison
			# --------------------------------------------------------
			with gr.Tab("算法对比"):
				gr.Markdown(
					"""
					## 15 种 MARL 算法
					PowerZoo 支持一套完整的多智能体强化学习算法，
					涵盖在策略、离策略、基于值和特殊方法。
					"""
				)
				gr.Dataframe(
					value=get_algorithm_df(),
					label="算法特性矩阵",
					interactive=False,
				)

			# --------------------------------------------------------
			# Tab 6: Architecture Diagrams
			# --------------------------------------------------------
			with gr.Tab("架构图"):
				gr.Markdown(
					"""
					## 交互式架构图
					探索算法继承层次结构、训练流水线流程以及运行器-算法兼容性矩阵。
					"""
				)

				if "algorithm_hierarchy" in ARCH_FIGS:
					gr.Markdown("### 算法继承层次结构")
					gr.Plot(value=ARCH_FIGS["algorithm_hierarchy"])

				if "training_pipeline" in ARCH_FIGS:
					gr.Markdown("### 训练流水线流程")
					gr.Plot(value=ARCH_FIGS["training_pipeline"])

				if "runner_algorithm_matrix" in ARCH_FIGS:
					gr.Markdown("### 运行器-算法兼容性矩阵")
					gr.Plot(value=ARCH_FIGS["runner_algorithm_matrix"])

				gr.Markdown("---\n## 算法内部架构")

				_algo_details = [
					("happo_family", "HAPPO / HATRPO / HAA2C"),
					("mappo_family", "MAPPO / SN-MAPPO"),
					("dan_happo", "DAN-HAPPO"),
					("ddpg_family", "DDPG 系列 (HADDPG / HATD3 / MADDPG / MATD3)"),
					("hasac", "HASAC"),
					("value_decomposition", "QMIX / HAD3QN（值分解）"),
					("twots_vvc", "2TS-VVC（两时间尺度）"),
				]
				for key, label in _algo_details:
					if key in ARCH_FIGS:
						gr.Markdown(f"### {label}")
						gr.Plot(value=ARCH_FIGS[key])

		# Footer
		gr.Markdown(
			"""
			---
			**PowerZoo** · MIT 许可证 · [XJTU-RL](https://github.com/XJTU-RL)
			· IEEE Transactions on Smart Grid, 2025
			"""
		)

	return app


# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
	app = build_app()
	app.launch(
		server_name="0.0.0.0",
		server_port=7860,
		share=False,
		factory_reboot=True,
		allowed_paths=["assets"],
	)
