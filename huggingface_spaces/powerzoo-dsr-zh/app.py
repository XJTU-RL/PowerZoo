"""
PowerZoo DSR: 配电网服务恢复演示
HuggingFace Spaces 应用，基于 Gradio + Plotly。

5 个标签页: 概览 | 恢复进度 | 网络状态 | 动作掩码与智能体决策 | 训练仪表盘

自包含演示 -- 所有数据在线生成，除 gradio、plotly、numpy、pandas 外无需外部依赖。
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

# === Color Palette ===
COLORS = {
	"primary": "#EF4444",
	"secondary": "#F97316",
	"accent": "#06B6D4",
	"success": "#10B981",
	"fault": "#DC2626",
	"energized": "#22C55E",
	"deenergized": "#6B7280",
	"priority_critical": "#EF4444",
	"priority_important": "#F97316",
	"priority_normal": "#3B82F6",
	"bg_dark": "#1a1a2e",
	"grid_dark": "#2d2d44",
	"text_light": "#e0e0e0",
}

N_STEPS = 15  # DSR 回合长度


# ============================================================
# 演示数据生成器
# ============================================================

def _generate_restoration_data(severity: str) -> dict:
	"""根据故障严重程度生成恢复进度数据。

	Args:
		severity: 'mild'、'moderate' 或 'severe' 之一。

	Returns:
		包含 steps, critical, important, normal, total_pct 的字典。
	"""
	np.random.seed({"mild": 10, "moderate": 20, "severe": 30}[severity])
	steps = list(range(N_STEPS))

	# 各优先级负荷总数: 关键=4, 重要=5, 普通=6 -> 总计 15
	n_critical, n_important, n_normal = 4, 5, 6
	total_loads = n_critical + n_important + n_normal

	# 初始损坏程度取决于严重性
	init_frac = {"mild": 0.55, "moderate": 0.30, "severe": 0.15}[severity]
	# 恢复速度
	speed = {"mild": 0.08, "moderate": 0.06, "severe": 0.04}[severity]

	critical = []
	important = []
	normal = []
	total_pct = []

	# 关键负荷优先恢复，其次是重要负荷，最后是普通负荷
	for t in steps:
		progress = min(1.0, init_frac + speed * t + 0.01 * np.random.randn())
		progress = np.clip(progress, 0.0, 1.0)

		# 关键负荷恢复更快（优先级最高）
		c_frac = min(1.0, progress * 1.3 + 0.05 * np.random.randn())
		c_frac = np.clip(c_frac, 0.0, 1.0)
		c_count = round(c_frac * n_critical)

		# 重要负荷紧随其后
		i_frac = min(1.0, progress * 1.1 + 0.04 * np.random.randn())
		i_frac = np.clip(i_frac, 0.0, 1.0)
		i_count = round(i_frac * n_important)

		# 普通负荷最后恢复
		n_frac = min(1.0, progress * 0.9 + 0.03 * np.random.randn())
		n_frac = np.clip(n_frac, 0.0, 1.0)
		n_count = round(n_frac * n_normal)

		critical.append(c_count)
		important.append(i_count)
		normal.append(n_count)
		total_pct.append((c_count + i_count + n_count) / total_loads * 100)

	return {
		"steps": steps,
		"critical": critical,
		"important": important,
		"normal": normal,
		"total_pct": total_pct,
		"n_critical": n_critical,
		"n_important": n_important,
		"n_normal": n_normal,
	}


def _generate_13bus_topology() -> dict:
	"""生成 IEEE 13 节点系统拓扑结构（坐标和连接关系）。

	Returns:
		包含 bus_names, x, y, connections (from_idx, to_idx) 列表的字典。
	"""
	bus_names = [
		"650", "632", "633", "634", "645", "646",
		"671", "680", "684", "611", "652", "692", "675",
	]
	# 布局坐标（手动放置以提高可读性）
	x = [0.0, 2.0, 3.5, 5.0, 3.5, 5.0,
		 4.0, 6.0, 3.0, 3.0, 1.5, 5.5, 7.0]
	y = [5.0, 5.0, 6.0, 6.0, 7.5, 7.5,
		 3.5, 3.5, 2.0, 0.5, 2.0, 2.0, 2.0]

	# 线路连接（索引对）
	connections = [
		(0, 1),   # 650-632
		(1, 2),   # 632-633
		(2, 3),   # 633-634
		(1, 4),   # 632-645
		(4, 5),   # 645-646
		(1, 6),   # 632-671
		(6, 7),   # 671-680
		(6, 8),   # 671-684
		(8, 9),   # 684-611
		(8, 10),  # 684-652
		(6, 11),  # 671-692
		(11, 12), # 692-675
	]

	# 联络开关（常开，可在恢复时闭合）
	tie_switches = [
		(7, 12),  # 680-675（联络）
		(9, 10),  # 611-652（联络）
	]

	return {
		"bus_names": bus_names,
		"x": x,
		"y": y,
		"connections": connections,
		"tie_switches": tie_switches,
	}


def _generate_network_states() -> list[dict]:
	"""生成 13 节点恢复场景中每步的母线/线路状态。

	Returns:
		15 个字典的列表，每个字典包含 bus_status 和 line_status 数组。
	"""
	topo = _generate_13bus_topology()
	n_bus = len(topo["bus_names"])
	n_lines = len(topo["connections"])
	n_ties = len(topo["tie_switches"])
	states = []

	# 线路 632-671（索引 5）发生故障，671 下游区域初始断电
	faulted_line = 5
	# 故障下游母线: 671(6), 680(7), 684(8), 611(9), 652(10), 692(11), 675(12)
	downstream_buses = {6, 7, 8, 9, 10, 11, 12}

	for t in range(N_STEPS):
		bus_status = ["energized"] * n_bus
		line_status = ["closed"] * n_lines
		tie_status = ["open"] * n_ties

		if t < 2:
			# 步骤 0-1: 检测到故障，下游断电
			for b in downstream_buses:
				bus_status[b] = "deenergized"
			line_status[faulted_line] = "faulted"
		elif t < 4:
			# 步骤 2-3: 故障隔离，联络开关 680-675 闭合
			for b in downstream_buses:
				bus_status[b] = "deenergized"
			line_status[faulted_line] = "faulted"
			# 隔离：断开故障相邻线路
			tie_status[0] = "closed"  # 680-675 联络闭合
			# 675(12) 和 692(11) 通过联络获得电力
			bus_status[12] = "energized"
			bus_status[11] = "energized"
		elif t < 7:
			# 步骤 4-6: 通过光伏 + 切负荷逐步恢复
			line_status[faulted_line] = "faulted"
			tie_status[0] = "closed"
			restored = {12, 11, 7}
			for b in downstream_buses:
				if b in restored:
					bus_status[b] = "energized"
				else:
					bus_status[b] = "deenergized"
		elif t < 10:
			# 步骤 7-9: 更多母线恢复供电
			line_status[faulted_line] = "faulted"
			tie_status[0] = "closed"
			tie_status[1] = "closed"  # 611-652 联络闭合
			restored = {12, 11, 7, 8, 10}
			for b in downstream_buses:
				if b in restored:
					bus_status[b] = "energized"
				else:
					bus_status[b] = "deenergized"
		else:
			# 步骤 10-14: 接近完全恢复
			line_status[faulted_line] = "faulted"
			tie_status[0] = "closed"
			tie_status[1] = "closed"
			restored = {12, 11, 7, 8, 10, 9}
			for b in downstream_buses:
				if b in restored:
					bus_status[b] = "energized"
				else:
					bus_status[b] = "deenergized"
			# 母线 6 (671) 保持在故障区域
			bus_status[6] = "energized" if t >= 12 else "deenergized"

		states.append({
			"bus_status": bus_status,
			"line_status": line_status,
			"tie_status": tie_status,
		})

	return states


def _generate_action_mask_data() -> dict:
	"""生成 15 步的动作掩码和智能体决策数据。

	智能体: 1 个开关（动作 0-4），3 个光伏（动作 5-7），4 个负荷（动作 8-11）
	共 12 个动作索引。

	Returns:
		包含 mask (15x12), selected (15x12), agent_labels, action_labels 的字典。
	"""
	np.random.seed(42)

	agent_labels = [
		"Switch-0", "Switch-1", "Switch-2", "Switch-3", "Switch-4",
		"PV-0", "PV-1", "PV-2",
		"Load-0", "Load-1", "Load-2", "Load-3",
	]
	n_actions = len(agent_labels)

	# mask: 1=可用, 0=屏蔽
	mask = np.ones((N_STEPS, n_actions), dtype=int)
	selected = np.zeros((N_STEPS, n_actions), dtype=int)

	for t in range(N_STEPS):
		# 开关动作：隔离前（步骤 2 之前）全部屏蔽
		if t < 2:
			mask[t, 0:5] = 0  # 所有开关动作屏蔽
		elif t < 4:
			mask[t, [0, 1]] = 1  # 仅前两个开关可用
			mask[t, [2, 3, 4]] = 0
		else:
			mask[t, 0:5] = 1  # 所有开关可用

		# 光伏动作：步骤 1 之后始终可用
		if t < 1:
			mask[t, 5:8] = 0
		else:
			mask[t, 5:8] = 1

		# 负荷动作：在其母线有电之前屏蔽
		if t < 3:
			mask[t, 8:12] = 0
		elif t < 6:
			mask[t, [8, 9]] = 1
			mask[t, [10, 11]] = 0
		else:
			mask[t, 8:12] = 1

		# 从可用动作中选择
		available_idx = np.where(mask[t] == 1)[0]
		if len(available_idx) > 0:
			# 选择约 40% 的可用动作
			n_select = max(1, len(available_idx) // 3)
			chosen = np.random.choice(available_idx, size=n_select, replace=False)
			selected[t, chosen] = 1

	return {
		"mask": mask,
		"selected": selected,
		"agent_labels": agent_labels,
	}


def _generate_agent_decisions() -> dict:
	"""生成各智能体的动作时间线数据。

	Returns:
		包含智能体名称和每步动作的字典。
	"""
	np.random.seed(55)
	agents = {
		"开关智能体": {
			"actions": ["idle", "idle", "close_tie_1", "close_tie_1", "close_tie_2",
						 "reroute", "reroute", "close_tie_2", "monitor", "monitor",
						 "monitor", "monitor", "open_fault", "open_fault", "verify"],
			"color": COLORS["fault"],
		},
		"光伏智能体 0": {
			"actions": ["off", "ramp_up", "ramp_up", "100%", "100%",
						"100%", "100%", "100%", "100%", "100%",
						"80%", "80%", "60%", "60%", "60%"],
			"color": COLORS["secondary"],
		},
		"光伏智能体 1": {
			"actions": ["off", "off", "ramp_up", "ramp_up", "100%",
						"100%", "100%", "100%", "100%", "80%",
						"80%", "60%", "60%", "40%", "40%"],
			"color": "#FBBF24",
		},
		"光伏智能体 2": {
			"actions": ["off", "off", "off", "ramp_up", "ramp_up",
						"100%", "100%", "80%", "80%", "80%",
						"60%", "60%", "40%", "40%", "40%"],
			"color": "#F59E0B",
		},
		"负荷智能体 0 (关键)": {
			"actions": ["shed", "shed", "shed", "restore", "restore",
						"restore", "full", "full", "full", "full",
						"full", "full", "full", "full", "full"],
			"color": COLORS["primary"],
		},
		"负荷智能体 1 (关键)": {
			"actions": ["shed", "shed", "shed", "shed", "restore",
						"restore", "full", "full", "full", "full",
						"full", "full", "full", "full", "full"],
			"color": "#F87171",
		},
		"负荷智能体 2 (重要)": {
			"actions": ["shed", "shed", "shed", "shed", "shed",
						"shed", "restore", "restore", "full", "full",
						"full", "full", "full", "full", "full"],
			"color": COLORS["secondary"],
		},
		"负荷智能体 3 (普通)": {
			"actions": ["shed", "shed", "shed", "shed", "shed",
						"shed", "shed", "shed", "restore", "restore",
						"restore", "full", "full", "full", "full"],
			"color": COLORS["accent"],
		},
	}
	return agents


def _generate_training_data() -> dict:
	"""生成 1000 回合的 DSR 训练曲线。

	Returns:
		包含 episodes, rewards, success_rate, avg_restoration_time 的字典。
	"""
	np.random.seed(77)
	n_episodes = 1000
	episodes = np.arange(n_episodes)

	# 回合奖励：从约 -25 开始，提升至约 +15
	base_reward = -25 + 40 * (1 - np.exp(-episodes / 300))
	noise = np.random.randn(n_episodes) * 3
	rewards = base_reward + noise

	# 平滑奖励（滑动窗口 50）
	kernel = np.ones(50) / 50
	rewards_smooth = np.convolve(rewards, kernel, mode="same")

	# 恢复成功率：从约 10% 攀升至约 92%
	base_success = 0.10 + 0.82 * (1 - np.exp(-episodes / 250))
	success_noise = np.random.randn(n_episodes) * 0.05
	success_rate = np.clip(base_success + success_noise, 0.0, 1.0)
	success_smooth = np.convolve(success_rate, kernel, mode="same") * 100

	# 平均恢复时间：从约 14 步缩短至约 6 步
	base_time = 14 - 8 * (1 - np.exp(-episodes / 350))
	time_noise = np.random.randn(n_episodes) * 0.8
	avg_time = np.clip(base_time + time_noise, 2.0, 15.0)
	time_smooth = np.convolve(avg_time, kernel, mode="same")

	return {
		"episodes": episodes.tolist(),
		"rewards": rewards.tolist(),
		"rewards_smooth": rewards_smooth.tolist(),
		"success_rate": success_smooth.tolist(),
		"avg_time": avg_time.tolist(),
		"avg_time_smooth": time_smooth.tolist(),
	}


# 模块加载时预生成所有数据
TOPO = _generate_13bus_topology()
NETWORK_STATES = _generate_network_states()
ACTION_DATA = _generate_action_mask_data()
AGENT_DECISIONS = _generate_agent_decisions()
TRAINING_DATA = _generate_training_data()


# ============================================================
# 图表工厂函数
# ============================================================

def _dark_layout(**kwargs) -> dict:
	"""返回通用暗色主题布局参数。"""
	base = dict(
		template="plotly_dark",
		paper_bgcolor="#1a1a2e",
		plot_bgcolor="#16213e",
		font=dict(color="#e0e0e0"),
		margin=dict(l=60, r=40, t=60, b=50),
	)
	base.update(kwargs)
	return base


def plot_restoration_progress(severity: str) -> go.Figure:
	"""绘制恢复进度图：堆叠面积图 + 总恢复率折线。

	Args:
		severity: '轻微'、'中等' 或 '严重'。
	"""
	severity_map = {"轻微": "mild", "中等": "moderate", "严重": "severe"}
	sev = severity_map.get(severity, severity.lower())
	data = _generate_restoration_data(sev)
	steps = data["steps"]

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# 堆叠面积图：普通（底部）、重要（中间）、关键（顶部）
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["normal"],
			name=f"优先级3 - 普通 (最大 {data['n_normal']})",
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(59, 130, 246, 0.5)",
			fill="tozeroy",
			stackgroup="loads",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["important"],
			name=f"优先级2 - 重要 (最大 {data['n_important']})",
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(249, 115, 22, 0.5)",
			fill="tonexty",
			stackgroup="loads",
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["critical"],
			name=f"优先级1 - 关键 (最大 {data['n_critical']})",
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(239, 68, 68, 0.5)",
			fill="tonexty",
			stackgroup="loads",
		),
		secondary_y=False,
	)

	# 总恢复率（副 Y 轴）
	fig.add_trace(
		go.Scatter(
			x=steps, y=data["total_pct"],
			name="总恢复率 (%)",
			mode="lines+markers",
			line=dict(color=COLORS["success"], width=3),
			marker=dict(size=7, symbol="diamond"),
		),
		secondary_y=True,
	)

	# 100% 目标线
	fig.add_hline(
		y=100, line_dash="dash", line_color="rgba(255,255,255,0.4)",
		annotation_text="100% 目标",
		annotation_font_color="rgba(255,255,255,0.6)",
		secondary_y=True,
	)

	severity_label = {"mild": "轻微", "moderate": "中等", "severe": "严重"}.get(sev, severity)
	fig.update_layout(
		**_dark_layout(
			height=520,
			title=f"负荷恢复进度（{severity_label}故障）",
			legend=dict(
				orientation="h", yanchor="bottom", y=1.02,
				xanchor="center", x=0.5, font=dict(size=11),
			),
			hovermode="x unified",
		),
	)
	fig.update_xaxes(title_text="恢复步骤", dtick=1)
	fig.update_yaxes(title_text="已恢复负荷数", secondary_y=False, rangemode="tozero")
	fig.update_yaxes(title_text="恢复率 (%)", secondary_y=True, range=[0, 110])

	return fig


def plot_network_state(step: int) -> go.Figure:
	"""绘制给定步骤的 13 节点网络状态。

	Args:
		step: 恢复步骤索引 (0-14)。
	"""
	step = int(np.clip(step, 0, N_STEPS - 1))
	state = NETWORK_STATES[step]
	bus_status = state["bus_status"]
	line_status = state["line_status"]
	tie_status = state["tie_status"]

	fig = go.Figure()

	# 绘制常规线路
	for idx, (i, j) in enumerate(TOPO["connections"]):
		ls = line_status[idx]
		if ls == "faulted":
			color = COLORS["fault"]
			width = 4
			dash = "solid"
		elif ls == "closed":
			color = COLORS["energized"] if bus_status[i] == "energized" and bus_status[j] == "energized" else COLORS["deenergized"]
			width = 2.5
			dash = "solid"
		else:
			color = COLORS["deenergized"]
			width = 1.5
			dash = "dash"

		fig.add_trace(go.Scatter(
			x=[TOPO["x"][i], TOPO["x"][j]],
			y=[TOPO["y"][i], TOPO["y"][j]],
			mode="lines",
			line=dict(color=color, width=width, dash=dash),
			showlegend=False,
			hoverinfo="skip",
		))

		# 故障线路上的故障标记
		if ls == "faulted":
			mx = (TOPO["x"][i] + TOPO["x"][j]) / 2
			my = (TOPO["y"][i] + TOPO["y"][j]) / 2
			fig.add_trace(go.Scatter(
				x=[mx], y=[my],
				mode="markers+text",
				marker=dict(size=18, color=COLORS["fault"], symbol="x"),
				text=["故障"],
				textposition="top center",
				textfont=dict(color=COLORS["fault"], size=10, family="monospace"),
				showlegend=False,
				hoverinfo="text",
				hovertext=f"线路故障: {TOPO['bus_names'][i]}-{TOPO['bus_names'][j]}",
			))

	# 绘制联络开关
	for idx, (i, j) in enumerate(TOPO["tie_switches"]):
		ts = tie_status[idx]
		if ts == "closed":
			color = COLORS["accent"]
			width = 2.5
			dash = "dot"
		else:
			color = COLORS["deenergized"]
			width = 1.5
			dash = "dash"

		status_text = "闭合" if ts == "closed" else "断开"
		fig.add_trace(go.Scatter(
			x=[TOPO["x"][i], TOPO["x"][j]],
			y=[TOPO["y"][i], TOPO["y"][j]],
			mode="lines",
			line=dict(color=color, width=width, dash=dash),
			showlegend=False,
			hoverinfo="text",
			hovertext=f"联络开关: {TOPO['bus_names'][i]}-{TOPO['bus_names'][j]} ({status_text})",
		))

	# 绘制母线
	colors_map = {
		"energized": COLORS["energized"],
		"deenergized": COLORS["deenergized"],
		"faulted": COLORS["fault"],
	}
	status_labels = {
		"energized": "带电",
		"deenergized": "断电",
		"faulted": "故障",
	}
	for status_type, legend_name, symbol in [
		("energized", "带电", "circle"),
		("deenergized", "断电", "circle"),
		("faulted", "故障", "circle"),
	]:
		indices = [i for i, s in enumerate(bus_status) if s == status_type]
		if not indices:
			continue
		fig.add_trace(go.Scatter(
			x=[TOPO["x"][i] for i in indices],
			y=[TOPO["y"][i] for i in indices],
			mode="markers+text",
			marker=dict(
				size=22,
				color=colors_map[status_type],
				line=dict(width=2, color="white"),
				symbol=symbol,
			),
			text=[TOPO["bus_names"][i] for i in indices],
			textposition="bottom center",
			textfont=dict(size=10, color="#e0e0e0"),
			name=legend_name,
			hovertext=[
				f"母线 {TOPO['bus_names'][i]}: {status_labels[status_type]}"
				for i in indices
			],
			hoverinfo="text",
		))

	# 统计信息
	n_energized = sum(1 for s in bus_status if s == "energized")
	n_total = len(bus_status)
	n_ties_closed = sum(1 for s in tie_status if s == "closed")

	fig.update_layout(
		**_dark_layout(
			height=600,
			title=f"IEEE 13 节点网络状态（步骤 {step}）| "
				  f"带电: {n_energized}/{n_total} | "
				  f"联络开关闭合: {n_ties_closed}/{len(tie_status)}",
			showlegend=True,
			legend=dict(
				orientation="h", yanchor="bottom", y=-0.12,
				xanchor="center", x=0.5,
			),
		),
	)
	fig.update_xaxes(
		showgrid=False, zeroline=False, showticklabels=False,
		range=[-0.8, 8.0],
	)
	fig.update_yaxes(
		showgrid=False, zeroline=False, showticklabels=False,
		scaleanchor="x", scaleratio=1,
		range=[-0.5, 8.5],
	)

	return fig


def plot_action_mask_heatmap() -> go.Figure:
	"""绘制动作掩码热力图，展示每步的可用性和选择情况。"""
	mask = ACTION_DATA["mask"]
	selected = ACTION_DATA["selected"]
	labels = ACTION_DATA["agent_labels"]

	# 构建颜色矩阵: 0=屏蔽(灰), 1=可用(绿), 2=已选(蓝)
	color_matrix = np.zeros_like(mask, dtype=float)
	color_matrix[mask == 0] = 0.0   # 屏蔽
	color_matrix[mask == 1] = 0.5   # 可用
	color_matrix[selected == 1] = 1.0  # 已选

	# 自定义颜色刻度: 灰 -> 绿 -> 蓝
	colorscale = [
		[0.0, "#374151"],   # 屏蔽（深灰）
		[0.25, "#374151"],
		[0.25, "#059669"],  # 可用（绿色）
		[0.75, "#059669"],
		[0.75, "#2563EB"],  # 已选（蓝色）
		[1.0, "#2563EB"],
	]

	# 悬停文本
	hover_text = []
	for t in range(N_STEPS):
		row = []
		for a in range(len(labels)):
			if selected[t, a] == 1:
				status = "已选"
			elif mask[t, a] == 1:
				status = "可用"
			else:
				status = "屏蔽"
			row.append(f"步骤 {t} | {labels[a]}<br>状态: {status}")
		hover_text.append(row)

	fig = go.Figure(data=go.Heatmap(
		z=color_matrix.T,
		x=list(range(N_STEPS)),
		y=labels,
		hovertext=np.array(hover_text).T.tolist(),
		hoverinfo="text",
		colorscale=colorscale,
		showscale=False,
		xgap=2,
		ygap=2,
	))

	# 智能体分组分隔线
	for y_pos in [4.5, 7.5]:
		fig.add_hline(y=y_pos, line_color="rgba(255,255,255,0.3)", line_width=2)

	# 智能体分组标注
	fig.add_annotation(
		x=-1.5, y=2, text="开关", textangle=-90,
		showarrow=False, font=dict(color=COLORS["fault"], size=12),
		xref="x", yref="y",
	)
	fig.add_annotation(
		x=-1.5, y=6, text="光伏", textangle=-90,
		showarrow=False, font=dict(color=COLORS["secondary"], size=12),
		xref="x", yref="y",
	)
	fig.add_annotation(
		x=-1.5, y=9.5, text="负荷", textangle=-90,
		showarrow=False, font=dict(color=COLORS["accent"], size=12),
		xref="x", yref="y",
	)

	fig.update_layout(
		**_dark_layout(
			height=500,
			title="动作掩码与选择热力图",
			xaxis_title="恢复步骤",
		),
	)
	fig.update_xaxes(dtick=1)

	return fig


def plot_agent_timeline() -> go.Figure:
	"""绘制智能体决策时间线（水平动作图）。"""
	agents = AGENT_DECISIONS

	fig = go.Figure()
	agent_names = list(agents.keys())

	for idx, (name, info) in enumerate(agents.items()):
		actions = info["actions"]
		color = info["color"]

		# 每个动作作为彩色标记，位于 (step, agent_idx)
		for t, action in enumerate(actions):
			fig.add_trace(go.Scatter(
				x=[t],
				y=[idx],
				mode="markers+text",
				marker=dict(
					size=16,
					color=color,
					opacity=0.85,
					line=dict(width=1, color="white"),
				),
				text=[action],
				textposition="top center",
				textfont=dict(size=7, color="#e0e0e0"),
				showlegend=False,
				hovertext=f"步骤 {t} | {name}: {action}",
				hoverinfo="text",
			))

	fig.update_layout(
		**_dark_layout(
			height=550,
			title="智能体决策时间线",
			xaxis_title="恢复步骤",
		),
	)
	fig.update_xaxes(dtick=1, range=[-0.5, N_STEPS - 0.5])
	fig.update_yaxes(
		tickvals=list(range(len(agent_names))),
		ticktext=agent_names,
		range=[-0.8, len(agent_names) - 0.2],
	)

	return fig


def plot_training_rewards() -> go.Figure:
	"""绘制回合奖励曲线（原始 + 平滑）。"""
	episodes = TRAINING_DATA["episodes"]
	rewards = TRAINING_DATA["rewards"]
	smooth = TRAINING_DATA["rewards_smooth"]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=episodes, y=rewards,
		mode="lines",
		name="原始奖励",
		line=dict(color="rgba(239, 68, 68, 0.2)", width=1),
	))
	fig.add_trace(go.Scatter(
		x=episodes, y=smooth,
		mode="lines",
		name="平滑值 (窗口=50)",
		line=dict(color=COLORS["primary"], width=3),
	))
	fig.add_hline(
		y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)",
	)
	fig.update_layout(
		**_dark_layout(
			height=420,
			title="回合奖励曲线（HAPPO 在 DSR 123 节点系统）",
			xaxis_title="回合",
			yaxis_title="回合总奖励",
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		),
	)
	return fig


def plot_success_rate() -> go.Figure:
	"""绘制训练过程中的恢复成功率。"""
	episodes = TRAINING_DATA["episodes"]
	success = TRAINING_DATA["success_rate"]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=episodes, y=success,
		mode="lines",
		name="恢复成功率",
		line=dict(color=COLORS["success"], width=3),
		fill="tozeroy",
		fillcolor="rgba(16, 185, 129, 0.15)",
	))
	fig.add_hline(
		y=90, line_dash="dash", line_color=COLORS["secondary"],
		annotation_text="90% 阈值",
		annotation_font_color=COLORS["secondary"],
	)
	fig.update_layout(
		**_dark_layout(
			height=420,
			title="恢复成功率（>90% 负荷恢复）",
			xaxis_title="回合",
			yaxis_title="恢复成功率 (%)",
		),
	)
	fig.update_yaxes(range=[0, 105])
	return fig


def plot_restoration_time() -> go.Figure:
	"""绘制训练过程中的平均恢复时间。"""
	episodes = TRAINING_DATA["episodes"]
	avg_time = TRAINING_DATA["avg_time"]
	smooth = TRAINING_DATA["avg_time_smooth"]

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=episodes, y=avg_time,
		mode="lines",
		name="原始值",
		line=dict(color="rgba(6, 182, 212, 0.2)", width=1),
	))
	fig.add_trace(go.Scatter(
		x=episodes, y=smooth,
		mode="lines",
		name="平滑值 (窗口=50)",
		line=dict(color=COLORS["accent"], width=3),
	))
	fig.add_hline(
		y=N_STEPS, line_dash="dash", line_color="rgba(255,255,255,0.2)",
		annotation_text=f"最大值 ({N_STEPS} 步)",
		annotation_font_color="rgba(255,255,255,0.4)",
	)
	fig.update_layout(
		**_dark_layout(
			height=420,
			title="达到 90% 恢复的平均步数",
			xaxis_title="回合",
			yaxis_title="步数",
			legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
		),
	)
	fig.update_yaxes(range=[0, N_STEPS + 2])
	return fig


# ============================================================
# 构建 Gradio 应用
# ============================================================

def build_app() -> gr.Blocks:
	"""构建包含 5 个标签页的 DSR 演示 Gradio 应用。"""
	with gr.Blocks(
		title="PowerZoo DSR 配电网故障恢复",
		theme=gr.themes.Soft(primary_hue="red"),
	) as app:
		# 顶部标题
		gr.Markdown(
			"""
			# PowerZoo DSR 配电网故障恢复
			**异构多智能体强化学习故障恢复** | **动作掩码** | **基于优先级的恢复策略**
			"""
		)

		with gr.Tabs():
			# --------------------------------------------------------
			# 标签页 1: 概览
			# --------------------------------------------------------
			with gr.Tab("概览"):
				gr.Markdown(
					"""
					## 配电网服务恢复 (DSR)

					DSR 模拟**配电网络中的故障恢复过程**。当线路发生故障时，
					系统必须隔离故障区域，并通过联络开关、分布式光伏发电和
					智能切负荷策略逐步恢复断电负荷的供电。

					### 场景描述
					1. **故障检测** -- 检测到线路故障，下游母线失去供电
					2. **故障隔离** -- 通过断开相邻开关隔离故障线路
					3. **服务恢复** -- 闭合联络开关、光伏发电爬坡、按优先级恢复负荷
					   （关键 > 重要 > 普通）

					---

					### 关键参数

					| 属性 | 值 |
					|---|---|
					| **智能体类型** | 异构: 1 个开关 + 3 个光伏 + 4 个负荷 |
					| **回合长度** | 15 步（故障恢复事件） |
					| **动作空间** | 离散空间 + 动作掩码 |
					| **观测空间** | 母线电压、线路状态、负荷状态、光伏出力 |
					| **奖励函数** | 恢复率 + 电压越限惩罚 + 过载惩罚 |
					| **动作掩码** | 防止无效开关操作（拓扑约束） |

					---

					### 支持的 IEEE 测试系统

					| 系统 | 母线数 | 线路数 | 开关数 | 复杂度 |
					|---|---|---|---|---|
					| **13 节点** | 13 | 12 + 2 联络 | 5 | 原型验证 |
					| **123 节点** | 123 | 120 + 8 联络 | 20 | 标准测试 |
					| **8500 节点** | 8500 | ~8400 + 联络 | 50+ | 可扩展性 |

					---

					### 异构智能体设计

					- **开关智能体**: 控制联络开关，绕过故障区段重新路由供电。
					  其动作基于网络拓扑约束进行严格掩码。
					- **光伏智能体**: 控制分布式光伏发电出力（0-100%）。在恢复过程中
					  逐步爬坡，为断电区域提供本地电力支持。
					- **负荷智能体**: 管理切负荷优先级。关键负荷（医院等）优先恢复；
					  普通负荷最后恢复。在母线未带电前其动作被屏蔽。

					### 算法支持
					DSR 兼容 PowerZoo 中全部 15 种 MARL 算法（HAPPO、MAPPO、HATRPO 等），
					推荐使用 HAPPO，因为其顺序更新机制天然适合处理异构智能体。
					"""
				)

			# --------------------------------------------------------
			# 标签页 2: 恢复进度
			# --------------------------------------------------------
			with gr.Tab("恢复进度"):
				gr.Markdown(
					"""
					## 负荷恢复进度

					堆叠面积图展示各优先级已恢复的负荷数量。
					绿色折线追踪总恢复率。
					调整故障严重程度以观察其对恢复速度的影响。
					"""
				)
				severity_slider = gr.Radio(
					choices=["轻微", "中等", "严重"],
					value="中等",
					label="故障严重程度",
				)
				restoration_plot = gr.Plot(
					value=plot_restoration_progress("中等"),
				)
				severity_slider.change(
					fn=plot_restoration_progress,
					inputs=severity_slider,
					outputs=restoration_plot,
				)

			# --------------------------------------------------------
			# 标签页 3: 网络状态
			# --------------------------------------------------------
			with gr.Tab("网络状态"):
				gr.Markdown(
					"""
					## IEEE 13 节点网络拓扑

					可视化每个恢复步骤的网络状态。
					- **绿色节点**: 带电母线
					- **灰色节点**: 断电母线
					- **红色 X**: 故障线路
					- **灰色虚线**: 断开的开关
					- **青色点线**: 闭合的联络开关（用于恢复）
					"""
				)
				step_slider = gr.Slider(
					minimum=0, maximum=N_STEPS - 1, step=1, value=0,
					label="恢复步骤",
				)
				network_plot = gr.Plot(
					value=plot_network_state(0),
				)
				step_slider.change(
					fn=plot_network_state,
					inputs=step_slider,
					outputs=network_plot,
				)

			# --------------------------------------------------------
			# 标签页 4: 动作掩码与智能体决策
			# --------------------------------------------------------
			with gr.Tab("动作掩码与智能体决策"):
				gr.Markdown(
					"""
					## 动作可用性与智能体决策

					**热力图**展示每步的动作可用性：
					- **深灰色**: 屏蔽（因物理约束不可用）
					- **绿色**: 可用但未被选择
					- **蓝色**: 被智能体选择

					下方**时间线**展示每个智能体在每步采取的具体动作。
					"""
				)
				mask_plot = gr.Plot(value=plot_action_mask_heatmap())
				gr.Markdown("---")
				timeline_plot = gr.Plot(value=plot_agent_timeline())

			# --------------------------------------------------------
			# 标签页 5: 训练仪表盘
			# --------------------------------------------------------
			with gr.Tab("训练仪表盘"):
				gr.Markdown(
					"""
					## DSR 训练性能（HAPPO 在 123 节点系统）

					HAPPO 智能体在 DSR 环境中使用 IEEE 123 节点系统
					训练 1000 回合的训练曲线。
					"""
				)
				reward_plot = gr.Plot(value=plot_training_rewards())

				with gr.Row():
					with gr.Column():
						success_plot = gr.Plot(value=plot_success_rate())
					with gr.Column():
						time_plot = gr.Plot(value=plot_restoration_time())

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
