# -*- coding: utf-8 -*-
"""
SmartGrid Render Application
SmartGrid 可视化与分析 Gradio 应用

提供 8 个标签页:
1. Model & Data -- 模型检查点和录制文件管理
2. Live Inference -- 360 步年度仿真实时推理
3. Playback -- Episode 回放 (Day 1 ~ Day 360)
4. Analytics -- CMDP 奖励分解与 Lagrangian 分析
5. Comparison -- Episode 对比
6. Stress Test -- 参数扫描与灵敏度分析
7. Training -- 训练进度查看
8. Export -- CSV/JSON/HTML/GIF 导出

使用方式:
	python -m envs.smartgrid.render.render_app
	或
	from envs.smartgrid.render.render_app import create_app
	app = create_app()
	app.launch()
"""

import logging
from typing import Any, Dict, Optional

import gradio as gr

from envs.smartgrid.render.tabs import (
	tab_model_data,
	tab_live_inference,
	tab_playback,
	tab_analytics,
	tab_comparison,
	tab_stress_test,
	tab_training,
	tab_export,
)

logger = logging.getLogger(__name__)

_APP_TITLE = "PowerZoo SmartGrid Render"
_APP_DESCRIPTION = (
	"Interactive visualization and analysis system for SmartGrid MARL environment. "
	"Supports 360-step annual simulation, CMDP Lagrangian analysis, "
	"and multi-system comparison (13Bus, 34Bus_PV, 123Bus, 8500-Node)."
)


def create_app(
	share: bool = False,
	server_port: Optional[int] = None,
) -> gr.Blocks:
	"""创建 SmartGrid Render Gradio 应用

	Args:
		share: 是否创建公共分享链接
		server_port: 服务端口号

	Returns:
		Gradio Blocks 应用实例
	"""
	shared_states: Dict[str, Any] = {
		"inference_engine": None,
		"live_runner": None,
		"live_snapshots": None,
		"live_system": "34Bus_PV",
		"loaded_episode": None,
		"recordings": [],
		"training_runs": [],
		"playback_snapshots": None,
		"playback_system": "34Bus_PV",
		"manual_override": None,
		"lambda_history": [],
		"cost_history": [],
	}

	with gr.Blocks(
		title=_APP_TITLE,
		theme=gr.themes.Base(
			primary_hue="emerald",
			neutral_hue="slate",
		),
		css=_CUSTOM_CSS,
	) as app:
		gr.Markdown(f"# {_APP_TITLE}")
		gr.Markdown(_APP_DESCRIPTION)

		# 创建 8 个标签页
		tab_components: Dict[str, Dict[str, Any]] = {}

		tab_components["model_data"] = tab_model_data.create_tab(shared_states)
		tab_components["live_inference"] = tab_live_inference.create_tab(shared_states)
		tab_components["playback"] = tab_playback.create_tab(shared_states)
		tab_components["analytics"] = tab_analytics.create_tab(shared_states)
		tab_components["comparison"] = tab_comparison.create_tab(shared_states)
		tab_components["stress_test"] = tab_stress_test.create_tab(shared_states)
		tab_components["training"] = tab_training.create_tab(shared_states)
		tab_components["export"] = tab_export.create_tab(shared_states)

		gr.Markdown(
			"---\n"
			"*PowerZoo SmartGrid Render System | "
			"CMDP Framework with Lagrangian Multiplier*"
		)

	logger.info(f"SmartGrid Render app created with 8 tabs")
	return app


# ------------------------------------------------------------------
# 自定义 CSS
# ------------------------------------------------------------------

_CUSTOM_CSS = """
.gradio-container {
	max-width: 1400px !important;
}
.tab-nav button {
	font-size: 14px !important;
	font-weight: 600 !important;
}
.tab-nav button.selected {
	border-bottom: 3px solid #10B981 !important;
	color: #10B981 !important;
}
"""


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
	)

	app = create_app()
	app.launch(
		server_name="0.0.0.0",
		server_port=7864,
		share=False,
	)
