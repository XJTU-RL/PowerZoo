# -*- coding: utf-8 -*-
"""
DSR Render Application
DSR (Demand Side Response / Service Restoration) 可视化与分析 Gradio 应用

提供 8 个标签页:
1. Model Inspector -- 模型检查点管理 (支持异构 agent)
2. Live Inference -- 实时推理 (含动作掩码、故障列表、恢复进度)
3. Playback -- Episode 回放 (短 episode, Step 0 ~ 20)
4. Analytics -- 恢复效率分析与网络状态跟踪
5. Episode Recorder -- Episode 录制管理
6. Stress Test -- 参数扫描 (n_faults, fault_severity)
7. Training -- 训练进度查看
8. Data Export -- CSV/JSON/HTML/GIF 导出

使用方式:
	python -m envs.dsr.render.render_app
	或
	from envs.dsr.render.render_app import create_app
	app = create_app()
	app.launch()
"""

import logging
from typing import Any, Dict, Optional

import gradio as gr

from envs.dsr.render.tabs import (
	tab_model_inspector,
	tab_live_inference,
	tab_playback,
	tab_analytics,
	tab_episode_recorder,
	tab_stress_test,
	tab_training_progress,
	tab_data_export,
)

logger = logging.getLogger(__name__)

_APP_TITLE = "PowerZoo DSR Render"
_APP_DESCRIPTION = (
	"Interactive visualization and analysis system for DSR (Demand Side Response / "
	"Service Restoration) MARL environment. "
	"Supports heterogeneous agents (Switch/PV/Load), action mask visualization, "
	"fault-highlighted topology, and priority load restoration tracking."
)


def create_app(
	share: bool = False,
	server_port: Optional[int] = None,
) -> gr.Blocks:
	"""创建 DSR Render Gradio 应用

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
		"live_system": "13Bus",
		"loaded_episode": None,
		"recordings": [],
		"training_runs": [],
		"playback_snapshots": None,
		"playback_system": "13Bus",
		"manual_override": None,
	}

	with gr.Blocks(
		title=_APP_TITLE,
		theme=gr.themes.Base(
			primary_hue="red",
			neutral_hue="slate",
		),
		css=_CUSTOM_CSS,
	) as app:
		gr.Markdown(f"# {_APP_TITLE}")
		gr.Markdown(_APP_DESCRIPTION)

		# 创建 8 个标签页
		tab_components: Dict[str, Dict[str, Any]] = {}

		tab_components["model_inspector"] = tab_model_inspector.create_tab(shared_states)
		tab_components["live_inference"] = tab_live_inference.create_tab(shared_states)
		tab_components["playback"] = tab_playback.create_tab(shared_states)
		tab_components["analytics"] = tab_analytics.create_tab(shared_states)
		tab_components["episode_recorder"] = tab_episode_recorder.create_tab(shared_states)
		tab_components["stress_test"] = tab_stress_test.create_tab(shared_states)
		tab_components["training"] = tab_training_progress.create_tab(shared_states)
		tab_components["data_export"] = tab_data_export.create_tab(shared_states)

		gr.Markdown(
			"---\n"
			"*PowerZoo DSR Render System | "
			"Heterogeneous Agents with Action Masks | "
			"Fault-Aware Service Restoration*"
		)

	logger.info("DSR Render app created with 8 tabs")
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
	border-bottom: 3px solid #DC2626 !important;
	color: #DC2626 !important;
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
		server_port=7866,
		share=False,
	)
