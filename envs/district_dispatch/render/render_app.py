# -*- coding: utf-8 -*-
"""
PowerZoo District Dispatch Render System - Gradio 主入口

基于 Gradio 6.x Blocks 构建的 8-Tab 交互式渲染系统。
包含模型加载、实时推理、回放、分析、对比、压测、训练监控和导出功能。
"""

import logging
from typing import Any, Dict

import gradio as gr

# --- Plotly Schema Monkey-patch (Gradio 6.x + Plotly 兼容性修复) ---
_original_plot_init = gr.Plot.__init__


def _patched_plot_init(self, *args, **kwargs):
	_original_plot_init(self, *args, **kwargs)
	if hasattr(self, "schema") and isinstance(self.schema, dict):
		self.schema.pop("additionalProperties", None)


gr.Plot.__init__ = _patched_plot_init

logger = logging.getLogger(__name__)


# --- Tab 模块延迟导入 ---

def _import_tab(module_path: str, label: str):
	"""安全导入 Tab 模块，导入失败时返回 None。

	Args:
		module_path: 模块的完整 import 路径
		label: Tab 标签名 (用于错误日志)

	Returns:
		模块对象或 None
	"""
	try:
		import importlib
		return importlib.import_module(module_path)
	except Exception as exc:
		logger.warning(f"Tab '{label}' 导入失败: {exc}")
		return None


def _placeholder_tab(label: str) -> Dict[str, Any]:
	"""创建占位 Tab (模块导入失败时使用)。

	Args:
		label: Tab 标签名

	Returns:
		空组件引用字典
	"""
	gr.Markdown(f"**{label}** module failed to load. Check logs for details.")
	return {}


def create_app() -> gr.Blocks:
	"""创建 Gradio Blocks 应用。

	构建包含 8 个 Tab 的完整渲染系统，通过 gr.State 在 Tab 间共享数据。

	Returns:
		gr.Blocks 实例
	"""
	# 导入 Tab 模块
	tab_modules = {
		"Model & Data": _import_tab(
			"envs.district_dispatch.render.tabs.tab_model_data", "Model & Data"
		),
		"Live Inference": _import_tab(
			"envs.district_dispatch.render.tabs.tab_live_inference", "Live Inference"
		),
		"Playback": _import_tab(
			"envs.district_dispatch.render.tabs.tab_playback", "Playback"
		),
		"Analytics": _import_tab(
			"envs.district_dispatch.render.tabs.tab_analytics", "Analytics"
		),
		"Comparison": _import_tab(
			"envs.district_dispatch.render.tabs.tab_comparison", "Comparison"
		),
		"Stress Test": _import_tab(
			"envs.district_dispatch.render.tabs.tab_stress_test", "Stress Test"
		),
		"Training": _import_tab(
			"envs.district_dispatch.render.tabs.tab_training", "Training"
		),
		"Export": _import_tab(
			"envs.district_dispatch.render.tabs.tab_export", "Export"
		),
	}

	with gr.Blocks(
		theme=gr.themes.Soft(primary_hue="indigo"),
		title="PowerZoo District Dispatch Render",
	) as demo:
		# --- 标题 ---
		gr.Markdown("# PowerZoo District Dispatch - Render System")

		# --- 共享状态 ---
		shared_states: Dict[str, gr.State] = {
			"snapshots": gr.State([]),
			"bus_coords": gr.State({}),
			"model_loaded": gr.State(False),
			"inference_engine": gr.State(None),
			"episode_runner": gr.State(None),
		}

		# --- 8 个 Tab ---
		tab_refs: Dict[str, Dict[str, Any]] = {}

		with gr.Tab("Model & Data"):
			mod = tab_modules["Model & Data"]
			if mod is not None:
				tab_refs["model_data"] = mod.create_tab(shared_states)
			else:
				tab_refs["model_data"] = _placeholder_tab("Model & Data")

		with gr.Tab("Live Inference"):
			mod = tab_modules["Live Inference"]
			if mod is not None:
				tab_refs["live_inference"] = mod.create_tab(shared_states)
			else:
				tab_refs["live_inference"] = _placeholder_tab("Live Inference")

		with gr.Tab("Playback"):
			mod = tab_modules["Playback"]
			if mod is not None:
				tab_refs["playback"] = mod.create_tab(shared_states)
			else:
				tab_refs["playback"] = _placeholder_tab("Playback")

		with gr.Tab("Analytics"):
			mod = tab_modules["Analytics"]
			if mod is not None:
				tab_refs["analytics"] = mod.create_tab(shared_states)
			else:
				tab_refs["analytics"] = _placeholder_tab("Analytics")

		with gr.Tab("Comparison"):
			mod = tab_modules["Comparison"]
			if mod is not None:
				tab_refs["comparison"] = mod.create_tab(shared_states)
			else:
				tab_refs["comparison"] = _placeholder_tab("Comparison")

		with gr.Tab("Stress Test"):
			mod = tab_modules["Stress Test"]
			if mod is not None:
				tab_refs["stress_test"] = mod.create_tab(shared_states)
			else:
				tab_refs["stress_test"] = _placeholder_tab("Stress Test")

		with gr.Tab("Training"):
			mod = tab_modules["Training"]
			if mod is not None:
				tab_refs["training"] = mod.create_tab(shared_states)
			else:
				tab_refs["training"] = _placeholder_tab("Training")

		with gr.Tab("Export"):
			mod = tab_modules["Export"]
			if mod is not None:
				tab_refs["export"] = mod.create_tab(shared_states)
			else:
				tab_refs["export"] = _placeholder_tab("Export")

	return demo


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
	)
	app = create_app()
	app.launch(server_name="0.0.0.0", server_port=7860, share=False)
