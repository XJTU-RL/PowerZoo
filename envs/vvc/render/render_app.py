# -*- coding: utf-8 -*-
"""
PowerZoo VVC Render Application

基于 Gradio 的 VVC 环境可视化系统主入口。
包含 8 个 Tab: Model & Data, Live Inference, Playback,
Analytics, Comparison, Stress Test, Training, Export。
"""

import importlib
import logging
from typing import Any, Callable, Dict, Optional

import gradio as gr

logger = logging.getLogger(__name__)

# --- Plotly Schema Monkey-patch ---
# Gradio 6.x 在校验 Plotly JSON 时可能因 additionalProperties 报错，
# 此补丁在 app 创建前移除该字段以规避问题。
try:
	from plotly.io._json import config as plotly_json_config

	_orig_get_schema = getattr(plotly_json_config, "get_schema", None)
	if _orig_get_schema is not None:
		def _patched_get_schema():
			schema = _orig_get_schema()
			if isinstance(schema, dict):
				schema.pop("additionalProperties", None)
			return schema
		plotly_json_config.get_schema = _patched_get_schema
		logger.debug("Plotly schema patched for Gradio 6.x compatibility")
except Exception:
	pass


def _import_tab(module_path: str) -> Optional[Callable]:
	"""延迟导入 Tab 模块，失败时返回 None。

	Args:
		module_path: Tab 模块的完整导入路径

	Returns:
		模块的 create_tab 函数，或 None
	"""
	try:
		mod = importlib.import_module(module_path)
		return getattr(mod, "create_tab", None)
	except Exception as exc:
		logger.warning(f"Failed to import {module_path}: {exc}")
		return None


def create_app() -> gr.Blocks:
	"""创建 VVC Render Gradio 应用。

	Returns:
		gr.Blocks 实例
	"""
	theme = gr.themes.Soft(primary_hue="indigo")

	with gr.Blocks(
		title="PowerZoo VVC Render",
		theme=theme,
		css="""
		.gradio-container { max-width: 1400px !important; }
		.status-bar { font-family: monospace; }
		""",
	) as app:
		gr.Markdown(
			"# PowerZoo VVC Render\n"
			"Volt-VAR Control environment visualization and analysis system."
		)

		# === 跨 Tab 共享状态 ===
		shared_states: Dict[str, gr.State] = {
			"model_loaded": gr.State(False),
			"inference_engine": gr.State(None),
			"bus_coords": gr.State({}),
		}

		# === 延迟导入 Tab 模块 ===
		tab_modules = {
			"model_data": _import_tab("envs.vvc.render.tabs.tab_model_data"),
			"live_inference": _import_tab("envs.vvc.render.tabs.tab_live_inference"),
			"playback": _import_tab("envs.vvc.render.tabs.tab_playback"),
			"analytics": _import_tab("envs.vvc.render.tabs.tab_analytics"),
			"comparison": _import_tab("envs.vvc.render.tabs.tab_comparison"),
			"stress_test": _import_tab("envs.vvc.render.tabs.tab_stress_test"),
			"training": _import_tab("envs.vvc.render.tabs.tab_training"),
			"export": _import_tab("envs.vvc.render.tabs.tab_export"),
		}

		tab_refs: Dict[str, Dict[str, Any]] = {}

		# === Tab 布局 ===
		with gr.Tabs():
			with gr.Tab("Model & Data"):
				if tab_modules["model_data"]:
					tab_refs["model_data"] = tab_modules["model_data"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Live Inference"):
				if tab_modules["live_inference"]:
					tab_refs["live_inference"] = tab_modules["live_inference"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Playback"):
				if tab_modules["playback"]:
					tab_refs["playback"] = tab_modules["playback"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Analytics"):
				if tab_modules["analytics"]:
					tab_refs["analytics"] = tab_modules["analytics"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Comparison"):
				if tab_modules["comparison"]:
					tab_refs["comparison"] = tab_modules["comparison"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Stress Test"):
				if tab_modules["stress_test"]:
					tab_refs["stress_test"] = tab_modules["stress_test"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Training"):
				if tab_modules["training"]:
					tab_refs["training"] = tab_modules["training"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

			with gr.Tab("Export"):
				if tab_modules["export"]:
					tab_refs["export"] = tab_modules["export"](shared_states)
				else:
					gr.Markdown("*Tab failed to load*")

	return app


def main():
	"""启动 VVC Render 应用。"""
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
	)
	app = create_app()
	app.launch(
		server_name="0.0.0.0",
		server_port=7870,
		share=False,
	)


if __name__ == "__main__":
	main()
