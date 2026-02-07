"""PowerZoo Training Management System - Gradio App."""

from pathlib import Path

import gradio as gr

CUSTOM_CSS_PATH = Path(__file__).parent / "static" / "custom.css"


def create_app() -> gr.Blocks:
	"""Create the main Gradio application.

	Note: In Gradio 6.x, ``theme`` and ``css`` must be passed to
	``app.launch()`` rather than the ``gr.Blocks()`` constructor.
	See ``launch.py`` for theming configuration.
	"""
	with gr.Blocks(title="PowerZoo Training Manager") as app:
		gr.Markdown(
			"# PowerZoo Training Management System\n"
			"Configure, launch, and monitor multi-agent reinforcement learning training for power system environments."
		)

		with gr.Tabs():
			# Environment tabs
			from training_ui.tabs.vvc_tab import VVCTab
			from training_ui.tabs.smartgrid_tab import SmartGridTab
			from training_ui.tabs.stackelberg_tab import StackelbergTab
			from training_ui.tabs.dsr_tab import DSRTab
			from training_ui.tabs.monitor_tab import build_monitor_tab
			from training_ui.tabs.models_tab import build_models_tab

			VVCTab().build()
			SmartGridTab().build()
			StackelbergTab().build()
			DSRTab().build()
			build_monitor_tab()
			build_models_tab()

	return app
