"""PowerZoo Training Management System - Gradio App."""

from pathlib import Path

import gradio as gr

from training_ui.i18n import t

CUSTOM_CSS_PATH = Path(__file__).parent / "static" / "custom.css"


def create_app() -> gr.Blocks:
	"""Create the main Gradio application.

	Note: In Gradio 6.x, ``theme`` and ``css`` must be passed to
	``app.launch()`` rather than the ``gr.Blocks()`` constructor.
	See ``launch.py`` for theming configuration.
	"""
	with gr.Blocks(title=t("app_title")) as app:
		gr.Markdown(t("app_heading"))

		with gr.Tabs():
			# Environment tabs
			from training_ui.tabs.vvc_tab import VVCTab
			from training_ui.tabs.smartgrid_tab import SmartGridTab
			from training_ui.tabs.stackelberg_tab import StackelbergTab
			from training_ui.tabs.dsr_tab import DSRTab
			from training_ui.tabs.district_dispatch_tab import DistrictDispatchTab
			from training_ui.tabs.monitor_tab import build_monitor_tab
			from training_ui.tabs.models_tab import build_models_tab

			VVCTab().build()
			SmartGridTab().build()
			StackelbergTab().build()
			DSRTab().build()
			DistrictDispatchTab().build()
			build_monitor_tab()
			build_models_tab()

	return app
