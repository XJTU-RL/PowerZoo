"""构建 PowerZoo 的 Gradio 训练工作台。"""

from pathlib import Path

import gradio as gr

from training_ui.i18n import t
from training_ui.tabs.district_dispatch_tab import DistrictDispatchTab
from training_ui.tabs.dsr_tab import DSRTab
from training_ui.tabs.environment_tab import build_environment_tab
from training_ui.tabs.models_tab import build_models_tab
from training_ui.tabs.monitor_tab import build_monitor_tab
from training_ui.tabs.smartgrid_tab import SmartGridTab
from training_ui.tabs.stackelberg_tab import StackelbergTab
from training_ui.tabs.vvc_tab import VVCTab

CUSTOM_CSS_PATH = Path(__file__).parent / "static" / "custom.css"


def create_app() -> gr.Blocks:
    """构建环境配置、任务监控与模型浏览页。"""
    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="sm",
        font=("system-ui", "PingFang SC", "Microsoft YaHei", "sans-serif"),
        font_mono=("ui-monospace", "SFMono-Regular", "Consolas", "monospace"),
    ).set(
        body_background_fill="#f2f5f8",
        body_text_color="#1b2c3b",
        body_text_color_subdued="#526578",
        color_accent="#264c69",
        block_border_color="#d5dee7",
        block_shadow="none",
        button_primary_background_fill="#264c69",
        button_primary_background_fill_hover="#1e3d55",
        button_primary_text_color="#ffffff",
        button_primary_text_color_hover="#ffffff",
    )
    with gr.Blocks(title=t("app_title"), theme=theme, css_paths=CUSTOM_CSS_PATH) as app:
        gr.Markdown(t("app_heading"), elem_id="training-heading")

        with gr.Tabs(elem_id="training-tabs"):
            VVCTab().build()
            SmartGridTab().build()
            StackelbergTab().build()
            DSRTab().build()
            DistrictDispatchTab().build()
            build_monitor_tab()
            build_models_tab()
            build_environment_tab()

    return app
