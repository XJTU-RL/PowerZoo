"""Stackelberg 配置页，直接复用公共 YAML 配置。"""

from training_ui.tabs.base_tab import BaseEnvironmentTab


class StackelbergTab(BaseEnvironmentTab):
    env_name = "stackelberg_13bus"
    tab_label = "Stackelberg"
