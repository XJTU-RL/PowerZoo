"""SmartGrid 配置页，直接复用公共 YAML 配置。"""

from training_ui.tabs.base_tab import BaseEnvironmentTab


class SmartGridTab(BaseEnvironmentTab):
    env_name = "smartgrid"
    tab_label = "SmartGrid"
