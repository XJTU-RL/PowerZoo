"""VVC 配置页，直接复用公共 YAML 配置。"""

from training_ui.tabs.base_tab import BaseEnvironmentTab


class VVCTab(BaseEnvironmentTab):
    env_name = "vvc"
    tab_label = "VVC"
    show_feeder_selection = True
