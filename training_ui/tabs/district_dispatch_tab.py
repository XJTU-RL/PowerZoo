"""区域调度 配置页，直接复用公共 YAML 配置。"""

from training_ui.tabs.base_tab import BaseEnvironmentTab


class DistrictDispatchTab(BaseEnvironmentTab):
    env_name = "district_dispatch"
    tab_label = "区域调度"
