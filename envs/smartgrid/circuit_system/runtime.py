"""SmartGrid的模型选择适配；物理执行只在公共DSSSession中实现。"""

from envs.powergrid.session import DSSSession


class SmartGridCircuit(DSSSession):
    def __init__(self, model_folder, master, pv_plan, step_seconds, svg_elements=()):
        overlays = (
            ()
            if pv_plan == "none"
            else ((f"pv_plans/{pv_plan}.dss" if pv_plan else "pv_systems_base.dss"),)
        )
        super().__init__(model_folder, master, overlays, step_seconds, svg_elements)
