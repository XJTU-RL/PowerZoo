"""潮流量测的单位和方向：设备注入为正，主网进口为正。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridState:
    time_s: float
    voltages_pu: dict[str, dict[int, float]]
    injections: dict[str, tuple[float, float]]  # kW/kvar，发出为正
    grid_import: tuple[float, float]  # kW/kvar
    losses: tuple[float, float]  # kW/kvar
    storage_soc: dict[str, float]
    branches: dict[str, dict]

    @property
    def active_balance_error_kw(self):
        return (
            self.grid_import[0]
            + sum(p[0] for p in self.injections.values())
            - self.losses[0]
        )
