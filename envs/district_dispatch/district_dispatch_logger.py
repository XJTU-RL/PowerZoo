"""区域调度按实际回合累计有单位的成本和物理指标。"""

from envs.powergrid.operational_logger import OperationalLogger


class DistrictDispatchLogger(OperationalLogger):
    extra_sum_keys = (
        "total_cost_yuan",
        "grid_bill_yuan",
        "carbon_cost_yuan",
        "battery_wear_cost_yuan",
        "curtailment_cost_yuan",
        "voltage_cost_yuan",
        "thermal_cost_yuan",
        "terminal_energy_cost_yuan",
    )
