"""恢复任务的实际服务能量、物理越限与操作计量。"""

from envs.powergrid.operational_logger import OperationalLogger


class DSRLogger(OperationalLogger):
    extra_sum_keys = (
        "unserved_kwh",
        "weighted_service_unserved_kwh",
        "operations",
        "invalid_actions",
    )
