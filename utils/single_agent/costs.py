"""检查每次约束 RL 转移是否提供全部成本通道。"""

from collections.abc import Mapping
from math import isfinite

from envs.cbess.physics import COST_KEYS


def extract_costs(info: dict) -> dict[str, float]:
    costs = info.get("costs")
    missing = [
        key for key in COST_KEYS if not isinstance(costs, Mapping) or key not in costs
    ]
    if missing:
        raise ValueError(f"LDCPPO 缺少 info['costs'] 约束通道：{', '.join(missing)}")
    result = {}
    for key in COST_KEYS:
        try:
            value = float(costs[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"LDCPPO 约束成本 {key} 不是有效数值") from exc
        if not isfinite(value):
            raise ValueError(f"LDCPPO 约束成本 {key} 必须为有限数值")
        result[key] = value
    return result
