"""约束成本通道提取：从 gym info 字典到 LDCPPO 的 {VD,VN,LO} 流。"""

from __future__ import annotations

COST_KEYS = ("VD", "VN", "LO")


def extract_costs(info: dict) -> dict | None:
    """优先 info["costs"]（cbess 家族）；兼容 info["cost_voltage"] 单流。

    返回 None 表示该环境无成本通道（调用方退化为纯 PPO 模式）。
    """
    costs = info.get("costs")
    if isinstance(costs, dict):
        return {k: float(costs.get(k, 0.0)) for k in COST_KEYS}
    if "cost_voltage" in info:
        value = float(info["cost_voltage"])
        return {"VD": value, "VN": 0.0, "LO": 0.0}
    return None
