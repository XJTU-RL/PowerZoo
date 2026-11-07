"""真实区域端口账本：支路端口正值流入支路，区域进口为负端口功率。"""

import numpy as np


def region_meters(session, state, owners, count):
    meters = [
        {
            key: np.zeros(2)
            for key in (
                "load",
                "der",
                "source",
                "shunt",
                "boundary_import",
                "internal_loss",
            )
        }
        for _ in range(count)
    ]
    for element, injection in state.injections.items():
        info = session.element_info(element)
        region = owners[info["buses"][0].split(".")[0].lower()]
        key = "load" if element.lower().startswith("load.") else "der"
        meters[region][key] += -np.asarray(injection) if key == "load" else injection
    for element in session.native.AllElementNames:
        kind = element.split(".", 1)[0].lower()
        if kind not in ("vsource", "isource", "capacitor"):
            continue
        info = session.element_info(element)
        # 电源与电容按每个实际非接地端口计量；未支持的支路类型由构造校验拒绝。
        for bus, power in zip(info["buses"], info["powers"]):
            base = bus.split(".")[0].lower()
            if base in owners:
                key = "source" if kind in ("vsource", "isource") else "shunt"
                meters[owners[base]][key] -= power
    crossing = np.zeros(2)
    for info in state.branches.values():
        regions = [owners[b.split(".")[0].lower()] for b in info["buses"]]
        if len(set(regions)) == 1:
            meters[regions[0]]["internal_loss"] += info["losses"]
        else:
            crossing += info["losses"]
            for region, power in zip(regions, info["powers"]):
                meters[region]["boundary_import"] -= power
    result = []
    for meter in meters:
        meter["balance_error"] = (
            meter["boundary_import"]
            + meter["source"]
            + meter["der"]
            + meter["shunt"]
            - meter["load"]
            - meter["internal_loss"]
        )
        meter["net_import"] = meter["boundary_import"] + meter["source"]
        result.append({key: value.tolist() for key, value in meter.items()})
    return result, crossing.tolist()
