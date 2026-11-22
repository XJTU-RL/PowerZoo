"""Target-aware logit offsets from previews that already passed the physical mask."""
import math


BATTERY_LEVELS = (0, 4, 8)
PV_LEVELS = (0, 2)


def tracking_offsets(table, rows, eligible, base_import_kw, target_kw, weight, *, battery=None):
    """Score attainable joint PCC support; battery uses its best eligible PV partner."""
    costs = {}
    for i in eligible:
        batt, pv = table[i]
        if battery is not None and batt != battery:
            continue
        row = rows[i]
        if row["veto"]:
            raise ValueError("tracking prior received a vetoed candidate")
        support_kw = base_import_kw - row["pcc"][0]
        distance = abs(support_kw - target_kw)
        if not math.isfinite(distance):
            raise ValueError("nonfinite candidate tracking error")
        level = batt if battery is None else pv
        costs[level] = min(distance, costs.get(level, math.inf))
    best = min(costs.values())
    levels = BATTERY_LEVELS if battery is None else PV_LEVELS
    # The outer physical mask still removes every unsupported level.
    return [-weight*(costs[k]-best) if k in costs else 0. for k in levels]
