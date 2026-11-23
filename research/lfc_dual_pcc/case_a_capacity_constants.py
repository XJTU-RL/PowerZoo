"""Native Case A one-step calibration at an unconstrained initial SOC and ramp."""
import json
from pathlib import Path

import numpy as np

ACTIONS = np.asarray(json.loads(Path(__file__).with_name("case_a.json").read_text())[
    "actions_battery_pv_kw"], dtype=float)
if ACTIONS.tolist() != [[0, 0], [0, 2], [4, 0], [8, 0], [4, 2], [8, 2]]:
    raise ValueError("native delivery calibration requires the Case A six-action order")
DELIVERY_KW = np.array([.0000003368311354, 2.134934957687392, 4.371243246255972,
                        8.73820835536958, 6.50482392991762, 10.870438424506801])
