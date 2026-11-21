"""Measured Bus8 samples, native clock and IEEE123 storage accounting."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from envs.lfc.native_feeder_123 import PQ, StorageReading

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios/lfc/ieee123-native/receiver_limits_v1.json"


def load_protocol():
    return json.loads(SCENARIO.read_bytes())


@dataclass(frozen=True)
class Frequency:
    time_s: float
    hz: float


@dataclass(frozen=True)
class Hold:
    start_s: int
    pause_s: float
    end_pause_s: float
    disturbance_kw: float
    requested_kw: tuple[float, ...]
    terminals: tuple[StorageReading, ...]
    pcc: PQ
    bus8: PQ
    command: PQ
    balance_kw: float
    online: Frequency | None = None


@dataclass(frozen=True)
class Arm:
    holds: tuple[Hold, ...]
    hz: tuple[Frequency, ...]
    anchor: PQ
    feeder_t0: PQ


def _near(a, b, tolerance):
    return math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def read_causal(reader, previous, pause, clock):
    """Reader runs at the pause, providing actual native timestamp/value pairs."""
    if reader is None:
        raise RuntimeError("native timestamped online Bus8 frequency unavailable")
    eps = clock["timestamp_epsilon_s"]
    last = -math.inf
    selected = None
    for item in reader():
        if not (
            isinstance(item, Frequency) and math.isfinite(item.hz)
            and math.isfinite(item.time_s) and last < item.time_s
        ):
            raise RuntimeError("invalid native frequency timestamps/values")
        last = item.time_s
        if last > pause + eps:
            raise RuntimeError("online Bus8 source supplied a future timestamp")
        if previous + eps < last < pause - eps:
            selected = item
    if selected is None or pause - selected.time_s > clock["native_frequency_period_s"] + eps:
        raise RuntimeError("no fresh causal Bus8 sample")
    return selected


def demand(anchor, feeder_t0, pcc, disturbance_kw):
    return PQ(anchor.kw + pcc.kw - feeder_t0.kw + disturbance_kw,
              anchor.kvar + pcc.kvar - feeder_t0.kvar)


def _trace(arm, clock):
    t = [x.time_s for x in arm.hz]
    step = clock["native_frequency_period_s"] + clock["timestamp_epsilon_s"]
    if (
        len(t) < 2
        or not all(math.isfinite(v) for s in arm.hz for v in (s.time_s, s.hz))
        or t[0] > clock["first_hold_s"] + step
        or t[-1] < clock["stop_s"] - step
        or any(not 0 < b - a <= step for a, b in zip(t, t[1:]))
    ):
        raise RuntimeError("missing, stale or malformed native frequency trace")
    event = clock["event_start_s"]
    quiet = [s.hz for s in arm.hz if event - clock["quiet_tail_s"] <= s.time_s < event]
    return math.fsum(quiet) / len(quiet)


def validate_storage_hold(readings, requests, previous, storage_ids, config):
    """Audit observed SOC and signed measured energy before applying a hold."""
    c, g, s = (config[x] for x in ("clock", "gates", "storage"))
    eps = c["timestamp_epsilon_s"]
    if tuple(x.device for x in readings) != storage_ids:
        raise RuntimeError("invalid storage terminal IDs")
    records = []
    for i, (x, req) in enumerate(zip(readings, requests, strict=True)):
        before = s["initial_soc"] if previous is None else previous[i].soc_after
        delivered = x.measured_injection.kw
        expected_kwh = -delivered * c["exchange_s"] / 3600 * (
            1 / s["discharge_efficiency"] if delivered >= 0 else s["charge_efficiency"]
        )
        observed_kwh = (x.soc_after - x.soc_before) * s["energy_kwh"]
        if not (
            _near(delivered, req, g["storage_p_kw"])
            and _near(x.measured_injection.kvar, 0, g["storage_q_kvar"])
            and all(math.isfinite(z) and s["min_soc"] - eps <= z <= 1 + eps
                    for z in (x.soc_before, x.soc_after))
            and _near(x.soc_before, before, eps)
            and _near(observed_kwh, expected_kwh, s["energy_abs_tol_kwh"])
        ):
            raise RuntimeError(f"{x.device}: terminal delivery/SOC/energy mismatch")
        records.append({"device": x.device, "expected_delta_kwh": expected_kwh,
                        "observed_delta_kwh": observed_kwh,
                        "residual_kwh": observed_kwh - expected_kwh})
    return records
