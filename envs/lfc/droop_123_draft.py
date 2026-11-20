"""Engine-free pre-registration. No native online Bus8 adapter yet."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from envs.lfc.native_feeder_123 import PQ, StorageReading

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios/lfc/ieee123-native/droop_v1.json"


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


def droop(reference, sample, config):
    error = reference - sample.hz
    if not math.isfinite(error):
        raise ValueError("nonfinite frequency")
    val = math.copysign(
        config["gain_kw_per_hz"] * max(0, abs(error) - config["deadband_hz"]) / 4,
        error,
    )
    limit = config["per_storage_limit_kw"]
    return (min(limit, max(-limit, val)),) * 4


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


def _validate_arm(name, arm, frozen_ref, config, storage_ids):
    c, g, p = (config[x] for x in ("clock", "gates", "controller"))
    eps = c["timestamp_epsilon_s"]
    s = config["storage"]
    if tuple(h.start_s for h in arm.holds) != tuple(range(c["first_hold_s"], c["stop_s"])):
        raise RuntimeError(f"{name}: missing or displaced holds")
    prior = None
    for hold in arm.holds:
        start = hold.start_s
        end = start + c["exchange_s"]
        lag = c["max_pause_lag_s"]
        event = c["event_start_s"] <= start < c["event_stop_s"]
        step = config["disturbance_kw"] if event and name != "undisturbed_drift" else 0
        if not (
            start - eps <= hold.pause_s <= start + lag + eps
            and end - eps <= hold.end_pause_s <= end + lag + eps
            and (prior is None or _near(hold.pause_s, prior.end_pause_s, eps))
            and hold.disturbance_kw == step
        ):
            raise RuntimeError(f"{name} @{start}: disturbance or clock mismatch")
        command = (0.0,) * 4
        if name == "disturbed_fixed_droop" and start >= c["event_start_s"]:
            latest = read_causal(
                lambda: (s for s in arm.hz if s.time_s <= hold.pause_s + eps),
                prior.pause_s, hold.pause_s, c,
            )
            sample = hold.online
            if sample is None or not (
                _near(sample.time_s, latest.time_s, eps)
                and _near(sample.hz, latest.hz, p["hz_postcheck_tol"])
            ):
                raise RuntimeError(f"{name} @{start}: online/post-stop native frequency mismatch")
            command = droop(frozen_ref, sample, p)
        if len(hold.requested_kw) != 4 or not all(
            _near(u, v, eps) for u, v in zip(hold.requested_kw, command)
        ):
            raise RuntimeError(f"{name} @{start}: action not fixed protocol")
        if tuple(x.device for x in hold.terminals) != storage_ids:
            raise RuntimeError(f"{name} @{start}: invalid storage terminal IDs")
        for i, (x, req) in enumerate(zip(hold.terminals, command, strict=True)):
            before = s["initial_soc"] if prior is None else prior.terminals[i].soc_after
            delivered = x.measured_injection.kw
            energy_delta = -delivered * c["exchange_s"] / 3600 * (
                1 / s["discharge_efficiency"] if delivered >= 0 else s["charge_efficiency"]
            )
            if not (
                _near(x.measured_injection.kw, req, g["storage_p_kw"])
                and _near(x.measured_injection.kvar, 0, g["storage_q_kvar"])
                and all(s["min_soc"] - eps <= z <= 1 + eps for z in (x.soc_before, x.soc_after))
                and _near(x.soc_before, before, eps)
                and _near((x.soc_after - x.soc_before) * s["energy_kwh"],
                          energy_delta, s["energy_abs_tol_kwh"])
            ):
                raise RuntimeError(f"{name} @{start}: storage delivery/SOC mismatch")
        target = demand(arm.anchor, arm.feeder_t0, hold.pcc, step)
        if not (
            _near(hold.command.kw, target.kw, eps)
            and _near(hold.command.kvar, target.kvar, eps)
            and _near(hold.bus8.kw, target.kw, g["bus8_p_abs_kw"])
            and _near(hold.bus8.kvar, target.kvar, g["bus8_q_abs_kvar"])
            and _near(hold.balance_kw, 0, g["feeder_balance_kw"])
        ):
            raise RuntimeError(f"{name} @{start}: PCC or Bus8 physical gate failed")
        prior = hold


def assess(arms, storage_ids, config):
    """Never interpret output as a repeated causal efficacy study."""
    c, g = config["clock"], config["gates"]
    if len(storage_ids) != 4 or len(set(storage_ids)) != 4:
        raise ValueError("exact four resource identities required")
    needed = tuple(config["arms"])
    if set(arms) != set(needed):
        return {"status": "inconclusive", "reason": "missing paired or undisturbed arm"}
    drift, no_control, controlled = (arms[name] for name in needed)
    evidence = []
    try:
        quiet = [_trace(arms[name], c) for name in needed]
        reference = quiet[1]
        for i, name in enumerate(needed):
            arm = arms[name]
            _validate_arm(name, arm, reference, config, storage_ids)
            if (
                arm.anchor != no_control.anchor or arm.feeder_t0 != no_control.feeder_t0
                or not _near(quiet[i], reference, g["pre_f_hz"])
            ):
                raise RuntimeError("mismatched initial anchor or quiet frequency")
        drift_ref = drift.holds[0].bus8.kw
        if (
            any(not _near(x.bus8.kw, drift_ref, g["quiet_drift_kw"]) for x in drift.holds)
            or any(not _near(x.hz, quiet[0], g["quiet_drift_hz"]) for x in drift.hz
                     if x.time_s >= c["event_start_s"] - c["quiet_tail_s"])
        ):
            raise RuntimeError("independent undisturbed drift exceeds floor")
        consecutive = 0
        best = 0
        detectable_request = False
        for baseline, ctrl in zip(no_control.holds, controlled.holds, strict=True):
            t = ctrl.start_s
            eps = c["timestamp_epsilon_s"]
            if (
                not _near(ctrl.pause_s, baseline.pause_s, eps)
                or not _near(ctrl.end_pause_s, baseline.end_pause_s, eps)
            ):
                raise RuntimeError("paired actual hold times differ")
            if t < c["event_start_s"]:
                if not (
                    _near(ctrl.bus8.kw, baseline.bus8.kw, g["pre_p_kw"])
                    and _near(ctrl.bus8.kvar, baseline.bus8.kvar, g["pre_q_kvar"])
                ):
                    raise RuntimeError("pre-event paired physical state mismatch")
                continue
            if t >= c["event_stop_s"]:
                continue
            pcc = ctrl.pcc.kw - baseline.pcc.kw
            bus8 = ctrl.bus8.kw - baseline.bus8.kw
            delivered = sum(x.measured_injection.kw for x in ctrl.terminals)
            requested = sum(ctrl.requested_kw)
            detectable_request |= abs(requested) >= g["min_detectable_kw"]
            # Positive BESS discharge against low f must lower received load;
            # charging against high f must raise it.
            direction = math.copysign(1, delivered)
            supported = (
                abs(requested) >= g["min_detectable_kw"]
                and -direction * pcc >= g["min_detectable_kw"]
                and -direction * bus8 >= g["min_detectable_kw"]
                and _near(bus8, pcc, g["paired_delta_mismatch_kw"])
            )
            consecutive = consecutive + 1 if supported else 0
            best = max(best, consecutive)
            evidence.append(dict(t=t, requested_kw=requested, delivered_kw=delivered,
                                 pcc_delta_kw=pcc, bus8_delta_kw=bus8, supported=supported))
    except RuntimeError as exc:
        return {"status": "inconclusive", "reason": str(exc), "evidence": evidence}
    result = "transport_observed" if best >= g["min_consecutive"] else (
        "inconclusive" if detectable_request else "no_detectable_controller_response"
    )
    diagnostics = {}
    for name, arm in arms.items():
        event = [s.hz for s in arm.hz if c["event_start_s"] <= s.time_s < c["event_stop_s"]]
        recovery = [s.hz for s in arm.hz if s.time_s >= c["event_stop_s"]]
        diagnostics[name] = {
            "event_min_hz": min(event),
            "sampled_abs_deviation_hz_s": sum(abs(f - reference) for f in event)
            * c["native_frequency_period_s"],
            "release_last_hz": recovery[-1],
            "saturated_holds": sum(
                any(abs(u) >= config["controller"]["per_storage_limit_kw"]
                    for u in hold.requested_kw)
                for hold in arm.holds
            ),
        }
    return {"status": result, "reference_hz": reference, "evidence": evidence,
            "diagnostic_only": diagnostics}
