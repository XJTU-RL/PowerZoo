"""Pure discharge-only controller; physical voltage/current adapter not supplied."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from envs.lfc.droop_123_draft import read_causal

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios/lfc/ieee123-native/damped_droop_v1.json"


@dataclass(frozen=True)
class MeasuredHeadroom:
    device: str
    soc: float
    observed_boundary_s: float
    # Absolute permitted discharge powers, not increments. Their trusted
    # instrumentation/engineering envelopes must be verified before a run.
    voltage_cap_kw: float
    current_cap_kw: float


@dataclass(frozen=True)
class Command:
    native_sample_s: float
    native_hz: float
    unconstrained_kw: float
    filtered_kw: float
    available_kw: float
    by_device_kw: dict[str, float]


def load_protocol():
    return json.loads(SCENARIO.read_bytes())


def step(reference_hz, reader, previous_pause_s, pause_s, previous_kw,
         headroom, storage_ids, protocol):
    """Compute a command or stop; never fall back to unverified headroom."""
    c, s, clock = (protocol[x] for x in ("controller", "storage", "clock"))
    if not (
        math.isfinite(reference_hz) and math.isfinite(previous_kw) and previous_kw >= 0
        and math.isfinite(previous_pause_s) and math.isfinite(pause_s)
        and previous_pause_s < pause_s
        and abs(pause_s - previous_pause_s - c["exchange_s"])
        <= clock["max_exchange_jitter_s"] + clock["timestamp_epsilon_s"]
        and previous_kw <= c["per_device_limit_kw"] * 4
        and len(storage_ids) == len(set(storage_ids)) == 4
        and tuple(h.device for h in headroom) == tuple(storage_ids)
    ):
        raise RuntimeError("missing or invalid prior control state or resource identity")
    sample = read_causal(reader, previous_pause_s, pause_s, clock)
    caps = {}
    for measured in headroom:
        if not (
            math.isfinite(measured.soc) and s["reserve_soc"] <= measured.soc <= 1
            and math.isclose(measured.observed_boundary_s, pause_s, rel_tol=0,
                             abs_tol=clock["timestamp_epsilon_s"])
            and all(math.isfinite(p) and p >= 0
                    for p in (measured.voltage_cap_kw, measured.current_cap_kw))
        ):
            raise RuntimeError(f"{measured.device}: infeasible or unverified headroom")
        energy_cap = (
            (measured.soc - s["reserve_soc"]) * s["capacity_kwh"]
            * s["discharge_efficiency"] * 3600 / c["exchange_s"]
        )
        caps[measured.device] = min(
            c["per_device_limit_kw"], energy_cap, measured.voltage_cap_kw, measured.current_cap_kw,
        )
    cap = math.fsum(caps.values())
    request = c["gain_kw_per_hz"] * max(0, reference_hz - sample.hz - c["deadband_hz"])
    if not math.isfinite(request):
        raise RuntimeError("invalid frequency arithmetic")
    filt = (1 - c["alpha"]) * previous_kw + c["alpha"] * request
    ramp = c["aggregate_ramp_kw"]
    lower = max(0, previous_kw - ramp)
    if cap < lower:
        # Never violate the physical cap to satisfy the rate-limit metric.
        # This version stops rather than pretending an emergency action
        # was physically executed or releasing a new unsafe setpoint.
        raise RuntimeError("headroom collapse: no physically feasible ramp-limited command")
    total = min(cap, previous_kw + ramp, max(lower, filt))
    allocation = ({device: total * (power / cap) for device, power in caps.items()}
                  if cap else dict.fromkeys(caps, 0.0))
    return Command(sample.time_s, sample.hz, request, filt, cap, allocation)


def planned_events(protocol, *, design_passed=False, envelope_reviewed=False):
    """Enumerate frozen case IDs; no implicit promotion to held-out cases."""
    if not envelope_reviewed:
        raise RuntimeError("voltage/current envelope not reviewed")
    design, test = protocol["design_checks"], protocol["held_out"]
    cases = design + test
    if (
        len({x["id"] for x in cases}) != len(cases)
        or set(x["load_kw"] for x in design) & set(x["load_kw"] for x in test)
        or any(all(x[k] == protocol["excluded_prior_diagnostic"][k]
                   for k in ("load_kw", "duration_s")) for x in cases)
    ):
        raise ValueError("non-independent event identities")
    # Held-out available only after independent review of the design checks.
    return tuple(test if design_passed else design)
