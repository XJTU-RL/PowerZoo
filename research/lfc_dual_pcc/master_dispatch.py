"""Causal dual-PCC support allocation from per-feeder one-step bids."""
import math


SITES = ("A", "B")
FIXED_ACTION = 4  # Equal preferred split, nominal droop, mid-level residual gain.

_REPORT_NUMBERS = (
    "p_kw", "q_kvar", "delivered_kw", "available_kw", "bid_kw",
    "duration_s", "ramp_kw_s", "consumed_request_kw", "consumed_s",
)
_REPORT_FLAGS = ("ramp_limited", "energy_limited", "network_limited")


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _valid_payload(data, sampled):
    if not isinstance(data, dict) or not all(_finite_number(data.get(k)) for k in _REPORT_NUMBERS):
        return False
    if not all(type(data.get(k)) is bool for k in _REPORT_FLAGS) or "energy_only_s" not in data:
        return False
    energy = data["energy_only_s"]
    command = data.get("consumed_command")
    return (
        min(data[k] for k in ("available_kw", "bid_kw", "ramp_kw_s", "consumed_request_kw")) >= 0
        and data["duration_s"] > 0
        and data["consumed_s"] < sampled
        and (energy is None or (_finite_number(energy) and energy >= 0))
        and "consumed_command" in data
        and ((command is None and data["consumed_request_kw"] == 0) or (type(command) is int and command >= 0))
    )


def site_reports(queue, now):
    """Select independently by feeder; reject conflicting copies of a sequence."""
    selected = {}
    for site in SITES:
        by_sequence = {}
        conflicts = set()
        for packet in queue:
            if not packet.valid(now):
                continue
            data = packet.data.get(site)
            if not _valid_payload(data, packet.sampled):
                continue
            old = by_sequence.get(packet.sequence)
            if old is None:
                by_sequence[packet.sequence] = packet
            elif (packet.sampled, packet.effective, packet.expiry, data) != (
                old.sampled, old.effective, old.expiry, old.data[site]
            ):
                conflicts.add(packet.sequence)
            elif packet.arrival < old.arrival:
                by_sequence[packet.sequence] = packet
        selected[site] = max(
            (packet for seq, packet in by_sequence.items() if seq not in conflicts),
            key=lambda packet: (packet.sampled, packet.sequence),
            default=None,
        )
    return selected


def bid_limits(reports, fast_s):
    limits = {}
    for site in SITES:
        packet = reports[site]
        data = packet.data[site] if packet else None
        limits[site] = (
            min(data["bid_kw"], data["available_kw"])
            if data is not None and data["duration_s"] >= fast_s
            else 0.0
        )
    return limits


def allocate_support(error_hz, action, reports, limits, max_request_kw, droop_kw_hz):
    """Project a preference onto the communicated caps and total upward request."""
    if not math.isfinite(error_hz) or not 0 <= action < 9:
        raise ValueError("invalid master frequency error or action")
    residual = sum(
        reports[s].data[s]["consumed_request_kw"] - reports[s].data[s]["delivered_kw"]
        for s in SITES if reports[s] is not None
    )
    split, gain = divmod(action, 3)
    request = 0.0
    if error_hz > 0:
        request = min(max_request_kw, max(
            0.0, error_hz * droop_kw_hz * (.75, 1., 1.25)[gain] + (0., .15, .30)[gain] * residual,
        ))
    total = min(request, limits["A"] + limits["B"])
    a = min(total, limits["A"], max(total - limits["B"], total * (.2, .5, .8)[split]))
    return dict(A=a, B=min(limits["B"], max(0.0, total - a))), request


def feasible_actions(table, rows, request):
    """At zero request, choose the lowest support still allowed by the native guard."""
    candidates = [i for i, row in enumerate(rows) if not row["veto"]]
    if not candidates:
        raise ValueError("no feasible feeder action")
    if request > 0:
        return candidates
    minimum = min(sum(table[i]) for i in candidates)
    return [i for i in candidates if sum(table[i]) == minimum]
