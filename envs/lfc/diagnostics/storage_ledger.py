"""Offline LFC research audit; no dependency on a solver or any runtime backend.

Pass measured AC-terminal kW (positive for charging) averaged over the exact
seconds interval. This benchmark excludes standby losses; do not apply it as
a full native-model test unless standby losses are disabled or accounted for
outside this diagnostic. The proxy trace reproduces only the historical ideal
integration + saturation + per-step rounding at the supplied interval; it does
not reconstruct actions, historical solver timing, or paper performance.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StorageContract:
    capacity_kwh: float
    reserve_kwh: float = 0.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.capacity_kwh) or self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be finite and positive")
        if (
            not math.isfinite(self.reserve_kwh)
            or not 0 <= self.reserve_kwh < self.capacity_kwh
        ):
            raise ValueError("reserve_kwh must be in [0, capacity_kwh)")
        for value in (self.charge_efficiency, self.discharge_efficiency):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("efficiency must be in (0, 1]")


@dataclass(frozen=True)
class TraceSample:
    duration_s: float
    terminal_kw: float
    observed_end_kwh: float

    def __post_init__(self):
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not all(map(math.isfinite, (self.terminal_kw, self.observed_end_kwh))):
            raise ValueError("terminal power and end energy must be finite")


@dataclass(frozen=True)
class EnergyAudit:
    ideal_delta_kwh: float
    model_delta_kwh: float
    expected_end_kwh: float
    observed_end_kwh: float
    step_residual_kwh: float
    cumulative_residual_kwh: float
    proxy_end_kwh: float
    expected_within_bounds: bool
    observed_within_bounds: bool


def audit_trace(contract, initial_kwh, samples, *, bound_atol_kwh=1e-9):
    """Return one audit record per successive interval, without clipping the reference.

    The step residual **observed minus expected** localizes an error using
    the previous *observed* energy. The cumulative residual integrates from
    the initial energy without using subsequent observations. Bounds flags
    do not silently turn infeasible metered energy transfer into success.
    """
    if not math.isfinite(bound_atol_kwh) or bound_atol_kwh < 0:
        raise ValueError("bound_atol_kwh must be finite and nonnegative")
    if (
        not math.isfinite(initial_kwh)
        or not contract.reserve_kwh <= initial_kwh <= contract.capacity_kwh
    ):
        raise ValueError("initial energy outside the allowed range")
    expected = previous_observed = proxy = initial_kwh
    rows = []

    def within_bounds(energy):
        return (
            contract.reserve_kwh - bound_atol_kwh
            <= energy
            <= contract.capacity_kwh + bound_atol_kwh
        )

    for sample in samples:
        ac = sample.terminal_kw * (sample.duration_s / 3600)
        model = (
            ac * contract.charge_efficiency
            if ac >= 0
            else ac / contract.discharge_efficiency
        )
        expected = math.fsum((expected, model))
        proxy_raw = proxy + ac
        if not all(map(math.isfinite, (ac, model, expected, proxy_raw))):
            raise ValueError("calculated storage energy is not finite")
        proxy = round(max(0.0, min(contract.capacity_kwh, proxy_raw)))
        observed = sample.observed_end_kwh
        rows.append(
            EnergyAudit(
                ideal_delta_kwh=ac,
                model_delta_kwh=model,
                expected_end_kwh=expected,
                observed_end_kwh=observed,
                step_residual_kwh=math.fsum((observed, -previous_observed, -model)),
                cumulative_residual_kwh=observed - expected,
                proxy_end_kwh=proxy,
                expected_within_bounds=within_bounds(expected),
                observed_within_bounds=within_bounds(observed),
            )
        )
        previous_observed = observed
    return tuple(rows)


def fixed_pf_q_kvar(p_kw, pf, *, q_sign=1):
    """Q in caller's P/Q convention: sign selected explicitly, magnitude from |P|.

    This is a geometry check, not a dispatch decision; PF alone does not
    determine whether reactive power is injected or absorbed.
    """
    if not all(map(math.isfinite, (p_kw, pf))) or not 0 < pf <= 1:
        raise ValueError("p_kw must be finite, and pf must be in (0, 1]")
    if q_sign not in (-1, 1):
        raise ValueError("q_sign must be -1 or 1")
    q = q_sign * abs(p_kw) * math.tan(math.acos(pf))
    if not math.isfinite(q):
        raise ValueError("computed reactive power is not finite")
    return q
