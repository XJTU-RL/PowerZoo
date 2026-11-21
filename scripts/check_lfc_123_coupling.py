"""Bus8 model staging, write readback and physical sensor windows."""
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/lfc/ieee123-native/coupling_v1.json"
CLOCK_EPS_S = 1e-7


def _validate_pause(target, actual, previous, exchange, max_lag):
    if not all(map(math.isfinite, (target, actual, previous, exchange, max_lag))):
        raise ValueError("clock inputs must be finite")
    elapsed = actual - previous
    eps = CLOCK_EPS_S
    if (
        elapsed <= 0
        or abs(elapsed - exchange) > max_lag + eps
        or actual < target - eps
        or actual > target + max_lag + eps
    ):
        raise ValueError(
            f"bad pause: previous={previous}; target={target}; actual={actual}"
        )


def _tail(samples, end_s, duration_s, max_gap_s, *, hold=None, audit=None):
    """Select a synchronized measured window or fail; never substitute command data."""
    if (
        duration_s <= 0
        or max_gap_s <= 0
        or not all(map(math.isfinite, (end_s, duration_s, max_gap_s)))
    ):
        raise ValueError("invalid required measurement window")
    if hold is not None and not (
        math.isfinite(hold[0]) and hold[1] == end_s
        and hold[0] < end_s - duration_s - CLOCK_EPS_S
    ):
        raise ValueError("measurement window must be inside the actual hold")
    values = []
    timestamps = []
    previous = -math.inf
    for seconds, value in samples:
        if (
            not math.isfinite(seconds)
            or not math.isfinite(value)
            or seconds <= previous
        ):
            raise RuntimeError("unusable measured signal (nonfinite or repeated time)")
        previous = seconds
        if hold is not None:
            if any(abs(seconds - boundary) <= CLOCK_EPS_S for boundary in hold):
                if audit is not None:
                    audit["excluded_boundary"].append([seconds, value])
            if not hold[0] + CLOCK_EPS_S < seconds < end_s - CLOCK_EPS_S:
                continue
        if end_s - duration_s - max_gap_s <= seconds <= end_s:
            timestamps.append(seconds)
            values.append(value)
    if audit is not None:
        audit["window_first_s"] = timestamps[0] if timestamps else None
        audit["window_last_s"] = timestamps[-1] if timestamps else None
        audit["window_samples"] = len(values)
    if (
        len(values) < 2
        or timestamps[0] > end_s - duration_s + max_gap_s
        or end_s - timestamps[-1] > max_gap_s
        or max(b - a for a, b in zip(timestamps, timestamps[1:])) > max_gap_s + 1e-7
    ):
        raise RuntimeError("measured signal missing or stale in required time window")
    if audit is not None:
        audit.update(last=values[-1], mean=math.fsum(values) / len(values),
                     minimum=min(values), maximum=max(values))
    return values


def _check_measurement(
    anchor, logs, policy, end_s, tail_s, gaps_s, *, settled, hold=None, audit=None
):
    selected = []
    signal_errors = []
    for name, clock in (("p_kw", "power"), ("q_kvar", "power"),
                        ("frequency_hz", "frequency")):
        signal_audit = {"excluded_boundary": []}
        if audit is not None:
            audit[name] = signal_audit
        try:
            selected.append(
                _tail(logs[name], end_s, tail_s, gaps_s[clock],
                      hold=hold, audit=signal_audit)
            )
        except (RuntimeError, KeyError) as exc:
            signal_errors.append(f"{name}: {exc}")
    if signal_errors:
        raise RuntimeError("; ".join(signal_errors))
    p, q, f = selected
    p_error = max(abs(x - anchor.kw) for x in p)
    q_error = max(abs(x - anchor.kvar) for x in q)
    stats = {
        "command_kw": anchor.kw,
        "command_kvar": anchor.kvar,
        "measured_p_kw": p[-1],
        "measured_q_kvar": q[-1],
        "frequency_hz": f[-1],
        "mean_p_kw": math.fsum(p) / len(p),
        "mean_q_kvar": math.fsum(q) / len(q),
        "mean_frequency_hz": math.fsum(f) / len(f),
        "frequency_min_hz": min(f),
        "frequency_max_hz": max(f),
        "frequency_span_hz": max(f) - min(f),
        "p_max_error_kw": p_error,
        "q_max_error_kvar": q_error,
        "samples": {"p": len(p), "q": len(q), "frequency": len(f)},
    }
    if audit is not None:
        audit["physical"] = stats
    if p_error > policy["p_abs_tol_kw"] or q_error > policy["q_abs_tol_kvar"]:
        raise RuntimeError(
            "realized sensor P/Q does not track the commanded boundary: "
            + json.dumps(stats, allow_nan=False)
        )
    if settled and (
        max(abs(x - policy["frequency_center_hz"]) for x in f)
        > policy["frequency_abs_tol_hz"]
        or stats["frequency_span_hz"] > policy["frequency_max_span_hz"]
    ):
        raise RuntimeError(
            "baseline frequency did not settle inside the declared band: "
            + json.dumps(stats, allow_nan=False)
        )
    return stats


def _set_pq(eng, model, demand):
    # Only the separately wired dynamic load is driven here. Fixed Bus 8
    # Load C and all other transmission load blocks remain untouched.
    for suffix, val in (("p", demand.kw), ("q", demand.kvar)):
        block = f"{model}/load1{suffix}"
        watts_or_var = val * 1000
        eng.set_param(block, "Value", f"{watts_or_var:.17g}", nargout=0)
        applied = float(eng.get_param(block, "Value"))
        if not math.isclose(applied, watts_or_var, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"Simulink P/Q command readback failed: {block}")


def _poll(eng, model, wanted, timeout_s=120):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = eng.get_param(model, "SimulationStatus")
        if status == wanted:
            return
        if status == "stopped" and wanted != "stopped":
            raise RuntimeError("Simulink stopped before the expected pause")
        time.sleep(0.05)
    raise TimeoutError(f"Simulink did not reach {wanted}")


def _column(eng, expression):
    import numpy as np

    return np.asarray(eng.eval(expression, nargout=1), dtype=float).reshape(-1).tolist()


def _copy_model(cfg, temporary):
    source = ROOT / cfg["model"]
    inputs = [(source, cfg["model_sha256"])]
    for name, digest in cfg["model_helpers"].items():
        if Path(name).name != name or Path(name).suffix != ".m":
            raise ValueError(f"invalid staged MATLAB helper: {name}")
        inputs.append((source.parent / name, digest))
    verified = []
    for path, digest in inputs:
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError(f"missing pinned MATLAB asset: {path.name}") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"unexpected pinned MATLAB asset bytes: {path.name}")
        verified.append((path.name, data))
    for name, data in verified:
        (temporary / name).write_bytes(data)
    return source.stem
