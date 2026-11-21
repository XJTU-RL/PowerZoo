"""Open calibration of the IEEE123/Bus8 receiver with independent native restarts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from envs.lfc.receiver_123 import SCENARIO as BASE, _trace
from scripts.check_lfc_123_coupling import SCENARIO as SOURCE
from scripts.lfc_bus8_receiver import _json_default, _native_arm

DESIGN = BASE.with_name("transfer_calibration_v1.json")
EPS = 1e-7


def deltas(arm, ref):
    result = []
    for now, base in zip(arm.hz, ref.hz, strict=True):
        if abs(now.time_s - base.time_s) > EPS or not math.isfinite(now.hz - base.hz):
            raise ValueError("unpaired native frequency samples")
        result.append((now.time_s, now.hz - base.hz))
    return result


def first_run(points, passes, count):
    streak = 0
    for index, (_, value) in enumerate(points):
        streak = streak + 1 if passes(value) else 0
        if streak == count:
            return points[index - count + 1][0]
    return None


def assess(name, arms, design):
    arm, ref = arms[name], arms["zero_a"]
    start, stop = design["event_start_s"], design["event_stop_s"]
    event_p, event_q = design["events"][name]
    if arm.anchor != ref.anchor or arm.feeder_t0 != ref.feeder_t0:
        raise ValueError("independent starts differ")
    sensor_deltas = []
    for now, base in zip(arm.holds, ref.holds, strict=True):
        if (now.start_s != base.start_s or abs(now.pause_s - base.pause_s) > EPS
                or abs(now.end_pause_s - base.end_pause_s) > EPS):
            raise ValueError("unpaired write boundaries")
        active = start <= now.start_s < stop
        row = []
        for channel, pulse in (("kw", event_p), ("kvar", event_q)):
            target = pulse if active else 0
            pcc = getattr(now.pcc, channel) - getattr(base.pcc, channel)
            cmd = getattr(now.command, channel) - getattr(base.command, channel)
            measured = getattr(now.bus8, channel) - getattr(base.bus8, channel)
            if abs(pcc) > design["max_pcc_drift_kw_kvar"] or abs(cmd - pcc - target) > EPS:
                raise ValueError(f"{name} {now.start_s} {channel} PCC or command mismatch")
            tol = (design["max_bus8_increment_error_kw_kvar"] if active and pulse
                   else design["sensor_floor_kw_kvar"])
            if abs(measured - target) > tol:
                raise ValueError(f"{name} {now.start_s} {channel} sensor delta {measured} vs {target}")
            row.append(measured)
        if active:
            sensor_deltas.append(row)
    first = next(h.pause_s for h in arm.holds if h.start_s == start)
    last = next(h.pause_s for h in arm.holds if h.start_s == stop)
    zero = deltas(arms["zero_b"], ref)
    floor = design["frequency_floor_hz"]
    noise = max(abs(v) for t, v in zero if t >= ref.holds[0].pause_s)
    if noise > design["absolute_noise_ceiling_hz"]:
        raise ValueError(f"zero-repeat noise {noise} Hz exceeds gate")
    threshold = max(floor, design["noise_multiplier"] * noise)
    series = deltas(arm, ref)
    if any(abs(v) > threshold for t, v in series if t < first):
        raise ValueError("pre-event arms differ above registered noise band")
    on = [(t, v) for t, v in series if first < t < last]
    post = [(t, v) for t, v in series if t > last]
    count = design["consecutive_samples"]
    sign = -1 if event_p > 0 else 1
    above = (lambda v: sign * v > threshold) if event_p else (lambda v: abs(v) > threshold)
    onset = first_run(on, above, count)
    # Only report recovery once a response was detected. Exact timestamps
    # bound the dwell; gaps are checked by _trace and the cadence gate.
    dwell_count = 1 + math.ceil(design["recovery_dwell_s"] / 0.2)
    recovery = first_run(post, lambda v: abs(v) <= threshold, dwell_count) if onset else None
    return dict(case=name, write_s=first, release_s=last, null_max_hz=noise,
                threshold_hz=threshold, event_min_delta_hz=min(v for _, v in on),
                event_max_delta_hz=max(v for _, v in on), first_visible_s=onset,
                latency_upper_s=None if onset is None else onset - first,
                recovered_band_entry_s=recovery, sensor_delta_kw_kvar=[
                    statistics.mean(values) for values in zip(*sensor_deltas)],
                status=("zero_repeat_passed" if name == "zero_b" else
                        "visible" if onset is not None else "unresolved"))


def run(destination):
    cfg, base, design = (json.loads(path.read_bytes()) for path in (SOURCE, BASE, DESIGN))
    clock = base["clock"].copy()
    for key in ("first_hold_s", "event_start_s", "event_stop_s", "stop_s"):
        clock[key] = design[key]
    arms = {}
    identities = []
    with destination.open("x") as report:
        def append(row):
            if row["phase"] == "feeder_identity":
                identities.append(row["versions"])
            report.write(json.dumps(row, default=_json_default, allow_nan=False) + "\n")
            report.flush()

        append({"phase": "inputs", "git": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (SOURCE, BASE, DESIGN)}, "design": design,
            "model_sha256": cfg["model_sha256"]})
        try:
            for name, (kw, kvar) in design["events"].items():
                append({"phase": "start", "arm": name})
                spec = dict(base, clock=clock, disturbance_kw=kw)
                arm, _ = _native_arm(name, cfg, spec, append, event_kvar=kvar)
                _trace(arm, clock)
                if any(abs(b.time_s - a.time_s - clock["native_frequency_period_s"]) > EPS
                       for a, b in zip(arm.hz, arm.hz[1:])):
                    raise ValueError("unexpected native frequency cadence")
                if identities[-1] != identities[0]:
                    raise ValueError("feeder source changed")
                arms[name] = arm
                if name != "zero_a":
                    summary = assess(name, arms, design)
                    append(dict(phase="paired", **summary))
                    print(json.dumps(summary), flush=True)
            append({"phase": "complete", "cases": list(arms)})
        except Exception as error:
            append({"phase": "failed", "reason": repr(error)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    os.umask(0o077)
    run(parser.parse_args().report)
