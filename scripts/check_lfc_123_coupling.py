"""Opt-in direct-engine IEEE123 → IEEE9 Bus8 acceptance probe (not training)."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from envs.lfc.native_feeder_123 import Native123Feeder, PQ, stock_probe_spec

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios/lfc/ieee123-native/coupling_v1.json"


def _validate_pause(target, actual, previous, exchange, max_lag):
    if not all(map(math.isfinite, (target, actual, previous, exchange, max_lag))):
        raise ValueError("clock inputs must be finite")
    elapsed = actual - previous
    eps = 1e-7
    if (
        elapsed <= 0
        or abs(elapsed - exchange) > max_lag + eps
        or actual < target - eps
        or actual > target + max_lag + eps
    ):
        raise ValueError(
            f"bad pause: previous={previous}; target={target}; actual={actual}"
        )


def _tail(samples, end_s, duration_s, max_gap_s):
    """Select a synchronized measured window or fail; never substitute command data."""
    if (
        duration_s <= 0
        or max_gap_s <= 0
        or not all(map(math.isfinite, (end_s, duration_s, max_gap_s)))
    ):
        raise ValueError("invalid required measurement window")
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
        if end_s - duration_s - max_gap_s <= seconds <= end_s:
            timestamps.append(seconds)
            values.append(value)
    if (
        len(values) < 2
        or timestamps[0] > end_s - duration_s + max_gap_s
        or end_s - timestamps[-1] > max_gap_s
        or max(b - a for a, b in zip(timestamps, timestamps[1:])) > max_gap_s + 1e-7
    ):
        raise RuntimeError("measured signal missing or stale in required time window")
    return values


def _check_measurement(anchor, logs, policy, end_s, tail_s, gaps_s, *, settled):
    p = _tail(logs["p_kw"], end_s, tail_s, gaps_s["power"])
    q = _tail(logs["q_kvar"], end_s, tail_s, gaps_s["power"])
    f = _tail(logs["frequency_hz"], end_s, tail_s, gaps_s["frequency"])
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


def _actual_logs(eng, model, sensor):
    # The .slx already records Simscape physical logs. Read them only after
    # stop, directly from the sensor block reaching the Bus8 dynamic load.
    # Missing sensor Q is a hard failure (not replaced by load1q readback).
    ename = lambda s: s.replace("'", "''")
    eng.eval(
        f"pz_node = simscape.logging.findNode(simlog_IEEE9BusSystem, "
        f"'{ename(model + '/' + sensor['block'])}');",
        nargout=0,
    )
    signals = {}
    for key, field, unit in (
        ("p_kw", "p_channel", "W"),
        ("q_kvar", "q_channel", "V*A"),
    ):
        eng.eval(
            f"pz_child = get(pz_node, '{ename(sensor[field])}'); pz_series = pz_child.series;",
            nargout=0,
        )
        t = _column(eng, "time(pz_series)")
        data = _column(eng, f"values(pz_series, '{unit}')")
        signals[key] = list(zip(t, (v / 1000 for v in data), strict=True))
    t = _column(eng, sensor["frequency_time_workspace_variable"])
    data = _column(eng, sensor["frequency_workspace_variable"])
    signals["frequency_hz"] = list(zip(t, data, strict=True))
    return signals


def _copy_model(cfg, temporary):
    source = ROOT / cfg["model"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != cfg["model_sha256"]:
        raise RuntimeError("unexpected IEEE9 model bytes")
    for path in (source, *source.parent.glob("*.m")):
        shutil.copyfile(path, temporary / path.name)
    return source.stem


def _engine_run(cfg, anchor, pulses):
    import matlab.engine

    policy = cfg["warmup"]
    exchange = cfg["clock"]["exchange_s"]
    step = cfg["clock"]["simulink_base_step_s"]
    max_lag = cfg["clock"]["max_pause_lag_ticks"] * step
    warmup = policy["seconds"]
    checkpoints = []
    with tempfile.TemporaryDirectory(prefix="pz-ieee9-probe-") as tmp:
        model = _copy_model(cfg, Path(tmp))
        eng = matlab.engine.start_matlab()
        try:
            eng.cd(tmp, nargout=0)
            eng.load_system(model, nargout=0)
            if not math.isclose(float(eng.eval("Ts", nargout=1)), step):
                raise RuntimeError("unexpected Simulink base step")
            stop = warmup + exchange * (len(pulses) + 1) if pulses else warmup
            eng.set_param(model, "StopTime", f"{stop}", nargout=0)
            eng.set_param(
                model + "/pause_time",
                "Value",
                f"{warmup if pulses else stop + 1}",
                nargout=0,
            )
            _set_pq(eng, model, anchor)
            eng.set_param(model, "SimulationCommand", "start", nargout=0)
            if pulses:
                _poll(eng, model, "paused")
                previous = float(eng.get_param(model, "SimulationTime"))
                if not warmup <= previous <= warmup + max_lag + 1e-7:
                    raise RuntimeError("unexpected warm-up pause timestamp")
                for index, create_target in enumerate(pulses, 1):
                    target_pq = create_target()
                    target_t = warmup + index * exchange
                    _set_pq(eng, model, target_pq)
                    eng.set_param(
                        model + "/pause_time", "Value", f"{target_t}", nargout=0
                    )
                    eng.set_param(model, "SimulationCommand", "continue", nargout=0)
                    deadline = time.monotonic() + 120
                    while True:
                        status = eng.get_param(model, "SimulationStatus")
                        actual_t = float(eng.get_param(model, "SimulationTime"))
                        if status == "paused" and actual_t > previous:
                            break
                        if status == "stopped" or time.monotonic() >= deadline:
                            raise RuntimeError("no new Simulink control boundary")
                        time.sleep(0.05)
                    _validate_pause(target_t, actual_t, previous, exchange, max_lag)
                    checkpoints.append((actual_t, target_pq))
                    previous = actual_t
                eng.set_param(model, "SimulationCommand", "stop", nargout=0)
            _poll(eng, model, "stopped")
            return _actual_logs(eng, model, cfg["sensor"]), checkpoints
        finally:
            eng.quit()


def run_probe(coupled=False):
    raw = SCENARIO.read_bytes()
    cfg = json.loads(raw)
    if cfg["schema_version"] != 1 or cfg["source_anchor"]["feeder_gain"] != 1:
        raise ValueError("unexpected coupled probe scenario")
    anchor = PQ(cfg["source_anchor"]["kw"], cfg["source_anchor"]["kvar"])
    policy, clock = cfg["warmup"], cfg["clock"]
    sensor = cfg["sensor"]
    gaps = {
        "power": sensor["power_log_sample_seconds"] * sensor["power_log_max_gap_ticks"]
        + 1e-7,
        "frequency": sensor["frequency_sample_seconds"]
        * sensor["frequency_max_gap_ticks"]
        + 1e-7,
    }
    measured, _ = _engine_run(cfg, anchor, [])
    warm = _check_measurement(
        anchor,
        measured,
        policy,
        policy["seconds"],
        policy["tail_seconds"],
        gaps,
        settled=True,
    )
    report = {
        "status": "warmup_passed",
        "scenario_sha256": hashlib.sha256(raw).hexdigest(),
        "slx_sha256": cfg["model_sha256"],
        "anchor_command": asdict(anchor),
        "warmup_realized": warm,
    }
    if coupled:
        feeder_spec, _ = stock_probe_spec()
        with Native123Feeder(anchor) as driver:
            rows = []

            def command_at_pause(pulse):
                request = dict(
                    zip(driver.storage_ids, pulse["storage_kw"], strict=True)
                )
                row = driver.step(request)
                exchange = clock["exchange_s"]
                if not (
                    math.isclose(row.start_s, pulse["start_s"], abs_tol=1e-6)
                    and math.isclose(row.end_s - row.start_s, exchange, abs_tol=1e-6)
                    and abs(row.active_balance_error_kw) <= 0.05
                ):
                    raise RuntimeError(
                        "123 native interval clock or source balance failed"
                    )
                rows.append(row)
                return row.equivalent_transmission_load

            report["feeder_background"] = asdict(driver.calibration.background)
            report["native_scenario_sha256"] = driver.scenario_sha256
            report["native_deck_sha256"] = driver.session.model_sha256
            inputs = [
                partial(command_at_pause, row) for row in feeder_spec["input_pulses"]
            ]
            measured, checkpoints = _engine_run(cfg, anchor, inputs)
        _check_measurement(
            anchor,
            measured,
            policy,
            policy["seconds"],
            policy["tail_seconds"],
            gaps,
            settled=True,
        )
        report["steps"] = []
        for row, (actual_t, command) in zip(rows, checkpoints, strict=True):
            realized = _check_measurement(
                command, measured, policy, actual_t, 0.2, gaps, settled=False
            )
            report["steps"].append(
                {
                    "simulation_time_s": actual_t,
                    "actual": realized,
                    "feeder": asdict(row),
                }
            )
        report["status"] = "coupled_physical_probe_passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coupled",
        action="store_true",
        help="run four 1s pulse exchanges after warm-up passes",
    )
    print(json.dumps(run_probe(parser.parse_args().coupled), indent=2, allow_nan=False))
