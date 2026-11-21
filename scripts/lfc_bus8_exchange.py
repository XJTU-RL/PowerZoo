"""Native Bus8 boundary loop shared by optimized feeder calibration drivers."""
import json
import math
from pathlib import Path
import tempfile

from envs.lfc.receiver_123 import read_causal
from envs.lfc.native_feeder_123 import PQ
from scripts.check_lfc_123_coupling import SCENARIO, _check_measurement, _copy_model, _poll, _set_pq
from scripts.lfc_bus8_receiver import (
    _advance, _instrument, _native_frequency, _physical_sensor, _verify_prefixes,
)


def exchange(clock, dispatch, emit):
    """At a measured pause dispatch(slot, start, fresh_frequency) returns kW/kvar."""
    import matlab.engine

    cfg = json.loads(SCENARIO.read_bytes())
    sensor, policy = cfg["sensor"], cfg["warmup"]
    first, stop = clock["first_hold_s"], clock["stop_s"]
    anchor = PQ(cfg["source_anchor"]["kw"], cfg["source_anchor"]["kvar"])
    eps = clock["timestamp_epsilon_s"]
    if (first != policy["seconds"] or clock["exchange_s"] != 1
            or sensor["frequency_sample_seconds"] != clock["native_frequency_period_s"]):
        raise ValueError("receiver clock differs from the pinned source")
    prefixes, commands = [], []
    with tempfile.TemporaryDirectory(prefix="lfc-bus8-") as tmp:
        model = _copy_model(cfg, Path(tmp))
        eng = matlab.engine.start_matlab()
        try:
            eng.cd(tmp, nargout=0)
            eng.load_system(model, nargout=0)
            emit({"phase": "source", **_instrument(eng, model, sensor),
                  "model_sha256": cfg["model_sha256"]})
            if abs(float(eng.eval("Ts", nargout=1)) - cfg["clock"]["simulink_base_step_s"]) > eps:
                raise RuntimeError("receiver base step changed")
            eng.set_param(model, "StopTime", str(stop + 1), nargout=0)
            eng.set_param(model + "/pause_time", "Value", str(first), nargout=0)
            _set_pq(eng, model, anchor)
            eng.set_param(model, "SimulationCommand", "start", nargout=0)
            _poll(eng, model, "paused", timeout_s=160)
            start = float(eng.get_param(model, "SimulationTime"))
            if not first - eps <= start <= first + clock["max_pause_lag_s"] + eps:
                raise RuntimeError("initial pause outside write guard")
            previous = start - 1
            for slot in range(first, stop):
                live = _native_frequency(eng)
                hz = read_causal(lambda: live, previous, start, clock)
                prefixes.append(live)
                kw, kvar = dispatch(slot, start, hz)
                target = PQ(kw, kvar)
                _set_pq(eng, model, target)
                emit({"phase": "write", "slot": slot, "pause_s": start, "native": hz,
                      "kw": kw, "kvar": kvar})
                end = _advance(eng, model, start, slot + 1, clock)
                commands.append((slot, start, end, target))
                previous, start = start, end
            eng.set_param(model, "SimulationCommand", "stop", nargout=0)
            _poll(eng, model, "stopped")
            hz = _native_frequency(eng)
            _verify_prefixes(prefixes, hz)
            if (not hz or hz[0].time_s > first or hz[-1].time_s < stop - .2 - eps
                    or any(not math.isfinite(s.hz) or not math.isfinite(s.time_s) for s in hz)
                    or any(abs(b.time_s-a.time_s-.2) > eps for a,b in zip(hz, hz[1:]))):
                raise RuntimeError("frequency trace coverage or cadence mismatch")
            logs = _physical_sensor(eng, model, sensor)
            logs["frequency_hz"] = [(s.time_s, s.hz) for s in hz]
            emit({"phase": "native_logs", "signals": logs})
            gaps = {"power": sensor["power_log_sample_seconds"] + eps, "frequency": .2 + eps}
            warm = _check_measurement(anchor, logs, policy, first, policy["tail_seconds"],
                                      dict(gaps, power=2 * sensor["power_log_sample_seconds"] + eps),
                                      settled=True)
            emit({"phase": "warm", "sensors": warm})
            rows = []
            for slot, start, end, target in commands:
                m = _check_measurement(target, logs, policy, end, .2, gaps,
                                       settled=False, hold=(start, end))
                row = {"slot": slot, "write_s": start, "end_s": end,
                       "sensor": [m["mean_p_kw"], m["mean_q_kvar"]]}
                rows.append(row)
                emit({"phase": "sensor_hold", **row})
            return hz, rows
        finally:
            eng.quit()
