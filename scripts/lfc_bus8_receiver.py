"""Native timestamped frequency receiver and open-loop calibration arm."""
from dataclasses import asdict, is_dataclass
import math
import tempfile
import time
from pathlib import Path

from envs.lfc.receiver_123 import (
    Arm, Frequency, Hold, demand, read_causal, validate_storage_hold,
)
from envs.lfc.native_feeder_123 import Native123Feeder, PQ
from scripts.check_lfc_123_coupling import (
    _check_measurement, _column, _copy_model, _poll, _set_pq, _validate_pause,
)

SID = "3596"
BLOCK = "To Workspace1"
SIGNAL = "out_f_load1"


def _instrument(eng, model, sensor):
    target = f"{model}/{BLOCK}"
    eng.eval(
        f"pz_b=find_system('{model}','BlockType','ToWorkspace','VariableName','{SIGNAL}');"
        f"assert(numel(pz_b)==1 && strcmp(pz_b{{1}},'{target}'));"
        f"assert(strcmp(Simulink.ID.getSID('{target}'),'{model}:{SID}'));",
        nargout=0,
    )
    mode = eng.get_param(model, "ReturnWorkspaceOutputs")
    if mode != "off" or sensor["frequency_workspace_variable"] != SIGNAL:
        raise RuntimeError("unsupported deployed Bus8 source/output provenance")
    if (eng.get_param(target, "SaveFormat") != "Array" or
            float(eng.get_param(target, "SampleTime")) != sensor["frequency_sample_seconds"]):
        raise RuntimeError("unexpected native frequency logger format or cadence")
    eng.set_param(target, "SaveFormat", "Timeseries", nargout=0)
    if eng.get_param(target, "SaveFormat") != "Timeseries":
        raise RuntimeError("Bus8 logger instrumentation readback failed")
    return {"block": target, "sid": SID, "mode": mode, "variable": SIGNAL,
            "sample_time_s": sensor["frequency_sample_seconds"]}


def _native_frequency(eng):
    # Only base workspace, only native timestamps from the same object.
    eng.eval(f"assert(exist('{SIGNAL}','var')~=0 && isa({SIGNAL},'timeseries'))", nargout=0)
    return tuple(Frequency(t, f) for t, f in zip(
        _column(eng, f"{SIGNAL}.Time"), _column(eng, f"{SIGNAL}.Data"), strict=True,
    ))


def _verify_prefixes(live, stopped):
    for snapshot in live:
        if not snapshot or snapshot != stopped[:len(snapshot)]:
            raise RuntimeError("live Bus8 native samples do not match stopped trace")


def _physical_sensor(eng, model, sensor):
    block = (model + "/" + sensor["block"]).replace("'", "''")
    eng.eval(
        f"pz_node = simscape.logging.findNode(simlog_IEEE9BusSystem,'{block}');",
        nargout=0,
    )
    result = {}
    for key, field, unit in (("p_kw", "p_channel", "W"), ("q_kvar", "q_channel", "V*A")):
        channel = sensor[field].replace("'", "''")
        eng.eval(f"pz_child=get(pz_node,'{channel}'); pz_series=pz_child.series;", nargout=0)
        result[key] = list(zip(
            _column(eng, "time(pz_series)"),
            (v / 1000 for v in _column(eng, f"values(pz_series,'{unit}')")),
            strict=True,
        ))
    return result


def _advance(eng, model, start, target, clock):
    eng.set_param(f"{model}/pause_time", "Value", str(target), nargout=0)
    eng.set_param(model, "SimulationCommand", "continue", nargout=0)
    until = time.monotonic() + 120
    while time.monotonic() < until:
        status = eng.get_param(model, "SimulationStatus")
        now = float(eng.get_param(model, "SimulationTime"))
        if status == "paused" and now > start:
            _validate_pause(target, now, start, clock["exchange_s"], clock["max_pause_lag_s"])
            return now
        if status == "stopped":
            break
        time.sleep(0.05)
    raise RuntimeError("no fresh Simulink communication pause")


def _native_arm(name, cfg, spec, append, *, event_kvar=0.0):
    import matlab.engine

    clock, sensor, policy = spec["clock"], cfg["sensor"], cfg["warmup"]
    eps = clock["timestamp_epsilon_s"]
    first, last = clock["first_hold_s"], clock["stop_s"]
    anchor = PQ(cfg["source_anchor"]["kw"], cfg["source_anchor"]["kvar"])
    prefix, staged = [], []
    with tempfile.TemporaryDirectory(prefix="pz-dss-transfer-") as temp:
        model = _copy_model(cfg, Path(temp))
        with Native123Feeder(anchor) as feeder:
            feeder_t0 = feeder.calibration.feeder_t0
            append({"arm": name, "phase": "feeder_identity", "versions": {
                "native_scenario_sha256": feeder.scenario_sha256,
                "native_deck_sha256": feeder.session.model_sha256,
            }, "anchor": asdict(anchor), "feeder_t0": asdict(feeder_t0)})
            eng = matlab.engine.start_matlab()
            try:
                eng.cd(temp, nargout=0)
                eng.load_system(model, nargout=0)
                append({"arm": name, "phase": "source", **_instrument(eng, model, sensor)})
                if abs(float(eng.eval("Ts", nargout=1)) - cfg["clock"]["simulink_base_step_s"]) > eps:
                    raise RuntimeError("unexpected Simulink base step")
                eng.set_param(model, "StopTime", str(last + 1), nargout=0)
                eng.set_param(f"{model}/pause_time", "Value", str(first), nargout=0)
                _set_pq(eng, model, anchor)
                eng.set_param(model, "SimulationCommand", "start", nargout=0)
                _poll(eng, model, "paused", timeout_s=160)
                actual = float(eng.get_param(model, "SimulationTime"))
                if not first - eps <= actual <= first + clock["max_pause_lag_s"] + eps:
                    raise RuntimeError("first pause outside approved clock")
                prev = actual - clock["exchange_s"]
                for slot in range(first, last):
                    command = (0.0,) * 4
                    live = _native_frequency(eng)
                    observed = read_causal(lambda: live, prev, actual, clock)
                    prefix.append(live)
                    disturbance = (
                        spec["disturbance_kw"] if clock["event_start_s"] <= slot < clock["event_stop_s"] else 0
                    )
                    storage = dict(zip(feeder.storage_ids, command, strict=True))
                    if staged:
                        s = spec["storage"]
                        for reading, kw in zip(staged[-1]["terminals"], command, strict=True):
                            hours = clock["exchange_s"] / 3600
                            new_kwh = reading.soc_after * s["energy_kwh"] - kw * hours * (
                                1 / s["discharge_efficiency"] if kw >= 0 else s["charge_efficiency"]
                            )
                            if not s["min_soc"] * s["energy_kwh"] <= new_kwh <= s["energy_kwh"]:
                                raise RuntimeError("storage command exceeds available energy headroom")
                    row = feeder.step(storage)
                    try:
                        ledger = validate_storage_hold(
                            row.storage, command, staged[-1]["terminals"] if staged else None,
                            tuple(feeder.storage_ids), spec,
                        )
                        if not (math.isfinite(row.active_balance_error_kw) and
                                abs(row.active_balance_error_kw) <= spec["gates"]["feeder_balance_kw"]):
                            raise RuntimeError("native PCC balance pre-application gate failed")
                    except RuntimeError as err:
                        # Encode malformed nonfinite telemetry as explicit text,
                        # never as a finite replacement or an invalid JSON number.
                        append({"arm": name, "phase": "pre_application_rejected",
                                "t": slot, "request_kw": command, "reason": str(err),
                                "observed_native_row": repr(row)})
                        raise
                    if (abs(row.start_s - (slot - first)) > eps
                            or abs(row.end_s - row.start_s - clock["exchange_s"]) > eps):
                        raise RuntimeError("native feeder exchange clock mismatch")
                    target = demand(anchor, feeder_t0, row.feeder_import, disturbance)
                    if event_kvar and clock["event_start_s"] <= slot < clock["event_stop_s"]:
                        target = PQ(target.kw, target.kvar + event_kvar)
                    _set_pq(eng, model, target)
                    end = _advance(eng, model, actual, slot + 1, clock)
                    record = dict(start_s=slot, pause_s=actual, end_pause_s=end,
                                  disturbance_kw=disturbance, requested_kw=command,
                                  terminals=row.storage, pcc=row.feeder_import, command=target,
                                  balance_kw=row.active_balance_error_kw, online=None)
                    # Save progress before post-stop sensor extraction; commands
                    # and DSS telemetry are never relabelled as Bus8 measurements.
                    append({"arm": name, "phase": "hold_command", "record": record,
                            "observed_native_hz": observed, "live_prefix_length": len(live),
                            "storage_energy": ledger})
                    staged.append(record)
                    prev, actual = actual, end
                eng.set_param(model, "SimulationCommand", "stop", nargout=0)
                _poll(eng, model, "stopped")
                hz = _native_frequency(eng)
                _verify_prefixes(prefix, hz)
                append({"arm": name, "phase": "native_frequency", "live_prefixes_verified": len(prefix),
                        "data": hz})
                physical = _physical_sensor(eng, model, sensor)
                physical["frequency_hz"] = [(s.time_s, s.hz) for s in hz]
                gaps = {"power": sensor["power_log_sample_seconds"] + eps,
                        "frequency": sensor["frequency_sample_seconds"] + eps}
                warm_gaps = dict(gaps, power=2 * sensor["power_log_sample_seconds"] + eps)
                rows, failures = [], []
                try:
                    warm = _check_measurement(anchor, physical, policy, first,
                                              policy["tail_seconds"], warm_gaps, settled=True)
                    append({"arm": name, "phase": "warmup_physical", "measurement": warm})
                except RuntimeError as err:
                    failures.append(f"warmup: {err}")
                for record in staged:
                    start, end = record["pause_s"], record["end_pause_s"]
                    audit = {}
                    try:
                        m = _check_measurement(record["command"], physical, policy,
                                               end, 0.2, gaps, settled=False,
                                               hold=(start, end), audit=audit)
                        rows.append(Hold(**record, bus8=PQ(m["mean_p_kw"], m["mean_q_kvar"])))
                    except RuntimeError as err:
                        failures.append(f"t={record['start_s']}: {err}")
                    append({"arm": name, "phase": "hold_physical", "t": record["start_s"],
                            "measurement": audit})
                if failures:
                    raise RuntimeError("; ".join(failures))
                return Arm(tuple(rows), hz, anchor, feeder_t0), tuple(feeder.storage_ids)
            finally:
                eng.quit()

def _json_default(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"cannot encode {type(obj).__name__}")
