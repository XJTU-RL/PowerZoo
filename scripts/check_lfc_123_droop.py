"""Opt-in optimized 123 fixed-droop diagnostic; no training or default changes."""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from envs.lfc.droop_123_draft import Arm, Frequency, Hold, assess, demand, droop, read_causal
from envs.lfc.droop_123_draft import SCENARIO as DROOP_SCENARIO
from envs.lfc.native_feeder_123 import Native123Feeder, PQ
from scripts.check_lfc_123_coupling import (
    SCENARIO as TRANSPORT_SCENARIO,
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


def _quiet_reference(arm, clock):
    t = clock["event_start_s"]
    points = [s.hz for s in arm.hz if t - clock["quiet_tail_s"] <= s.time_s < t]
    if not points:
        raise RuntimeError("no stopped baseline quiet-tail reference")
    # Same implementation as pairing assessor.
    if not all(map(math.isfinite, points)):
        raise RuntimeError("nonfinite stopped quiet-tail reference")
    return math.fsum(points) / len(points)


def _native_arm(name, reference, cfg, spec, append):
    import matlab.engine

    clock, sensor, policy = spec["clock"], cfg["sensor"], cfg["warmup"]
    eps = clock["timestamp_epsilon_s"]
    first, last = clock["first_hold_s"], clock["stop_s"]
    anchor = PQ(cfg["source_anchor"]["kw"], cfg["source_anchor"]["kvar"])
    prefix, staged = [], []
    with tempfile.TemporaryDirectory(prefix="pz-dss-droop-") as temp:
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
                    chosen = None
                    command = (0.0,) * 4
                    live = _native_frequency(eng)
                    observed = read_causal(lambda: live, prev, actual, clock)
                    prefix.append(live)
                    if name == "disturbed_fixed_droop" and slot >= clock["event_start_s"]:
                        chosen = observed
                        command = droop(reference, chosen, spec["controller"])
                    disturbance = (
                        spec["disturbance_kw"] if name != "undisturbed_drift"
                        and clock["event_start_s"] <= slot < clock["event_stop_s"] else 0
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
                    g = spec["gates"]
                    if (
                        tuple(x.device for x in row.storage) != tuple(feeder.storage_ids)
                        or abs(row.active_balance_error_kw) > g["feeder_balance_kw"]
                        or any(
                            abs(x.measured_injection.kw - req) > g["storage_p_kw"]
                            or abs(x.measured_injection.kvar) > g["storage_q_kvar"]
                            for x, req in zip(row.storage, command, strict=True)
                        )
                    ):
                        raise RuntimeError("native storage terminal or PCC pre-application gate failed")
                    if (abs(row.start_s - (slot - first)) > eps
                            or abs(row.end_s - row.start_s - clock["exchange_s"]) > eps):
                        raise RuntimeError("native feeder exchange clock mismatch")
                    target = demand(anchor, feeder_t0, row.feeder_import, disturbance)
                    _set_pq(eng, model, target)
                    end = _advance(eng, model, actual, slot + 1, clock)
                    record = dict(start_s=slot, pause_s=actual, end_pause_s=end,
                                  disturbance_kw=disturbance, requested_kw=command,
                                  terminals=row.storage, pcc=row.feeder_import, command=target,
                                  balance_kw=row.active_balance_error_kw, online=chosen)
                    # Save progress before post-stop sensor extraction; commands
                    # and DSS telemetry are never relabelled as Bus8 measurements.
                    append({"arm": name, "phase": "hold_command", "record": record,
                            "observed_native_hz": observed, "live_prefix_length": len(live)})
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

def _run_arms(cfg, spec, append, run_arm=_native_arm):
    arms = {}
    arm_names = ("undisturbed_drift", "disturbed_no_control", "disturbed_fixed_droop")
    if tuple(spec["arms"]) != arm_names:
        raise ValueError("arm order differs from the approved design")
    c, p = spec["clock"], spec["controller"]
    if (
        spec["schema_version"] != 1 or p["feeder_gain"] != 1
        or cfg["source_anchor"]["feeder_gain"] != 1
        or c["first_hold_s"] != cfg["warmup"]["seconds"]
        or c["native_frequency_period_s"] != cfg["sensor"]["frequency_sample_seconds"]
        or not 0 < spec["disturbance_kw"] < 4 * p["per_storage_limit_kw"]
    ):
        raise ValueError("unexpected study clock, gain or disturbance headroom")
    reference = None
    storage_ids = None
    for name in arm_names:
        append({"phase": "arm_started", "arm": name, "frozen_reference_hz": reference})
        arm, ids = run_arm(name, reference, cfg, spec, append)
        if storage_ids is None:
            storage_ids = ids
        if ids != storage_ids:
            raise RuntimeError("storage IDs changed between paired arms")
        arms[name] = arm
        if name == "disturbed_no_control":
            reference = _quiet_reference(arm, c)
            append({"phase": "reference_frozen", "arm": name, "hz": reference})
    return assess(arms, storage_ids, spec)


def _json_default(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"cannot encode {type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="new private JSONL file, one event per arm/hold")
    options = parser.parse_args()
    manifest, transport = DROOP_SCENARIO.read_bytes(), TRANSPORT_SCENARIO.read_bytes()
    with open(options.report, "x", encoding="utf8",
              opener=lambda path, flags: os.open(path, flags, 0o600)) as handle:
        def append(record):
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")
            handle.flush()

        append({"phase": "manifest", "sha256": hashlib.sha256(manifest).hexdigest(),
                "transport_sha256": hashlib.sha256(transport).hexdigest()})
        try:
            result = _run_arms(json.loads(transport), json.loads(manifest), append)
        except Exception as err:
            result = {"status": "inconclusive", "reason": f"{type(err).__name__}: {err}"}
        append({"phase": "outcome", **result})
    print(json.dumps(result, allow_nan=False))
    if result["status"] == "inconclusive":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
