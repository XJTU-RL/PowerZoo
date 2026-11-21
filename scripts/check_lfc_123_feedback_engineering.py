"""Bounded native IEEE123 feedback experiment after the signed open gate."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from envs.lfc.native_123_availability import envelope
from envs.lfc.native_feeder_123 import Native123Feeder, PQ
from scripts.check_lfc_123_availability_dynamic import _hold
from scripts.check_lfc_123_coupling import SCENARIO as SOURCE
from envs.lfc.receiver_123 import SCENARIO as RECEIVER_LIMITS
from scripts.lfc_bus8_receiver import _json_default
from scripts.check_lfc_123_transfer import DESIGN
from scripts.lfc_bus8_exchange import exchange

PROTOCOL = DESIGN.with_name("feedback_engineering_v1.json")


def open_gate(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or rows[-1].get("phase") != "complete":
        raise ValueError("independent open batch missing or incomplete")
    manifest, = [r for r in rows if r["phase"] == "inputs"]
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (SOURCE, RECEIVER_LIMITS, DESIGN)}
    # Only the byte-identical archived calibration used the precursor file.
    if digest == "0b901a95258b0c7616e10b998555f5b9c5e5c3611be5869b4f367fd3b5bf9c17" and expected[RECEIVER_LIMITS.name] == "c8ada609cfb8393da17f032b6c11fdc1bc1c7031958c5eba592d84ead10d737b":
        del expected[RECEIVER_LIMITS.name]
        expected["droop_v1.json"] = "7d2436caa9e88a59413c4805b4727f710927ae95b3c9a9af9c0478f826be1cfb"
    if manifest["sha256"] != expected:
        raise ValueError("different open source or design")
    pairs = {r["case"]:r for r in rows if r["phase"] == "paired"}
    if (set(pairs) != {"zero_b", "p_plus", "p_minus", "q_plus", "q_minus"}
            or pairs["zero_b"]["status"] != "zero_repeat_passed"
            or any(pairs[k]["status"] != "visible" for k in ("p_plus","p_minus"))):
        raise ValueError("zero/signed P open gate failed")
    return digest, pairs["p_plus"]["threshold_hz"]


def intended(reference, observed, previous, spec):
    raw = spec["gain_kw_per_hz"] * max(0, reference-observed-spec["deadband_hz"])
    filtered = (1-spec["alpha"])*previous + spec["alpha"]*raw
    ramp = spec["aggregate_ramp_kw_s"]
    return min(4*spec["per_device_max_kw"], previous+ramp, max(0, previous-ramp, filtered))


def run(append, noise):
    spec = json.loads(PROTOCOL.read_bytes())
    clock = json.loads(RECEIVER_LIMITS.read_bytes())["clock"]
    clock.update({k:spec[k] for k in ("event_start_s","event_stop_s","stop_s")})
    cfg = json.loads(SOURCE.read_bytes())
    anchor = PQ(cfg["source_anchor"]["kw"], cfg["source_anchor"]["kvar"])
    limits = envelope()
    outputs = {}
    for name in spec["cases"]:
        history = []
        def emit(row):
            append(dict(case=name, **row))
        emit({"phase":"start"})
        with Native123Feeder(anchor) as feeder:
            previous = dict.fromkeys(feeder.storage_ids, 0.)
            emit({"phase":"feeder_source", "sha256":feeder.session.model_sha256,
                  "scenario":feeder.scenario_sha256,"operating_point":feeder.calibration.feeder_t0})
            def dispatch(slot, pause, sample):
                nonlocal previous
                active = clock["event_start_s"] <= slot < clock["event_stop_s"]
                total = 0.
                if name != "uncontrolled" and slot >= clock["event_start_s"]:
                    total = intended(reference, sample.hz, sum(previous.values()), spec)
                command = dict.fromkeys(feeder.storage_ids, total/len(previous))
                op = _hold(feeder, previous, command, limits, emit)
                if abs(op.clock_s - (slot+1-clock["first_hold_s"])) > 1e-7:
                    raise RuntimeError("native frame clock mismatch")
                delta = (op.source_pq.kw-feeder.calibration.feeder_t0.kw,
                         op.source_pq.kvar-feeder.calibration.feeder_t0.kvar)
                applied = (anchor.kw + (spec["event_kw"] if active else 0)
                           + (0 if name == "cut_return" else delta[0]),
                           anchor.kvar + (0 if name == "cut_return" else delta[1]))
                history.append(dict(slot=slot,sample=sample,request_kw=total,
                                    pcc_delta=delta,source=op.source_pq,target=applied))
                emit(dict(phase="delivered", **history[-1]))
                previous = command
                return applied
            hz, sensors = exchange(clock, dispatch, emit)
        outputs[name] = hz, sensors, history
        t_on = next(r["write_s"] for r in sensors if r["slot"] == clock["event_start_s"])
        if name == "uncontrolled":
            reference = statistics.mean(s.hz for s in hz if t_on-2 < s.time_s < t_on)
            emit({"phase":"reference", "reference_hz":reference})
            continue
        base_hz, base_sensors, base_history = outputs["uncontrolled"]
        f = []
        for a,b in zip(hz,base_hz,strict=True):
            if abs(a.time_s-b.time_s) > 1e-7:
                raise ValueError("unaligned native frequency")
            f.append((a.time_s,a.hz-b.hz))
        if any(abs(v) > noise for t,v in f if name == "cut_return" or t < t_on):
            raise ValueError("cut-return or pre-event trace differs from uncontrolled")
        streak = longest = 0
        for sensor, ref, row, other in zip(sensors,base_sensors,history,base_history,strict=True):
            if (abs(sensor["write_s"]-ref["write_s"]) > 1e-7 or
                    abs(sensor["end_s"]-ref["end_s"]) > 1e-7):
                raise ValueError("unpaired physical windows")
            for m,b,target,base in zip(sensor["sensor"],ref["sensor"],row["target"],other["target"]):
                if abs(m-b-target+base) > spec["paired_sensor_tol"]:
                    raise ValueError("physical return gain mismatch")
            delivered = sensor["sensor"][0]-ref["sensor"][0]
            active = clock["event_start_s"] <= row["slot"] < clock["event_stop_s"]
            visible = active and delivered < -spec["min_delivery_kw"] and row["request_kw"] > 0
            streak = streak + 1 if visible else 0
            longest = max(longest,streak)
        significant = [t for t,v in f if t > t_on and abs(v)>noise]
        emit({"phase":"assessment", "max_consecutive":longest,
              "max_paired_hz":max(abs(v) for _,v in f),
              "first_feedback_difference_s":next(iter(significant), None)})
        if name == "closed" and (longest < spec["required_consecutive"] or not significant):
            raise RuntimeError("feedback continuity or downstream frequency difference unresolved")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("open_report",type=Path)
    p.add_argument("report",type=Path)
    args = p.parse_args()
    digest, noise = open_gate(args.open_report)
    os.umask(0o077)
    with args.report.open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row, default=_json_default, allow_nan=False)+"\n")
            stream.flush()
            if row["phase"] in ("assessment","failed","complete"):
                print(json.dumps(row), flush=True)
        emit({"phase":"manifest","open_sha256":digest,"noise_hz":noise,
              "design":json.loads(PROTOCOL.read_bytes()),"git":subprocess.check_output(
                  ["git","rev-parse","HEAD"],text=True).strip()})
        try:
            run(emit, noise)
            emit({"phase":"complete"})
        except Exception as error:
            emit({"phase":"failed","reason":repr(error)})
            raise


if __name__ == "__main__":
    main()
