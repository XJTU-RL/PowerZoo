"""Compute native-sample frequency and PCC metrics from private open/feedback JSONL."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def frequency_metrics(points, start, release, end, band=0.00005, dwell=1.0):
    ref = statistics.mean(v for t,v in points if start-2 < t < start)
    selected = [(t,v) for t,v in points if start < t < end]
    if len(selected) < 2 or any(not math.isfinite(t) or not math.isfinite(v) for t,v in points):
        raise ValueError("missing or nonfinite native trace")
    max_rocof = iae = 0.
    for (t,a),(s,b) in zip(selected, selected[1:]):
        dt = s-t
        if abs(dt-.2) > 1e-7:
            raise ValueError("native sample gap")
        max_rocof = max(max_rocof, abs(b-a)/dt)
        iae += dt*(abs(a-ref)+abs(b-ref))/2
    excursion = any(abs(v-ref) > band for t,v in selected if t < release)
    entry = recovered = None
    for t,v in selected:
        if not excursion or t <= release or abs(v-ref) > band:
            entry = None
        elif entry is None:
            entry = t
        elif t-entry >= dwell-1e-7:
            recovered = entry
            break
    hz = [v for _,v in selected]
    return dict(reference_hz=ref, first_sample_s=selected[0][0], last_sample_s=selected[-1][0],
                nadir_hz=min(hz),peak_hz=max(hz), max_abs_deviation_hz=max(abs(v-ref) for v in hz),
                max_abs_rocof_hz_s=max_rocof,rocof_window_s=.2, iae_hz_s=iae,
                recovery_from_release_s=None if recovered is None else recovered-release,
                recovery_band_hz=band,recovery_dwell_s=dwell)


def electrical_metrics(groups, requested, design):
    result = {"failed_power_flows_in_completed_arm":0}
    if groups["feeder"]:
        frames = [r["dss"] for r in groups["feeder"]]
        v = [value for f in frames for _,value in f["snapshot"]["phases_pu"]]
        windings = [w for f in frames for w in f["snapshot"]["windings"]]
        result.update(min_voltage_pu=min(v),max_voltage_pu=max(v),
                      max_winding_loading_pu=max(w["measured_kva"]/w["rating_kva"] for w in windings),
                      measured_max_l32_amp=max(c["terminal_current_amp"] for f in frames
                          for terminal in f["l32_terminal_flows"] for c in terminal["conductors"]),
                      soc_status="flexible load; no inventory of stored energy",
                      saturated_holds=sum(abs(k)>=design["controller"]["capacity_kw"]-1e-7
                                          for k in requested))
    elif groups["hold_observed_after_finish"]:
        obs = [r["actual"] for r in groups["hold_observed_after_finish"]]
        soc = [s for point in obs for s in point["soc"]]
        devices = [d for point in obs for d in point["storage"]]
        v = [value for d in devices for value in d["voltage_pu"]]
        result.update(min_soc=min(soc),max_soc=max(soc),min_device_voltage_pu=min(v),
                      max_device_voltage_pu=max(v),max_device_conductor_a=max(
                          a for d in devices for a in d["conductor_a"].values()),
                      max_energy_error_kwh=max(abs(e["residual_kwh"])
                          for h in groups["hold_verified"] for e in h["ledger"]),
                      saturated_holds=sum(k>=4*design["per_device_max_kw"]-1e-7 for k in requested))
    elif groups["hold_command"]:
        soc = [s["soc_after"] for h in groups["hold_command"] for s in h["record"]["terminals"]]
        result.update(min_soc=min(soc),max_soc=max(soc),
                      voltage_current_status="outside open receiver instrumentation")
    result["line_rating_status"] = "no sourced line ampacity; no clearance"
    return result


def summarize(path):
    by_case = defaultdict(list)
    collected = [json.loads(line) for line in path.read_text().splitlines()]
    marker = collected[-1]["phase"]
    if marker not in ("complete","failed"):
        raise ValueError("batch is incomplete")
    for row in collected:
        key = row.get("arm",row.get("case"))
        if key:
            by_case[key].append(row)
    head = collected[0]
    design = head.get("design", head.get("config"))
    clock = design.get("clock",design)
    metrics = {}
    for case, rows in by_case.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["phase"]].append(row)
        if groups["native_frequency"]:
            points = [(s["time_s"],s["hz"]) for s in groups["native_frequency"][0]["data"]]
            holds = [{"slot":h["record"]["start_s"],"write_s":h["record"]["pause_s"],
                      "end_s":h["record"]["end_pause_s"]} for h in groups["hold_command"]]
            source = groups["feeder_identity"][0]["feeder_t0"]["kw"]
            deltas = [h["record"]["pcc"]["kw"]-source for h in groups["hold_command"]]
            requested = [sum(h["record"]["requested_kw"]) for h in groups["hold_command"]]
        elif groups["native_logs"] and groups["sensor_hold"]:
            points = groups["native_logs"][0]["signals"]["frequency_hz"]
            holds = groups["sensor_hold"]
            feeder = groups["delivered"] or groups["feeder"]
            deltas = [h["pcc_delta"][0] for h in feeder]
            requested = [(h["request_kw"] if "request_kw" in h else h["support_kw"]) for h in feeder]
        else:
            metrics[case] = {"status":"incomplete"}
            continue
        slots = [h["slot"] for h in holds]
        if not slots or slots != list(range(slots[0], clock["stop_s"])) or len(holds) != len(deltas):
            metrics[case] = {"status":"incomplete", "verified_slots":slots}
            continue
        start = next(h["write_s"] for h in holds if h["slot"] == clock["event_start_s"])
        release = next(h["write_s"] for h in holds if h["slot"] == clock["event_stop_s"])
        m = frequency_metrics(points, start, release, holds[-1]["end_s"])
        support = energy = square = seconds = 0.
        for hold, delta, req in zip(holds,deltas,requested,strict=True):
            if hold["write_s"] >= start:
                dt = hold["end_s"] - hold["write_s"]
                support = max(support,abs(delta))
                energy -= delta*dt/3600
                square += dt*(-delta-req)**2
                seconds += dt
        m.update(pcc_peak_abs_kw=support,pcc_net_support_kwh=energy,
                 pcc_vs_device_request_rmse_kw=math.sqrt(square/seconds))
        changes = [abs(now-prev) for now,prev in zip(requested, [0.] + requested[:-1])]
        event_energy = sum(-v * (h["end_s"]-h["write_s"]) / 3600
                           for h,v in zip(holds,deltas,strict=True) if start <= h["write_s"] < release)
        m.update(event_pcc_net_support_kwh=event_energy, peak_requested_kw=max(map(abs,requested)),
                 peak_request_change_kw_per_s=max(changes))
        m.update(electrical_metrics(groups, requested, design))
        metrics[case]=m
    return {"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "batch_phase":marker,"cases":metrics}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("private_jsonl",type=Path)
    print(json.dumps(summarize(p.parse_args().private_jsonl), indent=2, allow_nan=False))
