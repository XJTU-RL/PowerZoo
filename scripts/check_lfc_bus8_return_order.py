"""Verify the two broken-link controls and order native physical readbacks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def first_native_after(series, base, write, band):
    first = None
    for (t,v),(bt,bv) in zip(series,base,strict=True):
        if abs(t-bt)>1e-7:
            raise ValueError("incompatible native timestamps")
        if abs(v-bv)>band:
            if t < write-1e-7:
                raise ValueError("paired response exceeds threshold before first feedback write")
            if t > write+1e-7 and first is None:
                first=t
    return first


def assess(path):
    groups=defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for line in f:
            row=json.loads(line)
            if row.get("case"):
                groups[row["case"]][row["phase"]].append(row)
    if row["phase"] not in ("complete","failed"):
        raise ValueError("unfinished report")
    if not groups["closed"]["native_logs"]:
        raise ValueError("closed native evidence unavailable")
    is123="uncontrolled" in groups
    off=groups["uncontrolled" if is123 else "plus"]
    cut=groups["cut_return" if is123 else "feedback_cut_return"]
    closed=groups["closed"]
    frames="delivered" if is123 else "feeder"
    req="request_kw" if is123 else "support_kw"
    band=.00001
    if not is123:
        band=groups["zero_b"]["assessment"][0]["band_hz"]
    else:
        with path.open() as f:
            band=json.loads(next(f))["noise_hz"]
    def signals(arm):
        data,=arm["native_logs"]
        return data["signals"]
    ctr, detached, baseline = (signals(arm) for arm in (closed,cut,off))
    null=0.
    for (t,v),(u,b) in zip(detached["frequency_hz"],baseline["frequency_hz"],strict=True):
        if abs(t-u)>1e-7:
            raise ValueError("cut frequency grid mismatch")
        null=max(null,abs(v-b))
    if null>band:
        raise ValueError("cut return changed frequency relative to no control")
    # A disabled support action cannot establish that a severed return was exercised.
    if not any(abs(h[req])>0.05 and abs(h["pcc_delta"][0])>0.05 for h in cut[frames]):
        raise ValueError("detached control did not deliver any measurable PCC support")
    write=None
    for w,h in zip(closed["write"],closed[frames],strict=True):
        if h["slot"]!=w["slot"] or w["native"]["time_s"]>=w["pause_s"]:
            raise ValueError("missing earlier same-arm native control input")
        if abs(h[req])>.05 and write is None:
            write=w["pause_s"]
    if write is None:
        raise ValueError("no feedback command")
    sensor=first_native_after(ctr["p_kw"],detached["p_kw"],write,.5)
    freq=first_native_after(ctr["frequency_hz"],detached["frequency_hz"],write,band)
    results=dict(first_active_write_s=write,first_resolved_sensor_p_s=sensor,
                 first_resolved_frequency_s=freq,frequency_band_hz=band,
                 cut_vs_no_control_max_abs_hz=null,
                 cut_peak_abs_pcc_kw=max(abs(h["pcc_delta"][0]) for h in cut[frames]),
                 ordered_measured_return=sensor is not None and freq is not None and sensor<=freq)
    # This reports order at the native resolutions, without claiming sub-sample delay.
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl",type=Path)
    result=assess(p.parse_args().jsonl)
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result["ordered_measured_return"]:
        raise SystemExit(1)
