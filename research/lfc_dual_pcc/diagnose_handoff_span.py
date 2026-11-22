"""Locate the existing handoff window extrema on measured injection and frequency clocks."""
import argparse
from pathlib import Path
from .compare import load
from .score_handoff import audit, signals
from .driver import dump, sha


def diagnose(neutral, issued, delivered):
    scored=audit(neutral,issued,delivered)
    z,_=signals(neutral)
    results={}
    for mode,path in (("issued",issued),("delivered",delivered)):
        a=load(path); ds,hz=signals(path); metrics=scored[mode]
        lo,hi=metrics["window_s"]
        points=[(t,f) for t,f in hz["samples"] if lo<t<=hi+1e-7]
        def at(sample):
            t,f=sample
            frame=next(v for v in a["frames"] if v["write_s"]<=t<v["end_s"])
            power={}
            for site in ("A","B"):
                arrivals=[i for i,(u,_) in enumerate(ds[site]) if u<=t+1e-8]
                idx=arrivals[-1];u,p=ds[site][idx];v,base=z[site][idx]
                if abs(u-v)>1e-7 or t-u>1/60+1e-7:raise ValueError("unpaired native sensor near frequency sample")
                power[site]=base-p
            source=next(v for v in a["frames"] if v["slot"]==frame["used_slot"])
            return {"time_s":t,"frequency_hz":f,"offset_uhz":(f-hz["reference_hz"])*1e6,
                    "grid_received_source_slot":frame["used_slot"],"returned_dss_delivery_kw":source["delivered"],
                    "absolute_measured_incremental_support_kw":power,"sum_support_kw":sum(power.values()),
                    "generating_request_kw":source["request"],"grid_event_kw":frame["event_kw"],
                    "backup_request_kw":frame["handoff"]["backup_kw"]}
        peak=max(points,key=lambda p:p[1]);low=min(points,key=lambda p:p[1])
        d=a["result"]["metrics"]["second_dip"]
        rows=[at(x) for x in points]
        p2p=(peak[1]-low[1])*1e6
        if abs(p2p-metrics["frequency_peak_to_peak_hz"]*1e6)>1e-5:raise ValueError("published span differs")
        results[mode]={"raw_sha256":sha(path),"window_s":[lo,hi],"frequency_peak":at(peak),
                       "frequency_trough":at(low),"span_uhz":p2p,"secondary_dip":d,"timeline":rows,
                       "release_slot":metrics["release_slot"],"release_reason":metrics["release_reason"],
                       "backup_trigger":metrics["backup_requested"]}
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("neutral","issued","delivered","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    dump(a.out,diagnose(a.neutral,a.issued,a.delivered))
