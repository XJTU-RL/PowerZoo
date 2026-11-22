"""Metrics aligned to the measured transmission write and release clocks."""
import math


def measure(frames,hz,sensors,reference,event):
    start=next(f["write_s"] for f in frames if f["slot"]==event["start"])
    release=next(f["write_s"] for f in frames if f["slot"]==event["stop"])
    obs=[(t,v-reference) for t,v in hz if start<t<frames[-1]["write_s"]]
    integrated=sum((abs(a)+abs(b))*(t-s)/2 for (s,a),(t,b) in zip(obs,obs[1:]))
    dwell=None
    for t,v in obs:
        if t<release: continue
        if abs(v)<=.00005:
            if dwell is None: dwell=t
            if t-dwell>=1-1e-7:
                break
        else: dwell=None
    if dwell is not None and obs[-1][0]-dwell<1-1e-7: dwell=None
    extrema=[(t,v) for (s,a),(t,v),(u,b) in zip(obs,obs[1:],obs[2:]) if a>v and b>v and v<-.00001]
    second=None
    if extrema:
        t0,v0=extrema[0]
        for t,v in extrema[1:]:
            recovered=max(y for x,y in obs if t0<=x<=t)
            if recovered-v0>=.00001 and recovered-v>=.00001:
                second={"time_s":t,"deviation_hz":v,"fall_from_local_recovery_hz":recovered-v}
                break
    def values(site):
        return [f["actual"][site] for f in frames]
    per={}
    for s in ("A","B"):
        rows=values(s)
        gap=[max(0.,f["request"][s]-f["delivered"][s]) for f in frames]
        per[s]=dict(peak_delivery_kw=max(f["delivered"][s] for f in frames),
                    energy_kwh=sum(f["delivered"][s] for f in frames)/3600,
                    local_tracking_shortfall_kwh=sum(gap)/3600,peak_local_shortfall_kw=max(gap),
                    min_soc=min(min(r["soc_after"]) for r in rows),max_soc=max(max(r["soc_after"]) for r in rows),
                    min_phase_pu=min(min(r["voltage"].values()) for r in rows),
                    max_phase_pu=max(max(r["voltage"].values()) for r in rows),
                    peak_device_kw=max(max(r["battery_kw"]) for r in rows),
                    pv_kw=[min(r["pv_kw"] for r in rows),max(r["pv_kw"] for r in rows)])
    gaps=[max(0.,sum(f["request"].values())-sum(f["delivered"].values())) for f in frames]
    return dict(event_write_s=start,event_release_s=release,reference_hz=reference,nadir_hz=reference+min(v for _,v in obs),
                zenith_hz=reference+max(v for _,v in obs),iae_hz_s=integrated,
                max_rocof_hz_s=max(abs(b-a)/(t-s) for (s,a),(t,b) in zip(obs,obs[1:])),
                recovery_s=None if dwell is None else dwell-release,second_dip=second,
                positive_rebound_hz=max(0.,max(v for t,v in obs if t>release)),per_port=per,
                net_shortfall_kwh=sum(gaps)/3600,net_peak_shortfall_kw=max(gaps),
                completed_native_steps=len(frames),completed_step_power_flow_failures=0,
                ace=None,thermal_rating_status="no independently sourced line limits")
