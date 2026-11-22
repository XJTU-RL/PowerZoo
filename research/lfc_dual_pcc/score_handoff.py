"""Audit the sole withdrawal treatment on an identical clock and measured common service target."""
import argparse
import json
import math
from pathlib import Path
from .compare import load
from .handoff import HANDOFF
from .driver import sha


def area_positive(left,right,dt):
    if min(left,right)>=0: return dt*(left+right)/2
    if max(left,right)<=0: return 0.
    top=max(left,right)
    return dt*top*top/(2*abs(right-left))


def interval_area(sig,lo,hi,kw):
    shortage=positive=total=covered=0.
    for (a,va),(b,vb) in zip(sig,sig[1:]):
        l,r=max(lo,a),min(hi,b)
        if r<=l: continue
        if b-a>2/60+1e-7: raise ValueError("gap in physical energy trace")
        vl=va+(vb-va)*(l-a)/(b-a); vr=va+(vb-va)*(r-a)/(b-a)
        dt=r-l
        covered+=dt; total+=dt*(vl+vr)/2
        shortage+=area_positive(kw-vl,kw-vr,dt)
        positive+=area_positive(vl-kw,vr-kw,dt)
    if abs(covered-hi+lo)>1e-7: raise ValueError("incomplete energy span")
    return {"measured_kwh":total/3600,"shortfall_kwh":shortage/3600,"excess_kwh":positive/3600}


def signals(path):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    hz,=[r for r in rows if r["phase"]=="native_frequency"]
    return {r["port"]:r["p_kw"] for r in rows if r["phase"]=="physical_native"},hz


def dwell(points,target,band,duration,start):
    entry=None
    for t,v in points:
        if abs(v-target)<=band:
            if entry is None: entry=t
            if t-entry>=duration-1e-7: return entry-start
        else: entry=None
    return None


def audit(neutral,issued,delivered):
    cfg=json.loads(HANDOFF.read_bytes())
    z=load(neutral)
    if z["header"]["git"]!=cfg["baseline_parent"] or z["header"]["method"]!="none":
        raise ValueError("wrong physical reference source")
    zs,_=signals(neutral)
    summaries={}
    arms={}
    for mode,path in (("issued",issued),("delivered",delivered)):
        a=load(path);arms[mode]=a
        if a["header"]["handoff_sha256"]!=sha(HANDOFF) or a["header"]["arm"]!=mode:
            raise ValueError("release policy mismatch")
        if any(a["header"][k]!=z["header"][k] for k in ("seed","config_sha256","reference_sha256")):
            raise ValueError("unmatched plant, seed or scenario")
        sig,hz=signals(path)
        merged=[]
        for ((t,v),(u,w),(tv,y),(uw,x)) in zip(sig["A"],sig["B"],zs["A"],zs["B"],strict=True):
            if max(abs(t-u),abs(t-tv),abs(t-uw))>1e-7: raise ValueError("native sensor clocks differ")
            merged.append((t,y+x-v-w))
        frames=a["frames"]
        last=frames[-1]["handoff"]
        issue=last["b_issue_slot"]
        release=last["release_slot"]
        if issue is None or release is None: raise ValueError("no pick-up or withdrawal")
        metric_start=next(f["write_s"] for f in frames if f["used_slot"]==issue)
        end=metric_start+cfg["post_issue_observation_s"]
        local=[(t,v) for t,v in hz["samples"] if metric_start<t<=end+1e-7]
        prev=next(v for t,v in reversed(hz["samples"]) if t<metric_start)
        nadir=min(v for _,v in local)
        err=0.;table=[]; backup_actions=[]
        for i,(f,base) in enumerate(zip(frames,z["frames"],strict=True)):
            if (f["event_kw"]!=base["event_kw"] or f["slot"]!=base["slot"]
                    or abs(f["write_s"]-base["write_s"])>1e-7 or abs(f["end_s"]-base["end_s"])>1e-7):
                raise ValueError("unmatched physical intervals")
            g=f["handoff"]
            if g["backup_kw"]:
                backup_actions.append(f["slot"])
            for s in ("A","B"):
                for j in (0,1):
                    measured=a["sensors"][s][i][j]-z["sensors"][s][i][j]
                    err=max(err,abs(measured-(f["target"][s][j]-base["target"][s][j])))
            table.append({"action_slot":f["slot"],"action_s":f["write_s"],"read_freq_s":f["frequency"][0],
                          "read_freq_hz":f["frequency"][1],"request_kw":f["request"],"actual_pcc_kw":f["delivered"],
                          "return_source_slot":f["used_slot"],"mean_sensor_delta_support_kw":{
                              s:z["sensors"][s][i][0]-a["sensors"][s][i][0] for s in ("A","B")},
                          "backup_kw":g["backup_kw"]})
        if err>1.5 or len(backup_actions)>cfg["backup_max_ticks"]:
            raise ValueError("sensor transport or reserve budget gate")
        if mode=="issued" and release!=issue: raise ValueError("command gate delayed")
        if mode=="delivered":
            g=next(f["handoff"] for f in frames if f["slot"]==release)
            if g["release_reason"]=="B_PCC_confirmed":
                old=[f for f in frames if f["slot"] in (g["used_slot"]-1,g["used_slot"])]
                if len(old)!=2 or any(f["slot"]<issue or f["delivered"]["B"]<cfg["b_ready_kw"] for f in old):
                    raise ValueError("release without two observed B deliveries")
            elif release-issue!=cfg["max_wait_after_issue_s"]:
                raise ValueError("misused deadline")
        eps=[(t,v) for t,v in merged if metric_start<=t<=end]
        result=dict(issue_slot=issue,release_slot=release,release_reason=last["release_reason"],
                    window_s=[metric_start,end],signal="paired actual native Sensor7 + Sensor2 versus no-control",
                    **interval_area(merged,metric_start,end,cfg["pickup_kw"]),nadir_hz=nadir,
                    nadir_offset_preevent_hz=nadir-hz["reference_hz"],nadir_offset_prehandoff_hz=nadir-prev,
                    frequency_peak_to_peak_hz=max(v for _,v in local)-nadir,
                    frequency_settling_s=dwell(local,hz["reference_hz"],cfg["settling_band_hz"],cfg["settling_dwell_s"],metric_start),
                    support_settling_s=dwell(eps,cfg["pickup_kw"],cfg["delivery_band_kw"],cfg["delivery_dwell_frames"]-1,metric_start),
                    backup_requested=bool(backup_actions),backup_slots=backup_actions,
                    backup_requested_kwh=len(backup_actions)*cfg["backup_kw"]/3600,
                    max_pq_residual_kw_kvar=err,elapsed_s=a["result"]["elapsed_s"],timeline=table)
        summaries[mode]=result
    a,b=arms["issued"],arms["delivered"]
    if a["header"]["git"]!=b["header"]["git"] or summaries["issued"]["issue_slot"]!=summaries["delivered"]["issue_slot"]:
        raise ValueError("different source or activation boundary")
    issue=summaries["issued"]["issue_slot"]
    for u,v in zip(a["frames"],b["frames"],strict=True):
        if u["slot"]<issue and (u["request"]!=v["request"] or max(abs(u["delivered"][s]-v["delivered"][s]) for s in ("A","B"))>.005):
            raise ValueError("trajectories differ before treatment")
        # B has the same regular request; only a consumed, bounded backup may differ.
        if abs((u["request"]["B"]-u["handoff"]["backup_kw"])-(v["request"]["B"]-v["handoff"]["backup_kw"]))>1e-8:
            raise ValueError("nominal B bridge changed")
    return summaries


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("neutral","issued","delivered","out"): p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    result=audit(a.neutral,a.issued,a.delivered)
    with a.out.open("x") as f: json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
