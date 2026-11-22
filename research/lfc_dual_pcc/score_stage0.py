"""Score the frozen, independently restarted native two-port wiring matrix."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .stage0 import CONFIG, demands


def near(a,b,tol,label):
    if not (math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol):
        raise ValueError(f"{label}: {a} vs {b}, tolerance {tol}")


def read(path,case,cfg):
    raw=path.read_bytes()
    def reject(value):
        raise ValueError(f"nonfinite JSON {value}")
    rows=[json.loads(s,parse_constant=reject) for s in raw.splitlines()]
    if (not rows or rows[-1]["phase"]!="complete" or
            any(r["phase"]=="failed" for r in rows)):
        raise ValueError(f"{case}: incomplete or failed")
    head,=[r for r in rows if r["phase"]=="manifest"]
    if head["case"]!=case or head["config_sha256"]!=hashlib.sha256(CONFIG.read_bytes()).hexdigest():
        raise ValueError("wrong source/configuration")
    base,=[r for r in rows if r["phase"]=="baseline"]
    hz,=[r["trace"] for r in rows if r["phase"]=="frequency"]
    holds=[r for r in rows if r["phase"]=="hold"]
    ports=cfg["ports"]
    c,g=cfg["clock"],cfg["gates"]
    slots=list(range(c["first_hold_s"],c["stop_s"]))
    sensors={p:[s for s in rows if s["phase"]=="sensor" and s["port"]==p] for p in ports}
    if ([r["slot"] for r in holds]!=slots or rows[-1]["holds"]!=len(slots)
            or any([s["slot"] for s in group]!=slots for group in sensors.values())):
        raise ValueError("incomplete port measurements")
    points=dict(hz)
    if len(points)!=len(hz) or any(abs(b[0]-a[0]-.2)>1e-7 for a,b in zip(hz,hz[1:])):
        raise ValueError("invalid native frequency clock")
    if len([r for r in rows if r["phase"]=="warm"])!=2:
        raise ValueError("missing warmup gate")
    energy={p:0.0 for p in ports}
    returned={p:0.0 for p in ports}
    volts=[math.inf,-math.inf]
    max_balance=0.
    for i,r in enumerate(holds):
        t0,t1,slot=r["write_s"],r["end_s"],slots[i]
        eligible=[z for z in holds[:i] if z["report"]["arrival_s"]<t0]
        prior=eligible[-1] if eligible else None
        want=prior["slot"] if prior else None
        if r["used_report"]!=want or (i>=2 and want!=slot-2):
            raise ValueError("wrong report age or future report")
        near(r["report"]["generated_s"],t1,1e-7,"generation")
        near(r["report"]["arrival_s"],t1+c["report_latency_s"],1e-7,"arrival")
        sample=r["frequency"]
        ts=sample["time_s"]
        if not 0<t0-ts<=.2+1e-7 or ts not in points:
            raise ValueError("feedback is stale or from future")
        near(sample["hz"],points[ts],1e-10,"native frequency readback")
        for p,spec in ports.items():
            obs=r["feeders"][p]
            req=demands(cfg,case,slot)[p]
            if list(req)!=r["request"][p] or not obs["snapshot"]["converged"]:
                raise ValueError("incorrect request or nonconvergence")
            near(obs["start_s"],i,1e-7,"DSS start")
            near(obs["end_s"],i+1,1e-7,"DSS end")
            sensor=sensors[p][i]
            near(sensor["write_s"],t0,1e-7,"sensor write alignment")
            near(sensor["end_s"],t1,1e-7,"sensor end alignment")
            for j in (0,1):
                near(r["report"]["pcc"][p][j],obs["pcc"][j],1e-8,"PCC report")
                near(obs["pcc"][j]-base["feeder"][p][j],obs["pcc_delta"][j],1e-8,"feeder delta")
                expected=prior["feeders"][p]["pcc_delta"][j] if prior else 0.
                near(r["target"][p][j]-spec["anchor_kw_kvar"][j],expected,1e-7,"applied feeder delta")
                near(obs["pulse_signed_kw_kvar"][j],req[j]+(10 if j==0 else 0),.05,"native device terminal")
            for k in ("residual_kw","residual_kvar"):
                residual=abs(obs["snapshot"][k])
                near(residual,0,g["feeder_balance_kw_kvar"],"native power balance")
                max_balance=max(max_balance,residual)
            phases=dict(obs["snapshot"]["phases_pu"])
            if len(phases)!=95 or not all(g["voltage_pu"][0]<=v<=g["voltage_pu"][1] for v in phases.values()):
                raise ValueError("native phase voltage screen")
            volts=[min(volts[0],min(phases.values())),max(volts[1],max(phases.values()))]
            delivered=-obs["pcc_delta"][0]*(obs["end_s"]-obs["start_s"])/3600
            near(delivered,obs["pcc_support_kwh"],1e-9,"delivered energy")
            energy[p]+=delivered
            returned[p]-=(r["target"][p][0]-spec["anchor_kw_kvar"][0])*(t1-t0)/3600
    for p in ports:
        # Events end well before the last two in-transit slots.
        near(returned[p],energy[p],1e-5,"two-clock event energy")
    return dict(holds=holds,hz=hz,sensors=sensors,head=head,base=base,
                metrics=dict(holds=len(holds),raw_sha256=hashlib.sha256(raw).hexdigest(),support_kwh=energy,
                             returned_kwh=returned,phase_pu=volts,max_feeder_balance=max_balance))


def score(folder):
    cfg=json.loads(CONFIG.read_bytes())
    runs={k:read(folder/(k+".jsonl"),k,cfg) for k in cfg["cases"]}
    zero=runs["zero_a"]
    g=cfg["gates"]
    summary={}
    for case,r in runs.items():
        for k in ("git","topology","reference_sha256","deck_tree","dss_version","config_sha256"):
            if r["head"][k]!=zero["head"][k]:
                raise ValueError("independent run source differs")
        if r["base"]!=zero["base"]:
            raise ValueError("unmatched feeder initialization or transmission anchors")
        freq=[]
        for (t,f),(z,v) in zip(r["hz"],zero["hz"],strict=True):
            near(t,z,1e-7,"paired native frequency timestamp")
            freq.append((t,f-v))
        live_residual=0.
        jitter=0.
        diagonals={}
        for i,(h,base) in enumerate(zip(r["holds"],zero["holds"],strict=True)):
            near(h["write_s"],base["write_s"],1e-7,"case write alignment")
            near(h["end_s"],base["end_s"],1e-7,"case end alignment")
            prior=i-2
            for p in cfg["ports"]:
                expected=[u-v for u,v in zip(h["target"][p],base["target"][p])]
                obs=[a-b for a,b in zip(r["sensors"][p][i]["pq"],zero["sensors"][p][i]["pq"])]
                request=demands(cfg,case,r["holds"][prior]["slot"])[p] if prior>=0 else (0.,0.)
                for j in range(2):
                    tol=g["neutral_sensor_kw_kvar"] if case.startswith("zero") else g["physical_delta_abs_kw_kvar"]
                    near(obs[j],expected[j],tol,"paired physical P/Q increment")
                    live_residual=max(live_residual,abs(obs[j]-expected[j]))
                    if not any(request):
                        near(expected[j],0,g["pcc_idle_kw_kvar"],"idle port changed")
                        jitter=max(jitter,abs(obs[j]))
                    if request[j]:
                        if obs[j]/request[j]<g["minimum_diagonal_fraction"]:
                            raise ValueError("missing independently signed receiver response")
                        diagonals.setdefault(p+(" P" if j==0 else " Q"),[]).append(obs[j])
        if case.startswith("zero"):
            near(max(abs(v) for _,v in freq),0,g["neutral_frequency_hz"],"neutral repeat")
        if case in ("a_only","b_only"):
            first_push=r["holds"][4]["write_s"]
            near(max(abs(v) for t,v in freq if t<first_push),0,g["frequency_detection_hz"],"pre-delivery response")
            direction=-1 if case=="a_only" else 1
            window=[(t,direction*v) for t,v in freq if first_push<t<r["holds"][7]["write_s"]]
            if not any(a>g["frequency_detection_hz"] and b>g["frequency_detection_hz"] for (_,a),(_,b) in zip(window,window[1:])):
                raise ValueError("no signed frequency response on isolated P pulse")
        summary[case]=dict(**r["metrics"],max_sensor_residual_kw_kvar=live_residual,
                           max_idle_sensor_delta_kw_kvar=jitter,
                           sensor_increments={k:[min(v),max(v)] for k,v in diagonals.items()},
                           paired_frequency_hz=[min(v for _,v in freq),max(v for _,v in freq)])
    return {"stage0_passed":True,"claim":"two fixed-voltage feeders and two measured incremental P/Q ports",
            "native_frequency_bus":8,"git":zero["head"]["git"],"cases":summary}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("summary",type=Path)
    args=parser.parse_args()
    result=score(args.folder)
    with args.summary.open("x") as fp:
        json.dump(result,fp,indent=2,allow_nan=False)
        fp.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"}))
