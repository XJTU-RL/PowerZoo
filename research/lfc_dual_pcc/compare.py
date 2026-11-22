"""Compare native trials on the same disturbance and verify both measured return branches."""
import argparse
import hashlib
import json
from pathlib import Path

from .case_a_plant import CONFIG


def load(path):
    rs=[json.loads(l) for l in path.read_text().splitlines()]
    if rs[0]["phase"]!="manifest" or rs[0]["mode"]!="native" or rs[-1]["phase"]!="complete" or any(x["phase"]=="failed" for x in rs):
        raise ValueError("requires complete native episodes")
    return dict(header=rs[0],frames=[r for r in rs if r["phase"]=="hold"],
                roles=[r for r in rs if r["phase"]=="decision"],
                windows=[r for r in rs if r["phase"]=="window"],
                sensors={r["port"]:r["means"] for r in rs if r["phase"]=="sensor_means"},result=rs[-1])


def audit(reference,paths,training_pair):
    cfg=json.loads(CONFIG.read_bytes())
    neutral=load(reference); records={}; trials=[]
    for path in paths:
        a=load(path); trials.append(a)
        for k in ("git","config_sha256","reference_sha256","seed"):
            if a["header"][k]!=neutral["header"][k]: raise ValueError(f"unmatched {k}")
        if a["result"]["event_split"]!=neutral["result"]["event_split"]: raise ValueError("mismatched event")
        maxerr=0.
        grid=cfg["clock"]
        for i,(f,z) in enumerate(zip(a["frames"],neutral["frames"],strict=True)):
            if f["slot"]!=z["slot"] or abs(f["write_s"]-z["write_s"])>1e-7 or abs(f["end_s"]-z["end_s"])>1e-7 or f["event_kw"]!=z["event_kw"]:
                raise ValueError("unpaired event or write clock")
            if f["used_slot"] is not None:
                if f["used_slot"]!=f["slot"]-2 or not f["received_s"]<f["write_s"]: raise ValueError("future or stale report")
                old=a["frames"][i-2]
            if not 0<f["write_s"]-f["frequency"][0]<=grid["native_frequency_period_s"]+1e-7: raise ValueError("native frequency age")
            for p in cfg["sites"]:
                if f["used_slot"] is not None:
                    delta=f["target"][p][0]-neutral["frames"][i]["target"][p][0]
                    expectation=-old["delivered"][p]+neutral["frames"][i-2]["delivered"][p]
                    if abs(delta-expectation)>1e-6: raise ValueError("delivery attributed to wrong request frame")
                for j in range(2):
                    expected=f["target"][p][j]-z["target"][p][j]
                    actual=a["sensors"][p][i][j]-neutral["sensors"][p][i][j]
                    maxerr=max(abs(actual-expected),maxerr)
        if maxerr>cfg["gates"]["sensor_increment_abs_kw_kvar"]:
            raise ValueError(f"physical return residual {maxerr}")
        records[path.name+":"+a["header"]["arm"]]=dict(metrics=a["result"]["metrics"],max_sensor_residual_kw_kvar=maxerr,
                     raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),elapsed_s=a["result"]["elapsed_s"])
    if training_pair:
        if len(trials)!=2 or {a["header"]["arm"] for a in trials}!={"attribution_on","attribution_off"}: raise ValueError("wrong training pair")
        a,b=trials
        if (a["result"]["model_before"]!=b["result"]["model_before"] or a["roles"]!=b["roles"] or
                any(x["actions"]!=y["actions"] for x,y in zip(a["frames"],b["frames"],strict=True))):
            raise ValueError("paired treatment changed sampled trajectories before first update")
        for x,y in zip(a["frames"],b["frames"],strict=True):
            if any(abs(x["delivered"][p]-y["delivered"][p])>1e-6 for p in cfg["sites"]): raise ValueError("paired physical delivery mismatch")
        for a in trials:
            certs=[s for w in a["windows"] for s in w["sites"].values()]
            records[a["header"]["arm"]+"_credit"]=dict(certified=sum(s["certified"] for s in certs),
                     total=len(certs),gradient=a["result"]["optimizer"]["gradient_norm"],
                     delta=a["result"]["optimizer"]["parameter_delta_l2"],
                     losses=a["result"]["optimizer"]["loss"],model_after=a["result"]["model_after"])
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--training-pair",action="store_true")
    p.add_argument("trials",nargs="+",type=Path)
    args=p.parse_args()
    result=audit(args.reference,args.trials,args.training_pair)
    with args.out.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write("\n")
