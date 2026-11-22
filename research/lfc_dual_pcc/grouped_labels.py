"""Restore grouped native states and certify sparse ramp-and-hold witnesses."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .audit_capability import check_samples
from .capability_state import compare_step, restore
from .case_a_plant import feeder
from .compare import load
from .driver import dump, sha
from .grouped_source import PROTOCOL, condition, read
from .probe_capability import run_policy, start
from .reach_hold import RULE, select
from .restore_capability import physical, timeline
from .score_handoff import signals

SOURCE_GIT="0163ee2ff5c3765b1fad8ed5cb17479153b3f231"
TOL={"time_s":1e-7,"pq_kw_kvar":.005,"voltage_pu":1e-6,"soc_pu":1e-8,"device_kw":.005}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def asof(frames,slot,site,cfg):
    k=slot-frames[0]["slot"]; now=frames[k]; prior=frames[k-1]; a=prior["actual"][site]
    if now["used_slot"]!=slot-2 or abs(prior["end_s"]-now["write_s"])>1e-7 or now["frequency"][0]>=now["write_s"]:
        raise ValueError("noncausal initial state")
    bucket=cfg["storage"]; v=list(a["voltage"].values()); ex=cfg["local_exogenous"]
    previous=cfg["actions_battery_pv_kw"][prior["actions"][site]]
    usable=sum((s-bucket["reserve"])*bucket["kwh"] for s in a["soc_after"])
    inputs={"usable_dc_kwh":usable,"min_soc":min(a["soc_after"]),"battery_kw":sum(a["battery_kw"]),
            "pv_kw":a["pv_kw"],"pcc_import_kw":a["pcc"][0],"pcc_import_kvar":a["pcc"][1],
            "voltage_lower_margin_pu":min(v)-cfg["gates"]["voltage"][0],
            "voltage_upper_margin_pu":cfg["gates"]["voltage"][1]-max(v),
            "previous_nominal_kw":sum(previous),"ramp_up_kw_s":cfg["ramp_up"][site],
            "ramp_down_kw_s":cfg["ramp_down"],"load_multiplier":ex["load_mult"],"irradiance":ex["irradiance"]}
    state={"as_of_s":now["write_s"],"site":site,"prefix_actions":[r["actions"][site] for r in frames[:k]],
           "local_pq":a["pcc"],"local_voltage_pu":a["voltage"],"local_soc":a["soc_after"],
           "previous_nominal_kw":sum(previous)}
    return state,inputs


def signed_paths(paths):
    return {cap:[b["actual"]["pcc"][0]-a["actual"]["pcc"][0] for a,b in zip(rows,paths[0],strict=True)]
            for cap,rows in paths.items()}


def query_witness(net,request,h,rule):
    good=[]
    for cap,power in net.items():
        for k in range(1,rule["reach_deadline_s"]+1):
            segment=power[k-1:k+h]
            if len(segment)==h+1 and min(segment)>=request+rule["power_guard_kw"]-1e-8:
                good.append((k,cap))
    return min(good) if good else (None,None)


def state_cases(deck,reference,group,site,slot,cfg,frames,out,protocol,rule):
    study=read(PROTOCOL)
    state,x=asof(frames,slot,site,cfg)
    with feeder(deck,reference,cfg,site) as f:
        restore(f,site,frames,slot,TOL)
        for row in frames[slot-frames[0]["slot"]:slot-frames[0]["slot"]+2]:
            compare_step(f.execute(row["actions"][site]),row["actual"][site],TOL)
    dump(out/"asof.json",{"group":group,"slot":slot,"pcc":site,"inputs":x,"state":state,
         "last_local_measurement_close_s":state["as_of_s"],"master_last_received_slot":slot-2})
    nominal=[sum(a) for a in cfg["actions_battery_pv_kw"]]
    paths={}; limits=Counter(); first={}
    for cap in study["ceiling_kw"]:
        with feeder(deck,reference,cfg,site) as f:
            rows=run_policy(f,state,nominal,cap,cfg,protocol,stop=13)
        if len(rows)!=13:raise ValueError("incomplete safe policy")
        with feeder(deck,reference,cfg,site) as f:
            start(f,state,TOL)
            for r in rows:
                a=f.execute(r["actual"]["action"])
                compare_step(a,r["actual"],TOL)
                for device,reading in a["terminals"].items():
                    kmax=cfg["pv"]["kva"] if device==cfg["pv"]["device"] else cfg["storage"]["kw"]
                    if not all(math.isfinite(q) for q in reading.values()) or reading["s_kva"]>kmax+1e-5:
                        raise ValueError("independent terminal limit")
                if cap==0 and a["action"]!=0:raise ValueError("active same-state baseline")
                limits.update(r["target_veto"])
                if str(cap) not in first and r["target_veto"]:
                    first[str(cap)]={"step":r["step"],"reasons":r["target_veto"]}
        paths[cap]=rows
        dump(out/f"ceiling-{cap}.json",rows)
    net=signed_paths(paths)
    capacity={}
    for h in study["hold_s"]:
        target,k,cap,floor=select(net,h,rule)
        capacity[str(h)]={"kw":target,"reach_s":k,"ceiling_kw":cap,"min_kw":floor,
                          "hold_interval_s":[state["as_of_s"]+k,state["as_of_s"]+k+h]}
    long=capacity["10"]; k=long["reach_s"]
    for h in (2,5):
        floor=long["kw"]+rule["power_guard_kw"] if long["kw"] else -rule["power_guard_kw"]
        if min(net[long["ceiling_kw"]][k-1:k+h])+1e-8 < floor or capacity[str(h)]["kw"]<long["kw"]:
            raise ValueError("long witness shorter prefix does not support its declared target")
    queries=[]
    for req in study["request_kw"]:
        times={}; witnesses={}
        for h in study["hold_s"]:
            times[str(h)],witnesses[str(h)]=query_witness(net,req,h,rule)
        present=[h for h in study["hold_s"] if times[str(h)] is not None]
        if present!=study["hold_s"][:len(present)]:raise ValueError("inconsistent duration evidence")
        queries.append({"request_kw":req,"confirmed_reach_s":times,"witness_ceiling_kw":witnesses,
                        "certified_duration_s":max(present,default=0)})
    extra={"capacity":capacity,"queries":queries,"rejected_targets":dict(limits),"first_rejection_by_ceiling":first,
           "voltage_range_pu":[min(min(r["actual"]["voltage"].values()) for rows in paths.values() for r in rows),
                               max(max(r["actual"]["voltage"].values()) for rows in paths.values() for r in rows)],
           "min_soc_pu":min(min(r["actual"]["soc_after"]) for rows in paths.values() for r in rows),
           "actual_accepted_violations":0,"independent_witness_paths":len(paths)}
    dump(out/"targets.json",extra)
    return {"group":group,"site":site,"slot":slot,"inputs":x,"asof_sha256":sha(out/"asof.json"),
            "targets_sha256":sha(out/"targets.json"),**extra}


def collect(args):
    cfg,split=condition(args.group); study=read(PROTOCOL);rule=read(RULE)
    raw=args.sources/(args.group+".jsonl");trial=load(raw);m=trial["header"]
    if (m["git"]!=SOURCE_GIT or m["reference_sha256"]!=sha(args.reference) or m["group"]!=args.group
            or m["split"]!=split or m["seed"]!=study["native_seed"] or m["grouped_sha256"]!=sha(PROTOCOL)
            or m["local_exogenous"]!=cfg["local_exogenous"] or m["initial_soc"]!=cfg["storage"]["initial_soc"]):
        raise ValueError("mismatched native source provenance")
    signals_pq=physical(raw); _,freq=signals(raw)
    aligned=timeline(trial,signals_pq,freq,cfg,TOL["time_s"]);check_samples(signals_pq,aligned,TOL["time_s"])
    if len(aligned)!=24:raise ValueError("incomplete alignment")
    ref=ObservedReference.read_report(args.reference.read_bytes())
    gate={"tolerances":TOL,"horizons_s":study["hold_s"]}
    rows=[]
    with tempfile.TemporaryDirectory() as t:
        deck=native.archived_deck(ROOT,Path(t))
        for slot in study["asof_slots"]:
            for site in study["sites"]:
                dest=args.out/(str(slot)+"-"+site);dest.mkdir()
                rows.append(state_cases(deck,ref,args.group,site,slot,cfg,trial["frames"],dest,gate,rule))
    dump(args.out/"group.json",{"group":args.group,"split":split,"source_id":m["source_id"],"source_sha256":sha(raw),
         "condition_sha256":digest([m["local_exogenous"],m["initial_soc"]]),"protocol_sha256":sha(PROTOCOL),
         "rows":rows,"native_elapsed_s":trial["result"]["elapsed_s"],"labeler_sha256":sha(Path(__file__)),
         "source_git":m["git"]})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("group","sources","reference","out"):p.add_argument("--"+key,type=str if key=="group" else Path,required=True)
    a=p.parse_args();os.umask(0o077);a.out.mkdir(parents=True,exist_ok=False)
    try:collect(a)
    except Exception as exc:
        dump(a.out/"failed.json",{"reason":repr(exc)})
        raise
