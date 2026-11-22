"""Paired 60 Hz physical trial with a common commanded prefix and finite horizon witnesses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .driver import dump, gates, report, setup, sha
from .master_dispatch import SITES, feasible_actions
from .metrics import measure


class HorizonController:
    def __init__(self, cfg, prefix, plan, start):
        self.cfg,self.prefix,self.plan,self.start = cfg,prefix,plan,start

    def initialize(self, plants, emit):
        self.emit=emit

    def dispatch(self, slot, now, freq_time, error, plants):
        if not freq_time<now:
            raise ValueError("noncausal native observation")
        if slot<self.start:
            return self.prefix[slot]["request"],self.prefix[slot]["actions"]
        active = slot<self.start+5
        request = self.plan["request_kw"] if active else dict(A=0.,B=0.)
        table = self.cfg["actions_battery_pv_kw"]
        actions, fallback = {},{}
        for s,p in plants.items():
            rows = p.candidates()
            eligible = feasible_actions(table,rows,request[s])
            selected = self.plan["action"][s] if active else None
            if selected not in eligible:
                selected = min(eligible,key=lambda i:(abs(p.base[0]-rows[i]["pcc"][0]-request[s]),i))
                if active:
                    fallback[s] = selected
            actions[s]=selected
        self.emit(dict(phase="horizon_dispatch",slot=slot,external_kw=10. if active else 0.,
                       request_kw=request,actions=actions,fallback=fallback))
        return request,actions

    def receive(self, f, plants):
        if f["slot"]>=self.start:
            return
        for s,old in self.prefix[f["slot"]]["actual"].items():
            got = f["actual"][s]
            if (any(abs(a-b)>1e-8 for a,b in zip(old["soc_after"],got["soc_after"],strict=True))
                    or any(abs(a-b)>self.cfg["gates"]["parity_kw"] for a,b in zip(old["pcc"],got["pcc"],strict=True))):
                raise ValueError("native initial prefix no longer matches the capability witness")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("horizon",choices=("1","5"))
    for name in ("out","witness","training","reference","stage0","mechanism"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if subprocess.check_output(["git","status","--porcelain"]):
        raise ValueError("requires clean native checkout")
    cfg = dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    args.dry_run=False
    gates(args,cfg)
    witness=json.loads(args.witness.read_bytes())
    original=args.training/"raw.jsonl"
    summary=json.loads((args.training/"summary.json").read_bytes())
    if (witness["mode"]!="horizon" or witness["source_raw_sha256"]!=sha(original)
            or witness["source_raw_sha256"]!=summary["raw_sha256"] or witness["source_key"]!=summary["key"]
            or witness["source_key"]["reference"]!=sha(args.reference)
            or witness["source_key"]["config"]!=sha(CONFIG) or witness["frequency_hz"]!=60):
        raise ValueError("mismatched native capability witness")
    result=witness["result"]
    start=result["first_effective_fast_slot"]
    if start!=25:
        raise ValueError("unregistered intervention time")
    seed,event=summary["seed"],summary["event"]
    setup(seed)
    prefix = {r["slot"]:r for r in map(json.loads,original.read_text().splitlines())
              if r["phase"]=="hold" and r["slot"]<start}
    plan=result["planning"][args.horizon]
    control=HorizonController(cfg,prefix,plan,start)
    args.out.mkdir(parents=True,exist_ok=False)
    raw=args.out/"raw.jsonl"
    key=dict(git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             source_raw=sha(original),witness=sha(args.witness),reference=sha(args.reference),
             config=sha(CONFIG),mechanism=sha(args.mechanism),frequency_hz=60)
    with report(raw) as emit:
        emit(dict(phase="manifest",argv=sys.argv,horizon=int(args.horizon),seed=seed,event=event,key=key,
                  policy="scheduled prefix then held native one- or five-second witnessed plan"))
        _,_,frames,trace,sensors,ref=native_episode(None,cfg,"clock_graph",event,args.reference,
                                                   seed,True,emit,coordinator=control)
        if abs(trace[0][0])>1e-8 or abs(trace[0][1]-60.)>.01:
            raise ValueError("unexpected native initialization frequency")
        metrics=measure(frames,trace,sensors,ref,event)
        physical={r["used_slot"]:r for r in frames if r["used_slot"] is not None}
        steps=[]
        for r in frames:
            if r["slot"] not in range(start,start+5):
                continue
            f=physical[r["slot"]]
            dt=f["end_s"]-f["write_s"]
            signed=10.-sum(r["delivered"].values())
            steps.append(dict(slot=r["slot"],physical_slot=f["slot"],duration_s=dt,request_kw=r["request"],
                              external_kw=10.,action=r["actions"],delivered_kw=r["delivered"],
                              pcc_kw_kvar=r["pcc"],signed_gap_kw=signed))
        metrics["unmet_external_kwh"]=sum(max(0.,r["signed_gap_kw"])*r["duration_s"] for r in steps)/3600
        metrics["excess_external_kwh"]=sum(max(0.,-r["signed_gap_kw"])*r["duration_s"] for r in steps)/3600
        for name,slot in (("handoff",start),("post_service",start+5)):
            pivot=physical[slot]["write_s"]
            hz=[v for t,v in trace if pivot<t<=pivot+4.]
            metrics[name+"_swing_hz"]=max(hz)-min(hz)
        audit=dict(key=key,seed=seed,event=event,horizon=int(args.horizon),steps=steps,metrics=metrics,formal_50hz=False)
        emit(dict(phase="complete",**audit))
    dump(args.out/"summary.json",dict(raw_sha256=sha(raw),**audit))


if __name__=="__main__":
    main()
