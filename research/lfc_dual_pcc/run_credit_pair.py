"""Independent 60 Hz matched-credit holdout under a fixed external service."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .clock_graph import ClockGraphPolicy, Coordinator, slow_input
from .driver import dump, gates, report, setup, sha, state_digest
from .equal_service import split_candidates
from .metrics import measure
from .revision_protocol import HOLDOUT, HOLDOUT_SEED, demand
from .run_equal_service import ledger
from .stage0 import CONFIG as WIRING


class CreditServiceCoordinator(Coordinator):
    def master_allocation(self,error,reports,limits,slot,now,freq_time,sequences):
        requested=demand(slot)
        choices,mask=split_candidates(requested,limits)
        x=slow_input(error,self.cfg["frequency_scale_hz"],reports,now)
        # Preserve the labels of the behavior head; gain labels share the same common R.
        allowed=[True]*9 if sum(mask)>1 else [a==4 for a in range(9)]
        action=self.choose("master",(x,),allowed,slot,slot,now,reports=sequences,x=x.tolist(),
                           frequency_sample_s=freq_time,distinct_commands=sum(mask))
        return choices[action//3],requested


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm",choices=("neutral","physical"))
    for v in ("out","audit","checkpoint","stage0","reference","mechanism"):
        p.add_argument("--"+v,type=Path,required=True)
    args=p.parse_args()
    if subprocess.check_output(["git","status","--porcelain"]):
        raise ValueError("requires a clean native checkout")
    cfg=dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    args.dry_run=False
    gates(args,cfg)
    audit=json.loads(args.audit.read_bytes())
    paired=audit["result"]["arms"]
    origin=audit["result"]["command_slot"]
    if (origin!=26 or "uncertified" in audit["result"]["nesting"]
            or audit["frequency_hz"]!=60
            or abs(paired["neutral"]["total_discounted_penalty_kw_s"]-
                   paired["physical"]["total_discounted_penalty_kw_s"])>1e-8):
        raise ValueError("missing the 26 s matched-credit evidence")
    base=audit["source_key"]
    expected=dict(config=sha(CONFIG),wiring=sha(WIRING),reference=sha(args.reference),mechanism=sha(args.mechanism))
    if any(base[k]!=v for k,v in expected.items()):
        raise ValueError("credit source physical provenance mismatch")
    source=dict(key=base,raw_sha256=audit["source_raw_sha256"],command_slot=origin)
    snapshot=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    h=sha(args.checkpoint)
    if (snapshot["source"]!=source or snapshot["arm"]!=args.arm
            or snapshot["behavior_digest"]!=audit["behavior_digest"]
            or h!=paired[args.arm]["checkpoint_sha256"]):
        raise ValueError("checkpoint does not belong to the requested matched update")
    setup(HOLDOUT_SEED)
    model=ClockGraphPolicy()
    model.load_state_dict(snapshot["weights"])
    before=state_digest(model)
    if before!=snapshot["digest"] or before!=paired[args.arm]["after"]:
        raise ValueError("matched update did not load exactly")
    control=CreditServiceCoordinator(model,cfg,True)
    args.out.mkdir(parents=True,exist_ok=False)
    key=dict(**expected,frequency_hz=60,credit_audit=sha(args.audit),
             git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    raw=args.out/"raw.jsonl"
    with report(raw) as emit,torch.inference_mode():
        emit(dict(phase="manifest",arm=args.arm,argv=sys.argv,checkpoint_sha256=h,key=key,
                  event=HOLDOUT,seed=HOLDOUT_SEED,before=before))
        _,_,frames,trace,sensors,ref=native_episode(model,cfg,"clock_graph",HOLDOUT,args.reference,
                                                   HOLDOUT_SEED,True,emit,coordinator=control)
        if abs(trace[0][0])>1e-8 or abs(trace[0][1]-60.)>.01 or state_digest(model)!=before:
            raise ValueError("native initialization or frozen evaluator mismatch")
        steps=ledger(frames,sensors,demand)
        metrics=measure(frames,trace,sensors,ref,HOLDOUT)
        for name in ("unmet","excess"):
            metrics[name+"_external_kwh"]=sum(r[name+"_kw"]*r["duration_s"] for r in steps)/3600.
        result=dict(arm=args.arm,key=key,event=HOLDOUT,seed=HOLDOUT_SEED,checkpoint_sha256=h,
                    before=before,after=state_digest(model),metrics=metrics,native_steps=steps,formal_50hz=False)
        emit(dict(phase="complete",**result))
    dump(args.out/"summary.json",dict(raw_sha256=sha(raw),**result))


if __name__=="__main__":
    main()
