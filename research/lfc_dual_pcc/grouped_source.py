"""Independent reset Case A trajectories for a grouped capability study."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .driver import report, setup, sha
from .handoff import HANDOFF, Release
from .hierarchy import Hierarchy
from .score_stage0 import score

PROTOCOL=Path(__file__).with_name("grouped_protocol.json")


def read(path):
    return json.loads(Path(path).read_bytes())


def condition(group):
    study=read(PROTOCOL)
    membership=[s for s,groups in study["groups"].items() if group in groups]
    if len(membership)!=1: raise ValueError("group must belong to exactly one split")
    levels=study["levels"]
    a,b,c=group
    cfg=read(CONFIG)
    cfg["storage"]["initial_soc"]={"A":levels["soc_a_pu"][a],"B":levels["soc_b_pu"][a]}
    cfg["local_exogenous"]={"load_mult":levels["load_mult"][b],"irradiance":levels["irradiance"][c]}
    return cfg,membership[0]


def collect(args):
    cfg,split=condition(args.group)
    study=read(PROTOCOL)
    if not score(args.stage0)["stage0_passed"]: raise ValueError("invalid original two-PCC gate")
    old=read(args.parent_gate)
    if not old["passed"] or old["gate_git"]!="89a627d861e997db2fa1e49ab830ceee6bfa481a" or old["reference_sha256"]!=sha(args.reference):
        raise ValueError("unverified original native parent")
    setup(study["native_seed"])
    source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record=args.out/(args.group+".jsonl")
    with report(record) as emit:
        emit({"phase":"manifest","mode":"native","git":source,"group":args.group,"split":split,
              "source_id":"independent-reset-"+args.group,"seed":study["native_seed"],"grouped_sha256":sha(PROTOCOL),
              "reference_sha256":sha(args.reference),"parent_gate_sha256":sha(args.parent_gate),
              "local_exogenous":cfg["local_exogenous"],"initial_soc":cfg["storage"]["initial_soc"]})
        start=time.monotonic()
        _,_,frames,_,_,_=native_episode(Hierarchy(cfg),cfg,"fixed",cfg["events"]["evaluation"],args.reference,
            study["native_seed"],True,emit,handoff=Release("delivered",read(HANDOFF)))
        if len(frames)!=24: raise ValueError("incomplete source")
        emit({"phase":"complete","frames":len(frames),"elapsed_s":time.monotonic()-start})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("group","stage0","parent-gate","reference","out"):
        p.add_argument("--"+k,type=str if k=="group" else Path,required=True)
    a=p.parse_args(); os.umask(0o077); a.out.mkdir(parents=True,exist_ok=True)
    collect(a)
