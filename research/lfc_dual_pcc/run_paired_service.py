"""Replay one matched service-only update and conditionally evaluate native closure."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import torch

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .clock_graph import tensor
from .driver import dump, gates, report, setup, sha, state_digest
from .equal_service import freeze_model
from .metrics import measure
from .paired_service import (DISCOUNT_PER_SECOND, LEARNING_RATE, MIN_CHANGED_WINDOWS, SOURCE_CODE, fit_bias,
                             paired_window, post_service_swing)
from .revision_protocol import demand
from .run_equal_service import ledger
from .run_service_tradeoff import declaration, get_checkpoint
from .service_tradeoff import (HOLDOUT, HOLDOUT_SEED, TOLERANCE_KW, TradeoffCoordinator,
                               TradeoffPolicy, compare_risk)
from .stage0 import CONFIG as WIRING


def read(path):
    return json.loads(path.read_bytes())


def fit(args, cfg, model, frontier, key):
    precheck = frontier["precheck"]
    if precheck["frozen_raw_sha256"] != sha(args.source):
        raise ValueError("physical consumers must come from the pinned development source")
    trace = [json.loads(line) for line in args.source.read_text().splitlines()]
    if trace[-1]["phase"]!="complete":
        raise ValueError("incomplete original clock")
    physical = {}
    for f in trace:
        if f["phase"] == "hold" and f["used_slot"] is not None:
            if f["used_slot"] in physical:
                raise ValueError("duplicate physical consumption")
            physical[f["used_slot"]] = f
    rows = [paired_window(w,physical,cfg["frequency_scale_hz"]) for w in precheck["windows"]]
    head = model.heads["master"]
    with torch.no_grad():
        old = [head.decision((tensor(w["x"]),),w["mask"],True)[0] for w in rows]
    parameter = "heads.master.actor.layers.2.bias"
    retained = {name:t.clone() for name,t in model.state_dict().items() if name!=parameter}
    change = fit_bias(model, rows)
    if any(not torch.equal(t, retained[name]) for name,t in model.state_dict().items() if name!=parameter):
        raise ValueError("auxiliary step changed the frozen feature map")
    rule = frontier["frontier"]["rule_action"]
    with torch.no_grad():
        for row,prev in zip(rows,old,strict=True):
            action, _ = head.decision((tensor(row["x"]),),row["mask"],True)
            labels = dict(ppo=prev, paired=action, rule=rule if row["mask"][rule] else 0)
            associated = {k:row["alternatives"][key["source_key"]["protocol"]["probes"][i]]
                          for k,i in labels.items()}
            a,b = (associated[k] for k in ("ppo","paired"))
            diff = max(abs(x[s][0]-y[s][0])
                       for x,y in zip(a["pcc_kw_kvar"],b["pcc_kw_kvar"],strict=True) for s in x)
            row.update(labels=labels,pcc_signature_diff_kw=diff,
                       changed=diff>=cfg["gates"]["power_abs_kw"],
                       selected_actions={k:r["actions"] for k,r in associated.items()},
                       selected_delivery_kw={k:r["delivered_kw"] for k,r in associated.items()})
            del row["alternatives"], row["x"]
    args.out.mkdir(parents=True,exist_ok=False)
    weights = args.out/"paired.pt"
    snapshot = dict(weights=model.state_dict(),key=key,digest=state_digest(model))
    with weights.open("xb") as handle:
        torch.save(snapshot,handle)
    restore_paired(model, weights, key, snapshot["digest"], model.lower_digest())
    result = dict(key=key,optimizer=change,windows=rows,rule_action=rule,
                  lower_digest=model.lower_digest(),changed_parameter=parameter,
                  checkpoint_sha256=sha(weights),model_digest=snapshot["digest"],
                  changed_windows=sum(w["changed"] for w in rows))
    result["passed"] = result["changed_windows"]>=MIN_CHANGED_WINDOWS
    dump(args.out/"summary.json",result)
    if not result["passed"]:
        raise ValueError("too few same-state native action changes after ranking")


def restore_paired(model,path,key,digest,lower):
    snapshot=torch.load(path,map_location="cpu",weights_only=True)
    if snapshot["key"]!=key or snapshot["digest"]!=digest:
        raise ValueError("wrong paired update checkpoint")
    model.load_state_dict(snapshot["weights"])
    if state_digest(model)!=digest or model.lower_digest()!=lower:
        raise ValueError("paired checkpoint did not read back exactly")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("fit","ppo","paired","rule","risk"))
    for option in ("out","stage0","reference","mechanism","lower","previous"):
        p.add_argument("--"+option,type=Path,required=True)
    for option in ("source","fitted","risk","development-ppo","development-paired"):
        p.add_argument("--"+option,type=Path)
    p.add_argument("--development",action="store_true")
    args=p.parse_args()
    if not args.development and args.mode in ("ppo","paired","rule") and args.risk is None:
        p.error("holdout requires prospective development risk clearance")
    if subprocess.check_output(["git","status","--porcelain"]):
        raise ValueError("requires clean checkout")
    cfg=dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    args.dry_run=False
    gates(args,cfg)
    frontier=read(args.previous/"frontier.json")
    frozen=frontier["key"]
    fields=dict(git=SOURCE_CODE,config=sha(CONFIG),wiring=sha(WIRING),reference=sha(args.reference),
                mechanism=sha(args.mechanism),stage0=sha(args.stage0/"summary.json"),lower=sha(args.lower),
                frequency_hz=60, protocol=declaration(cfg),
                versions={k:version(k) for k in ("dss-python","torch","matlabengine","numpy")})
    if any(frozen[k]!=v for k,v in fields.items()) or not frontier["frontier"]["passed"]:
        raise ValueError("mismatched source experiment")
    previous=args.previous/"train"
    source_key=dict(frozen,frontier=sha(args.previous/"frontier.json"))
    prior=read(args.previous/"clearance.json")
    train=read(previous/"summary.json")
    if (prior["key"]!=source_key or prior["passed"] or sha(previous/"raw.jsonl")!=train["raw_sha256"]
            or (args.previous/"verification.exit").read_text().strip()!="0"):
        raise ValueError("source is not the recorded third-round negative comparison")
    setup(cfg["study"]["seed"])
    model=freeze_model(args.lower,frozen,TradeoffPolicy)
    baseline=get_checkpoint(previous,"revised",model,source_key)
    lower=model.lower_digest()
    key=dict(source_key=source_key,baseline_sha256=baseline["sha256"],baseline_digest=baseline["after"],
             git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             source_train_raw_sha256=train["raw_sha256"], prior_clearance_sha256=sha(args.previous/"clearance.json"),
             update="single paired service bias SGD",learning_rate=LEARNING_RATE,
             discount=DISCOUNT_PER_SECOND,minimum_pair_gap_kw=2*TOLERANCE_KW,
             minimum_changed_windows=MIN_CHANGED_WINDOWS)
    if args.mode=="fit":
        fit(args,cfg,model,frontier,key)
        return
    fitted=read(args.fitted/"summary.json")
    weights=args.fitted/"paired.pt"
    if fitted["key"]!=key or not fitted["passed"] or fitted["checkpoint_sha256"]!=sha(weights):
        raise ValueError("missing reproducible physical clearance")
    key["fit"]=sha(args.fitted/"summary.json")
    if args.mode=="risk":
        files=(args.development_ppo,args.development_paired)
        controls=[read(path) for path in files]
        expected=(baseline["after"],fitted["model_digest"])
        if any(v["key"]!=key or not v["development"] or v["mode"]!=arm
               or v["event"]!=cfg["events"]["development"] or v["seed"]!=cfg["study"]["seed"]
               or v["model_digest"]!=digest or v["lower_digest"]!=lower
               or v["raw_sha256"]!=sha(file.parent/"raw.jsonl")
               for v,arm,digest,file in zip(controls,("ppo","paired"),expected,files,strict=True)):
            raise ValueError("risk comparison requires matched development runs")
        outcome=compare_risk(*controls)
        dump(args.out,dict(key=key,**outcome))
        if not outcome["passed"]:
            raise ValueError("prospective joint development risk gate did not pass")
        return
    if not args.development:
        risk=read(args.risk)
        if risk["key"]!=key or not risk["passed"]:
            raise ValueError("missing development risk gate")
    if args.mode=="paired":
        restore_paired(model,weights,fitted["key"],fitted["model_digest"],lower)
    before=state_digest(model)
    rule=fitted["rule_action"] if args.mode=="rule" else None
    seed=cfg["study"]["seed"] if args.development else HOLDOUT_SEED
    event=cfg["events"]["development"] if args.development else HOLDOUT
    setup(seed)
    control=TradeoffCoordinator(model,cfg,True,rule_action=rule)
    args.out.mkdir(parents=True,exist_ok=False)
    raw=args.out/"raw.jsonl"
    with report(raw) as emit, torch.inference_mode():
        emit(dict(phase="manifest",argv=sys.argv,key=key,mode=args.mode,before=before,
                  lower_digest=lower,seed=seed,event=event,development=args.development))
        _,_,frames,trace,sensors,ref=native_episode(model,cfg,"clock_graph",event,args.reference,
                                                   seed,True,emit,coordinator=control)
        if abs(trace[0][0])>1e-8 or abs(trace[0][1]-60.)>.01 or state_digest(model)!=before:
            raise ValueError("native initialization failed or weights changed")
        if any(f["actual"][s]["veto"] for f in frames for s in ("A","B")):
            raise ValueError("an executed action hit a native physical veto")
        steps=ledger(frames,sensors,demand)
        stats=measure(frames,trace,sensors,ref,event)
        stats.update(post_service_swing(frames,trace))
        for kind in ("unmet","excess"):
            stats[kind+"_external_kwh"]=sum(row[kind+"_kw"]*row["duration_s"] for row in steps)/3600.
        stats["peak_unmet_external_kw"]=max(row["unmet_kw"] for row in steps)
        result=dict(key=key,mode=args.mode,seed=seed,event=event,development=args.development,
                    model_digest=before,lower_digest=lower,metrics=stats,native_steps=steps,formal_50hz=False)
        emit(dict(phase="complete",**result))
    dump(args.out/"summary.json",dict(raw_sha256=sha(raw),**result))


if __name__=="__main__":
    main()
