"""Bounded native comparison of service rewards on the same explored split labels."""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import torch

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .clock_graph import tensor, update
from .driver import dump, gates, report, setup, sha, state_digest
from .equal_service import SERVICE_SLOTS, TRACKING_WEIGHT, freeze_model
from .metrics import measure
from .precheck_residual import precheck_residual
from .revision_protocol import REVISED_KW, demand
from .run_equal_service import ledger
from .service_tradeoff import (HOLDOUT, HOLDOUT_SEED, IAE_MARGIN_HZ_S, MIN_EXCESS_REDUCTION_KWH, NADIR_MARGIN_HZ,
                               PEAK_UNMET_KW, PROBES, TOLERANCE_KW, UNMET_BUDGET_KWH,
                               TradeoffCoordinator, TradeoffPolicy, allowed_splits,
                               compare_risk, summarize_frontier)
from .stage0 import CONFIG as WIRING


def declaration(cfg):
    return dict(service_kw=REVISED_KW, service_slots=list(SERVICE_SLOTS),
                probes=list(PROBES), tolerance_kw=TOLERANCE_KW,
                peak_unmet_kw=PEAK_UNMET_KW, unmet_budget_kwh=UNMET_BUDGET_KWH,
                emergency_abs_hz=2*cfg["frequency_scale_hz"],
                development=cfg["events"]["development"], development_seed=cfg["study"]["seed"],
                holdout=HOLDOUT, holdout_seed=HOLDOUT_SEED, iae_margin_hz_s=IAE_MARGIN_HZ_S,
                nadir_margin_hz=NADIR_MARGIN_HZ, excess_to_shortfall_weight=1.,
                minimum_excess_reduction_kwh=MIN_EXCESS_REDUCTION_KWH,
                frequency_scale_hz=cfg["frequency_scale_hz"])


def get_checkpoint(folder, arm, model, key):
    audit = json.loads((folder/"summary.json").read_bytes())
    checkpoint = folder/(arm+".pt")
    statement = torch.load(checkpoint,map_location="cpu",weights_only=True)
    expected = audit["updates"][arm]
    if (audit["key"] != key or audit["before"] == expected["after"]
            or statement["key"] != key or expected["sha256"] != sha(checkpoint)
            or statement["arm"] != arm or statement["digest"] != expected["after"]):
        raise ValueError("incompatible fitted trade-off checkpoint")
    lower = model.lower_digest()
    model.load_state_dict(statement["weights"])
    if state_digest(model) != statement["digest"] or model.lower_digest() != lower:
        raise ValueError("unexpected checkpoint state after restoration")
    return expected


def cleared_modes(model, args, key, proposal, cfg):
    precheck = proposal["precheck"]
    rows = []
    scores = {}
    for arm in ("old","revised"):
        get_checkpoint(args.training,arm,model,key)
        head = model.heads["master"]
        scores[arm] = []
        for w in precheck["windows"]:
            mask = allowed_splits(w["allocatable_kw"],*w["interval_kw"],w["reports"],
                                  w["frequency_error_hz"],cfg["frequency_scale_hz"])
            action, _ = head.decision((tensor(w["observation"]),),mask,True)
            scores[arm].append(action)
    for k,w in enumerate(precheck["windows"]):
        actions = {arm:scores[arm][k] for arm in scores}
        mask = allowed_splits(w["allocatable_kw"],*w["interval_kw"],w["reports"],
                              w["frequency_error_hz"],cfg["frequency_scale_hz"])
        rule = proposal["frontier"]["rule_action"]
        actions["rule"] = rule if mask[rule] else 0
        joined = {arm: next(t for t in w["alternatives"] if t["latent"]==PROBES[i])
                  for arm, i in actions.items()}
        def distinguish(a,b):
            this,other = joined[a],joined[b]
            return any(abs(p[s][0]-q[s][0])>=cfg["gates"]["power_abs_kw"]
                       for p,q in zip(this["pcc_kw_kvar"],other["pcc_kw_kvar"],strict=True) for s in p)
        rows.append(dict(slot=w["slot"], labels=actions, same_observation=True,
                         native_modes={k:t["actions"] for k,t in joined.items()},
                         delivery_kw={k:t["delivered_kw"] for k,t in joined.items()},
                         revised_differs_from_old=distinguish("revised","old"),
                         rule_differs_from_center=actions["rule"]!=0))
    passed = any(w["revised_differs_from_old"] for w in rows)
    return dict(key=key, passed=passed, windows=rows,
                scope="actual candidate trajectories joined on the exact frozen development observation and command")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("frontier","train","clearance","risk","old","revised","rule"))
    for v in ("out","stage0","reference","mechanism","lower"):
        p.add_argument("--"+v,type=Path,required=True)
    for v in ("frontier","frozen-trace","training","clearance","risk","dev-old","dev-revised"):
        p.add_argument("--"+v,type=Path)
    p.add_argument("--development",action="store_true")
    args=p.parse_args()
    if subprocess.check_output(["git","status","--porcelain"]):
        raise ValueError("requires clean native checkout")
    cfg = dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    args.dry_run=False
    gates(args,cfg)
    key = dict(git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               config=sha(CONFIG), wiring=sha(WIRING), reference=sha(args.reference),
               mechanism=sha(args.mechanism), stage0=sha(args.stage0/"summary.json"),
               lower=sha(args.lower), frequency_hz=60, track_weight=TRACKING_WEIGHT,
               versions={k:version(k) for k in ("dss-python","torch","matlabengine","numpy")},
               protocol=declaration(cfg))
    dev = args.mode in ("frontier","train","clearance","risk") or args.development
    seed = cfg["study"]["seed"] if dev else HOLDOUT_SEED
    event = cfg["events"]["development"] if dev else HOLDOUT
    setup(seed)
    model = freeze_model(args.lower,key,TradeoffPolicy)
    lower = model.lower_digest()
    if args.mode == "frontier":
        with torch.no_grad():
            precheck=precheck_residual(args,cfg,key,model,capture_delivery=True)
        evidence=summarize_frontier(precheck)
        dump(args.out,dict(key=key,lower_digest=lower,frontier=evidence,precheck=precheck))
        if not evidence["passed"]:
            raise ValueError("native frontier did not distinguish the causal rule from the center")
        return
    proposal = json.loads(args.frontier.read_bytes())
    if proposal["key"] != key or proposal["lower_digest"] != lower or not proposal["frontier"]["passed"]:
        raise ValueError("incompatible or failed preregistered frontier")
    key["frontier"] = sha(args.frontier)
    if args.mode == "clearance":
        with torch.no_grad():
            proof = cleared_modes(model,args,key,proposal,cfg)
        dump(args.out,proof)
        if not proof["passed"]:
            raise ValueError("the fitted greedy actions select the same native device modes")
        return
    train = args.mode == "train"
    if not train:
        clearance = json.loads(args.clearance.read_bytes())
        if clearance["key"] != key or not clearance["passed"]:
            raise ValueError("missing physical greedy clearance on development")
        if args.mode == "risk":
            a,b = [json.loads(f.read_bytes()) for f in (args.dev_old,args.dev_revised)]
            if (a["mode"]!="old" or b["mode"]!="revised"
                    or not all(v["key"] == key and v["event"] == event and v["seed"] == seed for v in (a,b))):
                raise ValueError("development risk comparison changed the domain")
            risk=compare_risk(a,b)
            dump(args.out,dict(key=key,clearance=sha(args.clearance),**risk))
            if not risk["passed"]:
                raise ValueError("development ran outside the registered service or frequency limits")
            return
        if not args.development:
            risk = json.loads(args.risk.read_bytes())
            if risk["key"] != key or risk["clearance"] != sha(args.clearance) or not risk["passed"]:
                raise ValueError("missing prospective development frequency clearance")
        arm = "old" if args.mode=="rule" else args.mode
        trained = get_checkpoint(args.training,arm,model,key)
    else:
        trained = None
    rule = proposal["frontier"]["rule_action"] if args.mode=="rule" else None
    control = TradeoffCoordinator(model,cfg,not train,rule_action=rule)
    before = state_digest(model)
    args.out.mkdir(parents=True,exist_ok=False)
    raw = args.out/"raw.jsonl"
    behavior = []
    with report(raw) as emit:
        def monitor(row):
            if row["phase"]=="policy" and row["role"]=="master":
                behavior.append(row)
            emit(row)
        emit(dict(phase="manifest",key=key,mode=args.mode,seed=seed,event=event,argv=sys.argv,
                  before=before,lower_digest=lower,scope="60 Hz engineering prototype",greedy=not train))
        with torch.no_grad():
            _,_,frames,trace,sensors,ref=native_episode(model,cfg,"clock_graph",event,args.reference,
                                                       seed,not train,monitor,coordinator=control)
        if abs(trace[0][0])>1e-8 or abs(trace[0][1]-60.)>.01 or len(behavior)!=len(cfg["window_slots"]):
            raise ValueError("incomplete joint rollout or invalid initialization")
        if any(f["actual"][s]["veto"] for f in frames for s in ("A","B")):
            raise ValueError("a completed action hit a native physical veto")
        if state_digest(model)!=before or model.lower_digest()!=lower:
            raise ValueError("frozen behavior model changed before optimization")
        steps = ledger(frames,sensors,demand)
        metrics = measure(frames,trace,sensors,ref,event)
        for tag in ("unmet","excess"):
            metrics[tag+"_external_kwh"]=sum(s[tag+"_kw"]*s["duration_s"] for s in steps)/3600.
        metrics["peak_unmet_external_kw"]=max(s["unmet_kw"] for s in steps)
        updates = {}
        if train:
            selected = {v["action"] for v in behavior if v["mask"][1]}
            realized = {tuple(f["actions"][s] for s in ("A","B")) for f in frames
                        if f["slot"]-1-(f["slot"]-1)%2 in SERVICE_SLOTS}
            if len(selected)<2 or len(realized)<2:
                raise ValueError("common behavior did not explore different executable native modes")
            for arm in ("old","revised"):
                clone = copy.deepcopy(model)
                control.penalize_excess = arm=="revised"
                emit(dict(phase="objective", arm=arm, excess_weight=float(control.penalize_excess)))
                groups = control.returns(frames,trace,ref)
                opt = update(clone,groups,cfg["study"]["learning_rate"])
                digest = state_digest(clone)
                file = args.out/(arm+".pt")
                with file.open("xb") as f:
                    torch.save(dict(weights=clone.state_dict(),key=key,arm=arm,digest=digest),f)
                restored = torch.load(file,map_location="cpu",weights_only=True)
                clone.load_state_dict(restored["weights"])
                if state_digest(clone)!=digest or digest==before or clone.lower_digest()!=lower:
                    raise ValueError("the matched checkpoint did not update or restore exactly")
                updates[arm] = dict(after=digest,sha256=sha(file),optimizer=opt)
        result=dict(key=key,mode=args.mode,seed=seed,event=event,greedy=not train,before=before,
                    lower_digest=lower,trained=trained,updates=updates,metrics=metrics,decisions=behavior,
                    native_steps=steps,formal_50hz=False)
        emit(dict(phase="complete",**result))
    dump(args.out/"summary.json",dict(raw_sha256=sha(raw),**result))


if __name__=="__main__":
    main()
