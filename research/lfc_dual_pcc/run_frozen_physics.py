"""Run small physical-credit or sustained-capability gates on a frozen native episode."""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile

import torch

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .case_a_plant import CONFIG, certify, feeder
from .clock_graph import ClockGraphPolicy, tensor
from .driver import dump, setup, sha, state_digest
from .equal_service import LOWER_SOURCE
from .horizon_witness import repeated_witnesses
from .phase_graph import PhaseGraph
from .physical_credit import credit_budget, matched_update, nested_bound


SLOT = 24
SITES = ("A","B")


def credit(rows, holds, deck, ref, cfg, model, origin, nested):
    graphs = {}
    frozen_graph = next(r["graphs"] for r in rows if r["phase"]=="graphs")
    for s in SITES:
        with feeder(deck,ref,cfg,s) as plant:
            graphs[s] = PhaseGraph(plant)
        if graphs[s].summary != frozen_graph[s]:
            raise ValueError("different physical graph during behavior replay")
    inputs = {(r["slot"],r["site"]):r for r in rows if r["phase"]=="device_input"}
    selected = {}
    count = 0
    with torch.no_grad():
        for r in rows:
            if r["phase"]!="policy":
                continue
            role,k = r["role"],r["slot"]
            if "x" in r:
                args = (tensor(r["x"]),)
            else:
                d,g = inputs[k,role[-1]],graphs[role[-1]]
                args = (tensor(d["x"]),g.adj,g.pools[int(role.startswith("pv"))],tensor(d["context"]))
            idx, record = model.heads[role].decision(args,r["mask"],False,logit_offsets=r.get("logit_offsets"))
            if idx != r["action"]:
                raise ValueError("behavior policy sampling could not be recovered from frozen seed")
            count += 1
            if r["command"]==origin:
                selected[role,k] = record
    if count !=104 or len(selected)!=13:
        raise ValueError("wrong number of recovered behavior decisions")
    sites = {}
    for s in SITES:
        previous = holds[origin]["actual"][s]
        initial = dict(pcc=previous["pcc"],voltage=previous["voltage"],soc=previous["soc_after"])
        prefix = [r["actions"][s] for k,r in sorted(holds.items()) if k<=origin]
        actions = [holds[k]["actions"][s] for k in (origin+1,origin+2)]
        pcc = [holds[k]["delivered"][s] for k in (origin+1,origin+2)]
        req = holds[origin+1]["request"][s]
        if req != holds[origin+2]["request"][s]:
            raise ValueError("request was not held over both fast steps")
        try:
            certificate = certify(deck,ref,cfg,s,prefix,initial,actions,collect_paths=nested)
            if any(abs(x-y)>cfg["gates"]["parity_kw"] for x,y in zip(pcc,certificate["actual_action_replay_kw"],strict=True)):
                raise ValueError("native alternative reconstruction differs")
            bound = certificate["capacity_kw"]
        except ValueError as exc:
            certificate,bound = dict(uncertified=str(exc)),None
        sites[s] = dict(request_kw=req, sustainable_kw=min(pcc), capacity_kw=bound,
                        actions=actions, certificate=certificate)
    nesting = None
    if nested:
        command = next(r for r in rows if r["phase"]=="dispatch" and r["sequence"]==origin)
        try:
            if any(v["capacity_kw"] is None for v in sites.values()):
                raise ValueError("one or more sites lack a common native certificate")
            nesting = nested_bound(sites,command["bid_limits_kw"],cfg["gates"]["parity_kw"])
            for gate in ("fixed","relaxed"):
                for s,w in nesting[gate]["witnesses"].items():
                    with feeder(deck,ref,cfg,s) as plant:
                        for k,row in sorted(holds.items()):
                            if k<=origin:
                                plant.execute(row["actions"][s])
                        support = min(plant.execute(i)["delivered_kw"] for i in w["actions"])
                    if abs(support-w["support_kw"])>cfg["gates"]["parity_kw"]:
                        raise ValueError("conditional winner did not reproduce")
        except ValueError as exc:
            nesting=dict(uncertified=str(exc))
            for s in sites:
                sites[s]["capacity_kw"]=None
    physical = {r["used_slot"]:r for r in holds.values() if r["used_slot"] is not None}
    freqs = next(r for r in rows if r["phase"]=="native_frequency")
    start = physical[origin+1]["write_s"]
    weights, rewards = {}, {}
    for k in (origin+1,origin+2):
        frame = physical[k]
        lo,hi = frame["write_s"],frame["end_s"]
        vals = [abs(v-freqs["reference_hz"]) for t,v in freqs["samples"] if lo<t<hi]
        weights[k] = .95**(lo-start)*(hi-lo)
        rewards[k] = -sum(vals)/len(vals)/cfg["frequency_scale_hz"]*weights[k]
    exposures = {key:(sum(weights.values()) if key[0]=="master" else weights[key[1]]) for key in selected}
    returns = {key:(sum(rewards.values()) if key[0]=="master" else rewards[key[1]]) for key in selected}
    arms = {}
    for name in ("neutral","physical"):
        allocation,ledger = credit_budget(sites,exposures,attributed=name=="physical")
        clone = copy.deepcopy(model)
        res = matched_update(clone,selected,returns,allocation,cfg["study"]["learning_rate"])
        if any(abs(r["ratio"]-1.)>1e-7 for r in res["examples"]):
            raise ValueError("behavior probabilities differ before the matched update")
        arms[name] = dict(total_discounted_penalty_kw_s=sum(allocation.values()),credit=ledger,
                          optimizer=res,after=state_digest(clone))
    if not math.isclose(arms["neutral"]["total_discounted_penalty_kw_s"],arms["physical"]["total_discounted_penalty_kw_s"],abs_tol=1e-8):
        raise ValueError("different matched budgets")
    return dict(command_slot=origin, recovered_decisions=count, behavior_samples=len(selected),
                consumed_slots={k:physical[k]["slot"] for k in weights}, sites=sites,
                gamma_per_second=.95,nesting=nesting,arms=arms)


def horizon(holds, deck, ref, cfg):
    evidence = {}
    @contextmanager
    def restore(site):
        with feeder(deck,ref,cfg,site) as plant:
            for k, row in sorted(holds.items()):
                if k>SLOT:
                    break
                native_row = plant.execute(row["actions"][site])
                original = row["actual"][site]
                if (any(abs(a-b)>1e-8 for a,b in zip(native_row["soc_after"],original["soc_after"],strict=True))
                        or any(abs(a-b)>cfg["gates"]["parity_kw"] for a,b in zip(native_row["pcc"],original["pcc"],strict=True))):
                    raise ValueError("reconstructed prefix differs")
            yield plant
    for site in SITES:
        result,paths = repeated_witnesses(lambda:restore(site),len(cfg["actions_battery_pv_kw"]))
        one = result[1]
        evidence[site] = dict(horizons=result,verified_paths=paths,
                              one_step_action_holds=len(paths[one["action"]]) if one["confirmed"] else 0)
    planning = {}
    for n in (1,5):
        caps = {s:evidence[s]["horizons"][n]["capacity_kw"] for s in SITES}
        if any(v is None for v in caps.values()):
            planning[n] = dict(status="missing sustained witness")
            continue
        total = min(10.,sum(caps.values()))
        a = min(caps["A"],max(0.,total-caps["B"],.8*total))
        request = dict(A=a,B=total-a)
        choices = {
            s:min((min(v[:n])-request[s],i) for i,v in evidence[s]["verified_paths"].items()
                  if len(v)>=n and min(v[:n])>=request[s]-1e-7)[1] for s in SITES
        }
        steps = []
        with ExitStack() as stack:
            plants = {s:stack.enter_context(restore(s)) for s in SITES}
            for k in range(5):
                delivered, replaced = {}, {}
                for s,p in plants.items():
                    idx = choices[s]
                    if p.probe(idx)["veto"]:
                        rows = p.candidates()
                        idx = min((abs(p.base[0]-r["pcc"][0]-request[s]),i)
                                  for i,r in enumerate(rows) if not r["veto"])[1]
                        replaced[s] = idx
                    delivered[s] = p.execute(idx)["delivered_kw"]
                steps.append(dict(slot=SLOT+1+k,delivered_kw=delivered,fallback=replaced,
                                  unmet_kw=max(0.,10.-sum(delivered.values()))))
        planning[n] = dict(request_kw=request,action=choices,steps=steps,external_kw=10.,
                           unmet_kw_s=sum(r["unmet_kw"] for r in steps))
    return dict(command_slot=SLOT,first_effective_fast_slot=SLOT+1,evidence=evidence,planning=planning,
                scope="conditional on replayed known prefix and fixed feeder boundary")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("credit","horizon"))
    p.add_argument("training",type=Path)
    p.add_argument("reference",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--first-development-gap",action="store_true",help="choose the earliest measured development gap")
    p.add_argument("--nested",action="store_true",help="require fixed and relaxed conditional action domains")
    args = p.parse_args()
    summary = json.loads((args.training/"summary.json").read_bytes())
    raw = args.training/"raw.jsonl"
    key = summary["key"]
    if (summary["mode"]!="train" or summary["raw_sha256"]!=sha(raw) or key["git"]!=LOWER_SOURCE
            or key["reference"]!=sha(args.reference) or key["config"]!=sha(CONFIG) or key["frequency_hz"]!=60):
        raise ValueError("requires the pinned native development trajectory and matching physical inputs")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if rows[-1]["phase"]!="complete" or any(r["phase"]=="failed" for r in rows):
        raise ValueError("incomplete native source")
    holds = {r["slot"]:r for r in rows if r["phase"]=="hold"}
    cfg = dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    ref = ObservedReference.read_report(args.reference.read_bytes())
    setup(summary["seed"])
    model = ClockGraphPolicy()
    if state_digest(model)!=summary["before"]:
        raise ValueError("cannot reconstruct original behavior weights")
    origin = SLOT
    if args.first_development_gap:
        if args.mode!="credit":
            p.error("only the credit gate accepts a development window selection")
        origin = next((k for k in cfg["window_slots"] if sum(
            max(0.,holds[k+1]["request"][s]-min(holds[k+i]["delivered"][s] for i in (1,2))) for s in SITES
        ) > cfg["gates"]["power_abs_kw"]),None)
        if origin is None:
            raise ValueError("no measured development shortfall above the physical threshold")
    with tempfile.TemporaryDirectory() as tmp:
        deck = native.archived_deck(ROOT,Path(tmp))
        result = credit(rows,holds,deck,ref,cfg,model,origin,args.nested) if args.mode=="credit" else horizon(holds,deck,ref,cfg)
    dump(args.out,dict(code=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                       frequency_hz=60, source_key=key, source_raw_sha256=sha(raw), seed=summary["seed"],
                       behavior_digest=summary["before"],mode=args.mode, result=result,formal_50hz=False))


if __name__=="__main__":
    main()
