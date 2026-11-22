"""Gated master-only frozen-lower 60 Hz engineering experiment."""
import argparse
from contextlib import ExitStack
from functools import partial
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .case_a_plant import CONFIG, feeder
from .case_a_runtime import native_episode
from .clock_graph import Message, update
from .continuous_service import ContinuousServiceCoordinator, ContinuousServicePolicy, allocation_interval
from .driver import dump, gates, report, setup, sha, state_digest
from .equal_service import (LOWER_SOURCE, SERVICE_KW, SERVICE_SLOTS, TRACKING_WEIGHT,
                            EqualServiceCoordinator, EqualServicePolicy, freeze_model, service_kw, split_candidates)
from .master_dispatch import SITES, bid_limits
from .metrics import measure
from .stage0 import CONFIG as WIRING


def precheck(args, cfg, key, model, controller_type):
    trace_dir = args.frozen_trace
    summary = json.loads((trace_dir/"summary.json").read_bytes())
    raw = trace_dir/"raw.jsonl"
    if (summary["raw_sha256"] != sha(raw) or summary["checkpoint_sha256"] != key["lower"]
            or summary["key"]["git"] != LOWER_SOURCE or summary["key"]["reference"] != key["reference"]):
        raise ValueError("precheck requires the frozen 83d30ce1 trajectory and checkpoint")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if rows[-1]["phase"] != "complete":
        raise ValueError("incomplete precheck source")
    holds = {r["slot"]:r for r in rows if r["phase"] == "hold"}
    packets = {r["sequence"]:Message(**{k:v for k,v in r.items() if k != "phase"})
               for r in rows if r["phase"] == "telemetry"}
    slow = next(r for r in rows if r["phase"] == "policy" and r["role"] == "master" and r["slot"] == 24)
    native_command = next(r for r in rows if r["phase"] == "dispatch" and r["sequence"] == 24)
    caps = bid_limits({s:packets[slow["reports"][s]] for s in SITES}, cfg["fast_s"])
    total, lo, hi = allocation_interval(service_kw(24),caps)
    if args.continuous:
        a = [lo+u*(hi-lo) for u in (.25,.75)]
        choices = [dict(A=v,B=total-v) for v in a]
        mask = [True,hi-lo>1e-9]
    else:
        choices, mask = split_candidates(service_kw(24), caps)
    saved = next(r["graphs"] for r in rows if r["phase"] == "graphs")
    ref = ObservedReference.read_report(args.reference.read_bytes())
    observed = holds[25]
    responses = []
    with tempfile.TemporaryDirectory() as scratch:
        deck = native.archived_deck(ROOT, Path(scratch))
        for i, allocation in enumerate(choices):
            if not mask[i]:
                continue
            with ExitStack() as stack:
                plants = {s:stack.enter_context(feeder(deck, ref, cfg, s)) for s in SITES}
                control = controller_type(model, cfg, True)
                events = []
                control.initialize(plants, events.append)
                if any(control.graphs[s].summary != saved[s] for s in SITES):
                    raise ValueError("precheck graph mismatch")
                for slot in range(cfg["clock"]["first_hold_s"], 25):
                    for s,p in plants.items():
                        p.candidates() if slot >= 23 else p.probe(0)
                        applied = p.execute(holds[slot]["actions"][s])
                        for field in ("pcc", "soc", "soc_after"):
                            if any(abs(a-b)>1e-7 for a,b in zip(applied[field],holds[slot]["actual"][s][field],strict=True)):
                                raise ValueError("precheck native state replay mismatch")
                        p.candidates()
                control.local_report = packets[24]
                urgency = control.command_urgency(summary["metrics"]["reference_hz"]-holds[24]["frequency"][1])
                control.commands.append(Message(24, native_command["sampled"], native_command["arrival"],
                                                native_command["effective"], native_command["expiry"], allocation,
                                                urgency))
                sample_time, hz = observed["frequency"]
                request, actions = control.dispatch(25, observed["write_s"], sample_time,
                                                    summary["metrics"]["reference_hz"]-hz, plants)
                applied = {s:p.execute(actions[s]) for s,p in plants.items()}
                responses.append(dict(share_index=i, request=request, action=actions,
                                      pcc_kw_kvar={s:r["pcc"] for s,r in applied.items()},
                                      delivered_kw={s:r["delivered_kw"] for s,r in applied.items()},
                                      decision_masks={r["role"]:r["mask"] for r in events if r["phase"]=="policy"}))
    full = [r for r in responses if math.isclose(sum(r["request"].values()),SERVICE_KW,abs_tol=1e-9)]
    separated = len(full)>1 and max(
        max(abs(a["delivered_kw"][s]-b["delivered_kw"][s]) for s in SITES)
        for a in full for b in full
    ) >= cfg["gates"]["power_abs_kw"]
    return dict(passed=separated, key=key, lower_digest=model.lower_digest(), frozen_raw_sha256=sha(raw),
                command_slot=24, action_slot=25, demand_kw=SERVICE_KW, limits_kw=caps,
                continuous_a_kw=[max(0.,SERVICE_KW-caps["B"]), min(SERVICE_KW,caps["A"])],
                projected_commands=choices, distinct_mask=mask, responses=responses)


def ledger(frames, sensors):
    consumed = {r["used_slot"]:r for r in frames if r["used_slot"] is not None}
    measured_by_slot = {r["slot"]:{s:sensors[s][i] for s in SITES} for i,r in enumerate(frames)}
    rows = []
    for r in frames:
        slot = r["slot"]
        origin = slot-1-(slot-1)%2
        physical = consumed.get(slot)
        demand = service_kw(origin)
        actual = sum(r["delivered"].values())
        dt = r["end_s"]-r["write_s"]
        gap = demand-actual
        rows.append(dict(slot=slot, command_slot=origin, external_demand_kw=demand,
                         request_kw=r["request"], pcc_kw_kvar=r["pcc"], delivered_kw=r["delivered"],
                         signed_gap_kw=gap, unmet_kw=max(0.,gap), excess_kw=max(0.,-gap),
                         duration_s=dt, physical_slot=physical["slot"] if physical else None,
                         native_port_kw_kvar=measured_by_slot[physical["slot"]] if physical else None,
                         veto={s:r["actual"][s]["veto"] for s in SITES}))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("precheck","train","learned","fixed"))
    for opt in ("out","stage0","reference","mechanism","lower"):
        p.add_argument("--"+opt, type=Path, required=True)
    for opt in ("precheck","frozen-trace","checkpoint"):
        p.add_argument("--"+opt, type=Path)
    p.add_argument("--continuous",action="store_true",help="bounded Beta master and issued urgency")
    p.add_argument("--ablate",choices=("uplink","downlink"),help="remove one added feature channel in a separate comparison")
    args = p.parse_args()
    if args.ablate and not args.continuous:
        p.error("feature ablation requires the continuous controller")
    if subprocess.check_output(["git","status","--porcelain"]):
        raise ValueError("requires clean checkout")
    train = args.mode == "train"
    cfg = dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    args.dry_run = False
    gates(args,cfg)
    key = dict(git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               config=sha(CONFIG), wiring=sha(WIRING), reference=sha(args.reference),
               mechanism=sha(args.mechanism), stage0=sha(args.stage0/"summary.json"), lower=sha(args.lower),
               frequency_hz=60, track_weight=TRACKING_WEIGHT, service_kw=SERVICE_KW, service_slots=list(SERVICE_SLOTS),
               versions={k:version(k) for k in ("dss-python","torch","matlabengine","numpy")})
    policy_type = ContinuousServicePolicy if args.continuous else EqualServicePolicy
    controller_type = (partial(ContinuousServiceCoordinator,ablation=args.ablate)
                       if args.continuous else EqualServiceCoordinator)
    if args.continuous:
        key["controller"] = "continuous_beta_issued_urgency"
        key["ablation"] = args.ablate
    setup(cfg["study"]["seed" if train or args.mode=="precheck" else "eval_seed"])
    model = freeze_model(args.lower,key,policy_type)
    lower_digest = model.lower_digest()
    if args.mode == "precheck":
        with torch.no_grad():
            result = precheck(args,cfg,key,model,controller_type)
        dump(args.out,result)
        if not result["passed"]:
            raise ValueError("precheck found no separable full-demand split")
        return
    gate = json.loads(args.precheck.read_bytes())
    if not gate["passed"] or gate["key"] != key or gate["lower_digest"] != lower_digest:
        raise ValueError("incompatible or failed native precheck")
    key["precheck"] = sha(args.precheck)
    if not train:
        ckpt = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        if ckpt["key"] != key:
            raise ValueError("evaluation checkpoint key mismatch")
        model.load_state_dict(ckpt["weights"])
        if state_digest(model) != ckpt["digest"] or model.lower_digest() != lower_digest:
            raise ValueError("evaluation checkpoint digest mismatch")
    before = state_digest(model)
    seed = cfg["study"]["seed" if train else "eval_seed"]
    event = cfg["events"]["development" if train else "evaluation"]
    controller = controller_type(model,cfg,not train,fixed_master=args.mode=="fixed")
    args.out.mkdir(parents=True,exist_ok=False)
    raw = args.out/"raw.jsonl"
    with report(raw) as emit:
        emit(dict(phase="manifest", key=key, argv=sys.argv, mode=args.mode, seed=seed, event=event,
                  scope="60 Hz engineering, master only", before=before, lower_digest=lower_digest))
        with torch.set_grad_enabled(train):
            _, _, frames, trace, sensors, ref = native_episode(model,cfg,"clock_graph",event,args.reference,
                                                                seed,not train,emit,coordinator=controller)
            if abs(trace[0][0])>1e-8 or abs(trace[0][1]-60)>.01:
                raise ValueError("unexpected native initialization frequency")
            groups = controller.returns(frames,trace,ref)
            opt = update(model,groups,cfg["study"]["learning_rate"]) if train else None
        if len(groups["master"]) != len(cfg["window_slots"]) or model.lower_digest() != lower_digest:
            raise ValueError("incomplete rollout or changed frozen lower")
        after = state_digest(model)
        path = args.checkpoint
        if train:
            path = args.out/"model.pt"
            with path.open("xb") as f:
                torch.save(dict(weights=model.state_dict(),key=key,digest=after),f)
            snap = torch.load(path,map_location="cpu",weights_only=True)
            model.load_state_dict(snap["weights"])
            if snap["key"]!=key or state_digest(model)!=after or model.lower_digest()!=lower_digest:
                raise ValueError("trained checkpoint readback mismatch")
        elif after != before:
            raise ValueError("evaluation updated weights")
        steps = ledger(frames,sensors)
        metrics = measure(frames,trace,sensors,ref,event)
        for name in ("unmet","excess"):
            metrics[name+"_external_kwh"] = sum(r[name+"_kw"]*r["duration_s"] for r in steps)/3600.
        result = dict(key=key, mode=args.mode, seed=seed, event=event, before=before, after=after,
                      lower_digest=lower_digest, checkpoint_sha256=sha(path), optimizer=opt,
                      metrics=metrics, native_steps=steps, formal_50hz=False)
        emit(dict(phase="complete",**result))
    dump(args.out/"summary.json",dict(raw_sha256=sha(raw),**result))


if __name__ == "__main__":
    main()
