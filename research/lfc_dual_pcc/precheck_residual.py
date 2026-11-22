"""Pre-registered feeder-only opportunity test on the existing development clock."""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .case_a_plant import feeder
from .driver import sha
from .equal_service import LOWER_SOURCE, SERVICE_SLOTS
from .master_dispatch import SITES, bid_limits, feasible_actions, site_reports
from .residual_service import ResidualServiceCoordinator
from .revision_protocol import MIN_PHYSICAL_WINDOWS, RESIDUAL_PROBES, demand


def distinct_signatures(signatures, tolerance):
    representatives=[]
    for value in signatures:
        if not any(max(abs(a-b) for a,b in zip(value,other,strict=True))<tolerance for other in representatives):
            representatives.append(value)
    return len(representatives)


def precheck_residual(args, cfg, key, model, *, information_only=False, capture_delivery=False):
    source=args.frozen_trace/"raw.jsonl"
    meta=json.loads((args.frozen_trace/"summary.json").read_bytes())
    if (meta["mode"]!="train" or meta["raw_sha256"]!=sha(source) or meta["checkpoint_sha256"]!=key["lower"]
            or meta["key"]["git"]!=LOWER_SOURCE or meta["key"]["reference"]!=key["reference"]):
        raise ValueError("precheck requires the matched engineering development clock")
    raw=[json.loads(row) for row in source.read_text().splitlines()]
    if raw[-1]["phase"]!="complete":
        raise ValueError("incomplete development clock")
    holds={r["slot"]:r for r in raw if r["phase"]=="hold"}
    reference=ObservedReference.read_report(args.reference.read_bytes())
    freq=meta["metrics"]["reference_hz"]
    master={}
    states=[]
    prefix=[]

    def log(r):
        if r["phase"]=="policy" and r["role"]=="master":
            master[r["slot"]]=r

    def select(control, plants, slot):
        h=holds[slot]
        sample,hz=h["frequency"]
        return control.dispatch(slot,h["write_s"],sample,freq-hz,plants)

    def execute(control, plants, slot, request, action):
        actual={s:p.execute(action[s]) for s,p in plants.items()}
        frame=dict(holds[slot],request=request,actual=actual,actions=action)
        control.receive(frame,plants)
        return actual

    def replay(stack, deck, control, cmd):
        plants = {s: stack.enter_context(feeder(deck, reference, cfg, s)) for s in SITES}
        for old_action, old_actual in prefix:
            for site, plant in plants.items():
                plant.candidates()
                got = plant.execute(old_action[site])
                expected = old_actual[site]
                for field, tolerance in (("pcc", cfg["gates"]["parity_kw"]), ("soc_after", 1e-8)):
                    if any(abs(x-y) > tolerance for x, y in zip(got[field], expected[field], strict=True)):
                        raise ValueError("feeder development prefix mismatch")
        branch = ResidualServiceCoordinator(model, cfg, True, fixed_master=True)
        branch.emit = lambda record: None
        branch.graphs = control.graphs
        branch.reports = list(control.reports)
        branch.local_report = control.local_report
        branch.commands = [cmd if m.sequence == cmd.sequence else m for m in control.commands]
        return plants, branch

    def information_control(deck, control, slot, cmd):
        origin = cmd.sequence
        now = cmd.sampled
        latest = site_reports(control.reports, now)
        packet = latest["A"]
        uplink = []
        if packet is not None:
            # Deliver the same packet just after the decision in the intervention.
            delayed = [replace(m, arrival=now+cfg["clock"]["report_latency_s"]) if m is packet else m
                       for m in control.reports]
            limits = bid_limits(latest, cfg["fast_s"])
            sample, hz = holds[origin]["frequency"]
            for label, queue in (("timely", control.reports), ("delayed_shared_packet", delayed)):
                reports = site_reports(queue, now)
                ids = {s: msg.sequence if msg else None for s, msg in reports.items()}
                diagnostics = ResidualServiceCoordinator(model, cfg, True)
                events = []
                diagnostics.emit = events.append
                proposal, _ = diagnostics.master_allocation(freq-hz, reports, limits, origin, now, sample, ids)
                uplink.append(dict(label=label, reports=ids, observation=events[-1]["x"],
                                   caps_held_kw=limits, proposed_kw=proposal, dispatched_kw=cmd.data))
        downlink = []
        for label, urgency in (("issued", cmd.urgency_hz), ("zero", 0.)):
            with ExitStack() as trial:
                plants, branch = replay(trial, deck, control, replace(cmd, urgency_hz=urgency))
                events = []
                branch.emit = events.append
                table = cfg["actions_battery_pv_kw"]
                eligible = {s: feasible_actions(table, p.candidates(), cmd.data[s]) for s, p in plants.items()}
                req, selected = select(branch, plants, slot)
                actual = execute(branch, plants, slot, req, selected)
                if req != cmd.data:
                    raise ValueError("information test changed the basic command")
                downlink.append(dict(label=label, urgency_hz=urgency, native_feasible=eligible,
                                     request_kw=req, actions=selected,
                                     pcc_kw_kvar={s: actual[s]["pcc"] for s in SITES},
                                     device_contexts={r["site"]: r["context"] for r in events
                                                      if r["phase"] == "device_input"}))
        if downlink[0]["native_feasible"] != downlink[1]["native_feasible"]:
            raise ValueError("urgency test changed physical eligibility")
        return dict(origin=origin, fast_slot=slot, uplink=uplink, downlink=downlink,
                    master_weights="frozen trained residual" if information_only else "pretraining exact rule")

    information = None
    with tempfile.TemporaryDirectory() as scratch, ExitStack() as stack:
        deck=native.archived_deck(ROOT,Path(scratch))
        plants={s:stack.enter_context(feeder(deck,reference,cfg,s)) for s in SITES}
        control=ResidualServiceCoordinator(model,cfg,True,fixed_master=True)
        control.initialize(plants,log)
        for slot in holds:
            request,action=select(control,plants,slot)
            if slot-1 in SERVICE_SLOTS:
                origin=slot-1
                evidence=master[origin]
                lo,hi=evidence["feasible_a_kw"]
                total=evidence["allocatable_kw"]
                trials=[]
                cmd=control.command_at(slot,holds[slot]["write_s"])
                if origin == SERVICE_SLOTS[0]:
                    information = information_control(deck, control, slot, cmd)
                    if information_only:
                        return dict(key=key, lower_digest=model.lower_digest(), information=information,
                                    frozen_raw_sha256=sha(source))
                if hi-lo<=1e-9 or not math.isclose(total,demand(origin),rel_tol=0.,abs_tol=1e-9):
                    states.append(dict(slot=origin,interval_kw=[lo,hi],allocatable_kw=total,
                                       external_kw=demand(origin),distinct_native_pcc=0,alternatives=[]))
                else:
                    for latent in RESIDUAL_PROBES:
                        a=lo+latent*(hi-lo)
                        alternative=dict(A=a,B=total-a)
                        with ExitStack() as trial:
                            p2, branch = replay(trial, deck, control, replace(cmd, data=alternative))
                            actions,outputs,delivered=[],[],[]
                            for step in (slot,slot+1):
                                req,selected=select(branch,p2,step)
                                if req!=alternative:
                                    raise ValueError("probe consumed the wrong command")
                                actual=execute(branch,p2,step,req,selected)
                                outputs.append({s:actual[s]["pcc"] for s in SITES})
                                actions.append(selected)
                                if capture_delivery:
                                    delivered.append({s:actual[s]["delivered_kw"] for s in SITES})
                                    if any(actual[s]["veto"] for s in SITES):
                                        raise ValueError("executed trial encountered a native veto")
                        trials.append(dict(latent=latent,request_kw=alternative,pcc_kw_kvar=outputs,actions=actions))
                        if capture_delivery:
                            trials[-1]["delivered_kw"] = delivered
                    sig=[[r[s][0] for r in t["pcc_kw_kvar"] for s in SITES] for t in trials]
                    n=distinct_signatures(sig,cfg["gates"]["power_abs_kw"])
                    states.append(dict(slot=origin,interval_kw=[lo,hi],allocatable_kw=total,
                                       external_kw=demand(origin),distinct_native_pcc=n,alternatives=trials))
                    if capture_delivery:
                        admitted = site_reports(control.reports, cmd.sampled)
                        table = cfg["actions_battery_pv_kw"]
                        one_step = {}
                        for site, plant in plants.items():
                            rows = plant.candidates()
                            candidates = feasible_actions(table, rows, cmd.data[site])
                            one_step[site] = [plant.base[0]-rows[idx]["pcc"][0] for idx in candidates]
                        states[-1].update(observation=evidence["x"], reports=evidence["reports"],
                                          frequency_error_hz=cmd.urgency_hz,
                                          caps_kw=bid_limits(admitted,cfg["fast_s"]),
                                          native_one_step_support_kw=one_step)
            actual=execute(control,plants,slot,request,action)
            prefix.append((action,actual))
    opportunities=sum(r["distinct_native_pcc"]>=2 for r in states)
    return dict(passed=opportunities>=MIN_PHYSICAL_WINDOWS,required_windows=MIN_PHYSICAL_WINDOWS,
                physical_opportunities=opportunities,windows=states,information=information,key=key,lower_digest=model.lower_digest(),
                frozen_raw_sha256=sha(source),scope="original development clock, independently restored native feeders")
