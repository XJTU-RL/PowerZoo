"""Validate time-to-power and a disjoint sustain segment against a same-state no-regulation branch."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .capability_state import PROTOCOL, compare_step, read_config
from .case_a_plant import feeder
from .driver import dump, sha
from .probe_capability import run_policy, start

RULE = Path(__file__).with_name("reach_hold_protocol.json")


def select(paths, duration, rule):
    """One reaching measurement plus H subsequent, non-overlapping 1 s holds."""
    if rule["report_local_step_s"] != 1 or duration not in rule["hold_seconds"]:
        raise ValueError("incompatible horizon or sampling period")
    precision, guard = rule["power_resolution_kw"], rule["power_guard_kw"]
    choices = []
    for cap, powers in paths.items():
        for reach in range(1, rule["reach_deadline_s"]+1):
            segment = powers[reach-1:reach+duration]
            if len(segment) != duration+1 or not all(map(math.isfinite, segment)):
                continue
            minimum = min(segment)
            if minimum < -guard:
                continue
            units = math.floor(max(0.,minimum-guard)/precision)
            choices.append((units,-reach,-cap,minimum))
    if not choices:
        raise ValueError("no safely validated candidate")
    units, nr, nc, minimum = max(choices)
    return round(units*precision,7), -nr, -nc, minimum


def first_veto(steps, start_s):
    for row in steps:
        if row["target_veto"]:
            return {"step":row["step"],"decision_s":start_s + row["step"]-1,
                    "reasons":row["target_veto"],"nominal_ceiling_kw":row["ceiling_kw"]}
    return None


def complete(deck, ref, state, root, cfg, protocol, rule, out):
    labels = [sum(v) for v in cfg["actions_battery_pv_kw"]]
    site=state["site"]
    horizon=rule["reach_deadline_s"]+max(rule["hold_seconds"])
    paths={}
    for cap in protocol["caps_kw"]:
        old=json.loads((root/f"ceiling-{cap}.json").read_bytes())
        if len(old)!=max(protocol["horizons_s"]):
            raise ValueError("archived prefix incomplete")
        with feeder(deck,ref,cfg,site) as f:
            start(f,state,protocol["tolerances"])
            for i,r in enumerate(old):
                if r["step"]!=i+1 or r["ceiling_kw"]!=cap or r["hypothetical_hold_s"]!=[state["as_of_s"]+i,state["as_of_s"]+i+1]:
                    raise ValueError("invalid archived candidate prefix")
                actual=f.execute(r["actual"]["action"])
                compare_step(actual,r["actual"],protocol["tolerances"])
            tail=run_policy(f,state,labels,cap,cfg,protocol,completed=len(old),stop=horizon)
            if len(tail)+len(old)!=horizon:
                raise ValueError("candidate continuation incomplete")
        paths[cap]=old+tail
        dump(out/f"extension-{cap}.json",{"source_sha256":sha(root/f"ceiling-{cap}.json"),"additional_steps":tail})
    if any(r["actual"]["action"]!=0 for r in paths[0]):
        raise ValueError("baseline does not remove incremental battery and PV dispatch")
    base=paths[0]
    net={c:[z["actual"]["pcc"][0]-r["actual"]["pcc"][0] for z,r in zip(base,steps,strict=True)]
         for c,steps in paths.items()}
    outlabels=[]
    t0=state["as_of_s"]
    for duration in rule["hold_seconds"]:
        target,reach,cap,minimum=select(net,duration,rule)
        stop=reach+duration
        readings=[]
        with feeder(deck,ref,cfg,site) as active, feeder(deck,ref,cfg,site) as neutral:
            start(active,state,protocol["tolerances"]);start(neutral,state,protocol["tolerances"])
            for i in range(stop):
                row=paths[cap][i]["actual"]
                a=active.execute(row["action"])
                z=neutral.execute(0)
                compare_step(a,row,protocol["tolerances"])
                compare_step(z,base[i]["actual"],protocol["tolerances"])
                delta=z["pcc"][0]-a["pcc"][0]
                if abs(net[cap][i]-delta)>rule["witness_parity_kw"]:
                    raise ValueError("independent same-state incremental delivery mismatch")
                readings.append({"step":i+1,"interval_s":[t0+i,t0+i+1],"candidate":a,"no_regulation":z,
                                 "delta_kw":delta,"delta_q_kvar":z["pcc"][1]-a["pcc"][1]})
        requirement=target+rule["power_guard_kw"] if target else -rule["power_guard_kw"]
        if min(s["delta_kw"] for s in readings[reach-1:]) < requirement-1e-8:
            raise ValueError("independent candidate and baseline do not meet the quantized label")
        crossing=next(s["step"] for s in readings if s["delta_kw"]>=requirement-1e-8)
        dump(out/f"witness-{duration}s.json",readings)
        outlabels.append({"target_kw":target,"decision_s":t0,"rise_deadline_s":t0+rule["reach_deadline_s"],
              "first_crossing_since_decision_s":crossing,"confirmed_reach_since_decision_s":reach,
              "confirmed_reach_s":t0+reach,"sustain_s":duration,"sustain_interval_s":[t0+reach,t0+stop],
              "reach_sample_kw":readings[reach-1]["delta_kw"],
              "sustain_min_kw":min(s["delta_kw"] for s in readings[reach:]),
              "sustain_energy_kwh":sum(s["delta_kw"] for s in readings[reach:])/3600,
              "ceiling_kw":cap,"minimum_including_reach_kw":minimum,
              "winning_ceiling_first_veto":first_veto(paths[cap][:stop],t0),
              "max_ceiling_first_veto":first_veto(paths[max(paths)][:stop],t0),
              "witness_sha256":sha(out/f"witness-{duration}s.json")})
    return {"key":state["key"],"site":site,"labels":outlabels}


def main(args):
    cfg, protocol=read_config(); rule=json.loads(RULE.read_bytes())
    gate=json.loads((args.gate/"restoration.json").read_bytes())
    audit=json.loads((args.audit/"audit.json").read_bytes())
    if (not gate["passed"] or not audit["passed"] or gate["reference_sha256"]!=sha(args.reference)
            or gate["protocol_sha256"]!=sha(PROTOCOL) or gate["states_sha256"]!=sha(args.gate/"asof-states.json")
            or audit["gate_sha256"]!=sha(args.gate/"restoration.json")
            or audit["labels_sha256"]!=sha(args.probes/"labels.json")):
        raise ValueError("unmatched restored states or catalogue")
    states=json.loads((args.gate/"asof-states.json").read_bytes())
    if sorted({len(s["prefix_actions"])+cfg["clock"]["first_hold_s"] for s in states})!=rule["decision_states"]:
        raise ValueError("unexpected state set")
    ref=ObservedReference.read_report(args.reference.read_bytes())
    report=[]
    with tempfile.TemporaryDirectory() as t:
        deck=native.archived_deck(ROOT,Path(t))
        for state in states:
            name=state["key"]+"-"+state["site"]
            group=args.out/name
            group.mkdir()
            report.append(complete(deck,ref,state,args.probes/name,cfg,protocol,rule,group))
    dump(args.out/"reach-hold-labels.json",{"git":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
          "gate_sha256":audit["gate_sha256"],"historical_trace_sha256":gate["historical_trace_sha256"],
          "rules":rule,"rule_sha256":sha(RULE),"states":report,
          "scope":"feasible local PCC support; independent same-state zero incremental action baseline and witness; no full Simulink counterfactual"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("gate","probes","audit","reference","out"):p.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        main(args)
    except Exception as exc:
        dump(args.out/"failed.json",{"reason":repr(exc)})
        raise
