"""Probe sustainable upward PCC delivery from three restored DSS states with a fixed policy catalogue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .capability_state import PROTOCOL, compare_step, largest, read_config
from .case_a_plant import CONFIG, feeder
from .driver import dump, sha


def start(plant, state, tolerance):
    for action in state["prefix_actions"]:
        plant.execute(action)
    if (plant.s != len(state["prefix_actions"]) or plant.previous != state["previous_nominal_kw"]
            or largest(plant.soc(), state["local_soc"]) > tolerance["soc_pu"]
            or largest(plant.last["pcc"], state["local_pq"]) > tolerance["pq_kw_kvar"]
            or set(plant.last["voltage"]) != set(state["local_voltage_pu"])
            or largest(plant.last["voltage"].values(), [state["local_voltage_pu"][k] for k in plant.last["voltage"]]) > tolerance["voltage_pu"]):
        raise ValueError("unmatched as-of state")


def run_policy(plant, state, labels, cap, cfg, protocol):
    start(plant, state, protocol["tolerances"])
    requested = labels.index(cap)
    trace = []
    for index in range(max(protocol["horizons_s"])):
        slot_start = state["as_of_s"] + index
        previous = plant.previous
        choices = plant.candidates()
        valid = [i for i, row in enumerate(choices) if labels[i] <= cap and not row["veto"]]
        if not valid:
            return trace
        selected = max(valid, key=lambda i: plant.base[0] - choices[i]["pcc"][0])
        before = plant.soc()
        battery, _ = cfg["actions_battery_pv_kw"][requested]
        usable = min(max(0., s-plant.storage["reserve"])*plant.storage["kwh"] for s in before)
        reasons = list(choices[requested]["veto"])
        limit_flags = {
            "target_ramp_infeasible": cap-previous > cfg["ramp_up"][plant.site]+1e-8 or previous-cap > cfg["ramp_down"]+1e-8,
            "target_battery_energy_infeasible": battery/(len(before)*3600*plant.storage["eta"]) > usable+1e-10}
        row = plant.execute(selected)
        trace.append({"step":index+1,"hypothetical_hold_s":[slot_start,slot_start+1],
                      "ceiling_kw":cap,"previous_nominal_kw":previous,"selected_nominal_kw":labels[selected],
                      "target_veto":reasons,"target_limit_flags":limit_flags,
                      "veto_by_action":{i:c["veto"] for i,c in enumerate(choices)},
                      "actual":row, "delivered_support_kwh":row["delivered_kw"]/3600,
                      "battery_terminal_export_kwh":sum(row["battery_kw"])/3600,
                      "battery_internal_draw_kwh":sum(s-e for s,e in zip(before,row["soc_after"],strict=True))*plant.storage["kwh"]})
    return trace


def probe(args):
    cfg, protocol = read_config()
    gate = json.loads((args.gate/"restoration.json").read_bytes())
    if (not gate["passed"] or gate["origin_git"] != protocol["origin_git"]
            or gate["protocol_sha256"] != sha(PROTOCOL) or gate["config_sha256"] != sha(CONFIG)
            or gate["reference_sha256"] != sha(args.reference)):
        raise ValueError("unmatched restoration, reference, or frozen protocol")
    for file, key in (("asof-states.json","states_sha256"),("inputs.json","inputs_sha256"),("observed-history.json","history_sha256")):
        if sha(args.gate/file) != gate[key]:
            raise ValueError("modified restoration evidence")
    ref = ObservedReference.read_report(args.reference.read_bytes())
    states = json.loads((args.gate/"asof-states.json").read_bytes())
    if {(s["key"],s["site"]) for s in states} != {(k,p) for k in protocol["key_slots"] for p in cfg["sites"]} or len(states)!=6:
        raise ValueError("unexpected probe states")
    action_kw = [sum(action) for action in cfg["actions_battery_pv_kw"]]
    if sorted(action_kw) != protocol["caps_kw"]:
        raise ValueError("candidate policies do not match frozen action table")
    results = []
    with tempfile.TemporaryDirectory() as t:
        deck = native.archived_deck(ROOT,Path(t))
        for state in states:
            group = args.out / (state["key"] + "-" + state["site"])
            group.mkdir()
            paths = {}
            for cap in protocol["caps_kw"]:
                with feeder(deck,ref,cfg,state["site"]) as f:
                    paths[cap] = run_policy(f,state,action_kw,cap,cfg,protocol)
                dump(group / f"ceiling-{cap}.json",paths[cap])
            capability = []
            for h in protocol["horizons_s"]:
                eligible = {cap:min(step["actual"]["delivered_kw"] for step in trace[:h])
                            for cap,trace in paths.items() if len(trace)>=h}
                if not eligible:
                    raise ValueError("no safe witness reaches horizon")
                cap = max(eligible, key=eligible.get)
                winner = paths[cap][:h]
                replay = []
                with feeder(deck,ref,cfg,state["site"]) as f:
                    start(f,state,protocol["tolerances"])
                    for entry in winner:
                        actual=f.execute(entry["actual"]["action"])
                        compare_step(actual,entry["actual"],protocol["tolerances"])
                        replay.append(actual["delivered_kw"])
                if abs(min(replay)-eligible[cap])>protocol["tolerances"]["witness_kw"]:
                    raise ValueError("independent witness delivery mismatch")
                binding = next(({"step":x["step"],"reasons":x["target_veto"]} for x in winner if x["target_veto"]),None)
                max_command = next(({"step":x["step"],"reasons":x["veto_by_action"][action_kw.index(max(action_kw))]}
                                    for x in winner if x["veto_by_action"][action_kw.index(max(action_kw))]),None)
                capability.append({"horizon_s":h,"validated_sustainable_kw":min(replay),
                        "witness_ceiling_kw":cap,"witness_actions":[s["actual"]["action"] for s in winner],
                        "witness_measured_kw":replay,"witness_energy_kwh":sum(replay)/3600,
                        "first_rejection_of_witness_ceiling":binding,"first_rejection_of_max_command":max_command,
                        "tested_policy_values_kw":eligible,"safe_policy_count":len(eligible)})
            results.append({"state":state["key"],"site":state["site"],"as_of_s":state["as_of_s"],
                            "labels":capability})
    dump(args.out/"labels.json",{"probe_git":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "restoration_sha256":sha(args.gate/"restoration.json"),"origin_sha256":gate["historical_trace_sha256"],
            "search":protocol["probe"],"scope":"OpenDSS local PCC actual support against initial import. Static local exogenous inputs, no transmission rerun or voltage feedback. Each row is a feasible lower bound.",
            "states":results})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("gate","reference","out"): p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        probe(args)
    except Exception as exc:
        dump(args.out/"failed.json",{"reason":repr(exc)})
        raise
