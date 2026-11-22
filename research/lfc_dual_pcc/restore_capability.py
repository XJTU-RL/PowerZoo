"""Align the completed handoff trace once, then gate three independent DSS restorations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .capability_state import PROTOCOL, add_max, compare_step, exogenous, largest, read_config, restore
from .case_a_plant import CONFIG, feeder
from .compare import load
from .driver import dump, sha
from .handoff import HANDOFF
from .score_handoff import audit, signals
from .stage0 import CONFIG as WIRING


def timeline(trial, physical, hz, cfg, tol):
    frames = trial["frames"]
    consumed = {row["used_slot"]: row for row in frames if row["used_slot"] is not None}
    anchors = json.loads(WIRING.read_bytes())["ports"]
    history = []
    initial = frames[0]["pcc"]
    for i, row in enumerate(frames):
        slot, start, end = row["slot"], row["write_s"], row["end_s"]
        t, freq = row["frequency"]
        previous = frames[i-2] if i >= 2 else None
        if row["used_slot"] != (None if previous is None else previous["slot"]):
            raise ValueError("wrong report source slot")
        eligible = [(x, y) for x, y in hz["samples"] if x < start-tol]
        native_time, native_hz = eligible[-1]
        if (abs(native_time-t)>tol or abs(native_hz-freq)>1e-8
                or not 0 < start-t <= cfg["clock"]["native_frequency_period_s"]+tol
                or abs(end-start-1)>tol or abs(row["arrival_s"]-end-cfg["clock"]["report_latency_s"])>tol):
            raise ValueError("clock, ZOH duration or frequency mismatch")
        if previous and (not previous["arrival_s"] < start or abs(row["received_s"]-previous["arrival_s"])>tol):
            raise ValueError("report not yet available or mis-stamped")
        consumer = consumed.get(slot)
        if (i < len(frames)-2) != (consumer is not None):
            raise ValueError("missing or duplicate consumer")
        sensors = {}
        for site in ("A", "B"):
            measurements = physical[site]
            if len(measurements["p_kw"]) != len(measurements["q_kvar"]):
                raise ValueError("unmatched physical P/Q lengths")
            post = [(x, p, q) for (x,p),(z,q) in zip(measurements["p_kw"],measurements["q_kvar"], strict=True)
                    if abs(x-z)<=tol and start+tol<x<=end+tol]
            if not post or not 0 < post[0][0]-start <= 2/60+tol:
                raise ValueError("no strict post-write physical sample")
            anchor=anchors[site]["anchor_kw_kvar"]
            expected=[a + (previous["pcc"][site][j]-initial[site][j] if previous else 0.)
                      + (row["event_kw"] if site=="A" and j==0 else 0.) for j,a in enumerate(anchor)]
            if largest(expected,row["target"][site]) > .005:
                raise ValueError("P/Q from wrong source or baseline")
            act=row["actual"][site]
            if act["action"]!=row["actions"][site] or act["end_local_s"]!=i+1 or largest(row["pcc"][site], act["pcc"])>1e-8:
                raise ValueError("device action, local clock or PCC readback mismatch")
            sensors[site]={"first_strict_postwrite_s_p_kw_q_kvar":post[0],
                           "last_sample_s":post[-1][0], "count":len(post),
                           "tail_mean_p_kw_q_kvar":trial["sensors"][site][i]}
        history.append({"slot":slot,"command_generated_at_sim_pause_s":start,"dss_local_solve_s":i,
                        "dss_local_finished_s":i+1,"hold_effective_sim_s":[start,end],
                        "report_sample_close_s":end,"report_arrival_s":row["arrival_s"],
                        "this_report_consumed_at_sim_s":None if consumer is None else consumer["write_s"],
                        "main_used_report_slot":row["used_slot"],"main_used_arrival_s":row["received_s"],
                        "causal_frequency_s_hz":[t,freq],"event_load_A_kw":row["event_kw"],
                        "incremental_feeder_exogenous_load_kw":0.,"exogenous_binding":"inputs.json",
                        "target_grid_pq_kw_kvar":row["target"],"main_sensors":sensors,
                        "request_kw":row["request"],"actions":row["actions"],"observed_dss":row["actual"],
                        "report":row["reports"]})
    return history


def physical(path):
    rows = [json.loads(l) for l in path.open()]
    return {r["port"]:r for r in rows if r["phase"]=="physical_native"}


def gate(args):
    cfg, protocol=read_config()
    tol=protocol["tolerances"]
    paired=audit(args.neutral,args.issued,args.delivered)
    trial=load(args.delivered)
    header=trial["header"]
    if (header["git"]!=protocol["origin_git"] or header["arm"]!=protocol["source_arm"] or
            header["reference_sha256"]!=sha(args.reference) or header["config_sha256"]!=sha(CONFIG)
            or header["handoff_sha256"]!=sha(HANDOFF)):
        raise ValueError("wrong source or reference")
    _, hz=signals(args.delivered)
    history=timeline(trial,physical(args.delivered),hz,cfg,tol["time_s"])
    frames=trial["frames"]
    slots={f["slot"]:f for f in frames}
    keys=protocol["key_slots"]
    if (paired["delivered"]["issue_slot"]<=keys["before_pickup"]
            or not slots[keys["b_first_at_target"]-1]["delivered"]["B"]>=7
            or slots[keys["b_first_at_target"]-2]["delivered"]["B"]>=7
            or slots[keys["a_limited_after_b_ready"]-1]["delivered"]["A"]>=4):
        raise ValueError("frozen state landmarks do not describe this trace")
    reference=ObservedReference.read_report(args.reference.read_bytes())
    inputs={}; checks={}; features=[]
    with tempfile.TemporaryDirectory() as t:
        deck=native.archived_deck(ROOT,Path(t))
        for site in cfg["sites"]:
            with feeder(deck,reference,cfg,site) as f:
                inputs[site]=exogenous(f)
                totals={}
                for frame in frames:
                    current=f.execute(frame["actions"][site])
                    add_max(totals,compare_step(current,frame["actual"][site],tol))
                    if exogenous(f)!=inputs[site]:
                        raise ValueError("external input bindings changed during the source prefix")
                checks[site]={"all_frame_max":totals,"states":{}}
            for key,slot in keys.items():
                with feeder(deck,reference,cfg,site) as f:
                    prefix=restore(f,site,frames,slot,tol)
                    initial_soc=f.soc()
                    previous=f.previous
                    if key=="a_limited_after_b_ready" and site=="A" and f.precheck(3)!="energy":
                        raise ValueError("A limit not attributable to usable energy")
                    following={}
                    for offset in range(protocol["same_action_steps"]):
                        frame=slots[slot+offset]
                        add_max(following,compare_step(f.execute(frame["actions"][site]),frame["actual"][site],tol))
                    checks[site]["states"][key]={"start_slot":slot,"prefix_max":prefix,"subsequent_same_action_max":following}
                    now=slots[slot];old=slots[now["used_slot"]]
                    features.append({"key":key,"site":site,"as_of_s":now["write_s"],
                              "local_state_from_completed_slot":slot-1, "local_soc":initial_soc,
                              "previous_nominal_kw":previous, "local_pq":slots[slot-1]["pcc"][site],
                              "local_voltage_pu":slots[slot-1]["actual"][site]["voltage"],
                              "upper_report_sample_s":old["end_s"],"upper_report_received_s":old["arrival_s"],
                              "upper_report":old["reports"][site],"upper_frequency_s_hz":now["frequency"],
                              "prefix_actions":[r["actions"][site] for r in frames if r["slot"]<slot]})
    dump(args.out/"inputs.json",{"source":"frozen native deck, restored private reference, verified every solve",
         "reference_sha256":sha(args.reference),"deck_tree":native.DECK_TREE,"sites":inputs,
         "ramp_up_kw_per_s":cfg["ramp_up"],"ramp_down_kw_per_s":cfg["ramp_down"],
         "actions_battery_pv_kw":cfg["actions_battery_pv_kw"]})
    dump(args.out/"observed-history.json",history)
    dump(args.out/"asof-states.json",features)
    report={"passed":True,"physics_scope":"OpenDSS prefix restoration; recorded Simulink trajectory, no Simulink state restoration",
            "gate_git":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "origin_git":header["git"],"historical_trace_sha256":sha(args.delivered),"reference_sha256":sha(args.reference),
            "protocol_sha256":sha(PROTOCOL),"config_sha256":sha(CONFIG),"key_slots":keys,"tolerances":tol,
            "frames":len(frames),"sites":checks,"inputs_sha256":sha(args.out/"inputs.json"),
            "history_sha256":sha(args.out/"observed-history.json"),"states_sha256":sha(args.out/"asof-states.json")}
    dump(args.out/"restoration.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("neutral","issued","delivered","reference","out"): p.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        gate(args)
    except Exception as exc:
        dump(args.out/"failed.json",{"reason":repr(exc)})
        raise
