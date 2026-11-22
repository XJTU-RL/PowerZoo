"""Audit the historical timing, policy witnesses and first rejected demand on each trajectory."""
import argparse
import json
import math
from pathlib import Path

from .capability_state import read_config
from .driver import dump, sha
from .restore_capability import physical


def read(path):
    return json.loads(path.read_bytes())


def check_samples(raw, historical, tol):
    for site, quantities in raw.items():
        pq=list(zip(quantities["p_kw"], quantities["q_kvar"], strict=True))
        for ((t,p),(u,q)) in pq:
            if not all(math.isfinite(v) for v in (t,p,u,q)) or abs(t-u)>tol:
                raise ValueError("mismatched native P/Q timestamp or invalid value")
        samples=[t for (t,_),_ in pq]
        if any(abs((u-t)-1/60)>tol for t,u in zip(samples,samples[1:])):
            raise ValueError("gap or duplicate in the native electrical clock")
        for i,h in enumerate(historical):
            lo,hi=h["hold_effective_sim_s"]
            observed=[t for t in samples if lo+tol<t<=hi+tol]
            if len(observed) != h["main_sensors"][site]["count"]:
                raise ValueError("unmatched electrical frame count")
            if abs(observed[0]-lo-1/60)>tol or hi-observed[-1] > (1/60 if i==len(historical)-1 else 0)+tol:
                raise ValueError("electrical samples do not cover the declared clock")


def extract(gate_dir, probe_dir, trace, out):
    cfg, protocol=read_config()
    gate=read(gate_dir/"restoration.json")
    labels=read(probe_dir/"labels.json")
    if not gate["passed"] or gate["historical_trace_sha256"] != sha(trace) or labels["restoration_sha256"] != sha(gate_dir/"restoration.json"):
        raise ValueError("unmatched raw, gate or labels")
    historical=read(gate_dir/"observed-history.json")
    bindings=read(gate_dir/"inputs.json")
    if sha(gate_dir/"observed-history.json")!=gate["history_sha256"] or sha(gate_dir/"inputs.json")!=gate["inputs_sha256"]:
        raise ValueError("modified history or plant inputs")
    check_samples(physical(trace),historical,protocol["tolerances"]["time_s"])
    previous={s:0. for s in cfg["sites"]}
    ticks=[]
    for row in historical:
        if row["main_used_arrival_s"] is not None and row["main_used_arrival_s"]>=row["command_generated_at_sim_pause_s"]:
            raise ValueError("future input in a historical feature")
        uses=row["this_report_consumed_at_sim_s"]
        if uses is not None and uses<=row["report_arrival_s"]:
            raise ValueError("observed report was consumed early")
        inputs={}
        for s in cfg["sites"]:
            battery,pv=cfg["actions_battery_pv_kw"][row["actions"][s]]
            act=row["observed_dss"][s]
            command=battery+pv
            inputs[s]={"battery_dispatch_kw":battery,"pv_setpoint_kw":cfg["pv"]["base_kw"]+pv,
                       "pv_irradiance":float(bindings["sites"][s]["pv"]["irradiance"]),
                       "load_multiplier":bindings["sites"][s]["load_multiplier"],
                       "load_parameters":"inputs.json#sites/"+s+"/loads",
                       "device_cmd_increment_kw_per_s":command-previous[s],
                       "ramp_limits_down_up_kw_per_s":[cfg["ramp_down"],cfg["ramp_up"][s]],
                       "soc_before_after_pu":[act["soc"],act["soc_after"]],
                       "dss_measured_pq_kw_kvar":act["pcc"],"actual_storage_kw":act["battery_kw"],
                       "actual_pv_kw":act["pv_kw"]}
            previous[s]=command
        ticks.append({"slot":row["slot"],"command_s":row["command_generated_at_sim_pause_s"],
                      "hold_s":row["hold_effective_sim_s"],"report_sample_s":row["report_sample_close_s"],
                      "report_arrival_s":row["report_arrival_s"],"consumed_s":uses,
                      "grid_frequency_s_hz":row["causal_frequency_s_hz"],"external_grid_step_kw":row["event_load_A_kw"],
                      "inputs_and_actual":inputs,"phase_voltage_source":"observed-history.json: observed_dss[site].voltage at this slot"})
    dump(out/"tick-index.json",ticks)
    checked=[]
    for state in labels["states"]:
        root=probe_dir/(state["state"]+"-"+state["site"])
        candidates={c:read(root/f"ceiling-{c}.json") for c in protocol["caps_kw"]}
        aggressive=next(({"step":r["step"],"reasons":r["target_veto"]} for r in candidates[max(candidates)] if r["target_veto"]),None)
        for measured in state["labels"]:
            h=measured["horizon_s"]
            finite={c:min(r["actual"]["delivered_kw"] for r in steps[:h]) for c,steps in candidates.items() if len(steps)>=h}
            best=max(finite.values())
            chosen=measured["witness_ceiling_kw"]
            if (not math.isclose(best,finite[chosen],abs_tol=1e-8) or
                    abs(best-measured["validated_sustainable_kw"])>protocol["tolerances"]["witness_kw"]):
                raise ValueError("label does not match safe paths and independent witness")
        checked.append({"state":state["state"],"site":state["site"],"first_veto_at_10_kw_ceiling":aggressive,
                        "sustainable_kw_by_horizon":{str(v["horizon_s"]):v["validated_sustainable_kw"] for v in state["labels"]}})
    dump(out/"audit.json",{"passed":True,"gate_sha256":sha(gate_dir/"restoration.json"),
                           "labels_sha256":sha(probe_dir/"labels.json"),"native_period_s":1/60,
                           "observed_frames":len(ticks),"last_frames_without_later_consumer":[r["slot"] for r in ticks if r["consumed_s"] is None],
                           "scope":"aligned original trajectory and prefix-restored DSS local capability witnesses", "states":checked})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("gate","probes","trace","out"): p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(mode=0o700,parents=True,exist_ok=False)
    extract(args.gate,args.probes,args.trace,args.out)
