"""Check a feeder-local as-of estimator request without predicting or fitting capacity."""
import argparse
import json
import math
from pathlib import Path
from .capability_state import read_config
from .driver import dump, sha


def request(gate_dir, slot, site, requested_kw, deadline_s, sustain_s):
    cfg, protocol=read_config()
    gate=json.loads((gate_dir/"restoration.json").read_bytes())
    if (not gate["passed"] or gate["states_sha256"]!=sha(gate_dir/"asof-states.json")
            or gate["inputs_sha256"]!=sha(gate_dir/"inputs.json")):
        raise ValueError("unverified observation source")
    if not (math.isfinite(requested_kw) and requested_kw>=0 and deadline_s>0 and sustain_s>0):
        raise ValueError("invalid upward service query")
    states=json.loads((gate_dir/"asof-states.json").read_bytes())
    state,=[s for s in states if protocol["key_slots"][s["key"]]==slot and s["site"]==site]
    now=state["as_of_s"]
    if (state["local_state_from_completed_slot"]!=slot-1 or state["upper_report_received_s"]>=now
            or not state["upper_report_sample_s"]<state["upper_report_received_s"]
            or state["upper_frequency_s_hz"][0]>=now):
        raise ValueError("noncausal feature provenance")
    inputs=json.loads((gate_dir/"inputs.json").read_bytes())["sites"][site]
    previous=json.loads((gate_dir/"observed-history.json").read_bytes())[slot-1-cfg["clock"]["first_hold_s"]]
    if previous["slot"]!=slot-1 or abs(previous["report_sample_close_s"]-now)>1e-7 or gate["history_sha256"]!=sha(gate_dir/"observed-history.json"):
        raise ValueError("local current output lacks a completed observation")
    actual=previous["observed_dss"][site]
    low,high=cfg["gates"]["voltage"]
    return {"status":"input contract only","role":"feeder-local estimator before dispatch",
            "decision_s":now,"asof_inputs":{"site":site,"soc_after":state["local_soc"],
            "phase_voltage_margin_pu":{k:[v-low,high-v] for k,v in state["local_voltage_pu"].items()},
            "measured_pcc_kw_kvar":state["local_pq"],"actual_storage_kw":actual["battery_kw"],
            "actual_pv_kw":actual["pv_kw"],"last_nominal_kw":state["previous_nominal_kw"],
            "ramp_down_up_kw_s":[cfg["ramp_down"],cfg["ramp_up"][site]],"load_multiplier":inputs["load_multiplier"],
            "irradiance":float(inputs["pv"]["irradiance"]),"last_arrived_grid_frequency":state["upper_frequency_s_hz"]},
            "request":{"kw":requested_kw,"reach_deadline_s":deadline_s,"sustain_after_reach_s":sustain_s},
            "outputs":{"estimated_kw":None,"estimated_reach_s":None,"estimated_sustain_s":None,
                       "constraint_violation_estimate":None,"uncertainty":None},
            "scope":"No fitted model or inference. Local voltages are not made available to the master before arrival."}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--slot",type=int,required=True)
    p.add_argument("--site",choices=["A","B"],required=True)
    p.add_argument("--request-kw",type=float,required=True)
    p.add_argument("--reach-deadline-s",type=int,required=True)
    p.add_argument("--sustain-s",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    dump(a.out,request(a.gate,a.slot,a.site,a.request_kw,a.reach_deadline_s,a.sustain_s))
