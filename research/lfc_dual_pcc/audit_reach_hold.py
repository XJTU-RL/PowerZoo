"""Check the achieved support, reach, sustain and operational limits against saved native witnesses."""
import argparse
import json
import math
from pathlib import Path
from .capability_state import read_config
from .driver import dump, sha


def read(path):
    return json.loads(path.read_bytes())


def verify(gate, directory):
    cfg,_=read_config()
    inputs=read(gate/"asof-states.json")
    result=read(directory/"reach-hold-labels.json")
    if result["gate_sha256"]!=sha(gate/"restoration.json"):
        raise ValueError("unmatched restoration")
    rule=result["rules"]
    rows=[]
    prefixes=[]
    for s in result["states"]:
        state,=[i for i in inputs if i["key"]==s["key"] and i["site"]==s["site"]]
        for label in s["labels"]:
            duration=label["sustain_s"]
            path=directory/(s["key"]+"-"+s["site"])/f"witness-{duration}s.json"
            if sha(path)!=label["witness_sha256"]:
                raise ValueError("modified native witness")
            steps=read(path)
            due=label["confirmed_reach_since_decision_s"]
            t0=state["as_of_s"]
            if due>rule["reach_deadline_s"] or len(steps)!=due+duration or label["sustain_interval_s"]!=[t0+due,t0+due+duration]:
                raise ValueError("reach and sustain intervals overlap or exceed deadline")
            required=label["target_kw"]+(rule["power_guard_kw"] if label["target_kw"] else -rule["power_guard_kw"])
            previous={k:state["previous_nominal_kw"] for k in ("candidate","no_regulation")}
            voltages=[]; min_soc=1.; energy_error=0.; achieved=[]
            for j,step in enumerate(steps):
                if step["interval_s"]!=[t0+j,t0+j+1]:
                    raise ValueError("incomplete/doubled clock")
                delta=step["no_regulation"]["pcc"][0]-step["candidate"]["pcc"][0]
                achieved.append(delta)
                if abs(delta-step["delta_kw"])>1e-8 or j>=due-1 and delta<required-1e-8:
                    raise ValueError("power below label on reach or sustain")
                for k in previous:
                    row=step[k]
                    if row["veto"] or k=="no_regulation" and row["action"]!=0:
                        raise ValueError("unsafe witness or active baseline")
                    battery,pv=cfg["actions_battery_pv_kw"][row["action"]]
                    nom=battery+pv
                    if nom-previous[k]>cfg["ramp_up"][s["site"]]+1e-8 or previous[k]-nom>cfg["ramp_down"]+1e-8:
                        raise ValueError("command ramp failed")
                    previous[k]=nom
                    if len(row["voltage"])!=95 or not all(math.isfinite(v) and cfg["gates"]["voltage"][0]<=v<=cfg["gates"]["voltage"][1] for v in row["voltage"].values()):
                        raise ValueError("phase voltage violation")
                    voltages.extend(row["voltage"].values())
                    min_soc=min(min_soc,*row["soc_after"])
                    if min_soc<cfg["storage"]["reserve"]-1e-8 or abs(row["pv_kw"]-cfg["pv"]["base_kw"]-pv)>cfg["gates"]["power_abs_kw"]:
                        raise ValueError("SOC or PV terminal mismatch")
                    for pre,post,kw in zip(row["soc"],row["soc_after"],row["battery_kw"],strict=True):
                        if abs(kw-battery/len(row["battery_kw"]))>cfg["gates"]["power_abs_kw"]:
                            raise ValueError("storage terminal mismatch")
                        energy_error=max(energy_error,abs((post-pre)*cfg["storage"]["kwh"]+kw/(3600*cfg["storage"]["eta"])))
            if energy_error>cfg["gates"]["energy_abs_kwh"] or abs(sum(achieved[due:])/3600-label["sustain_energy_kwh"])>1e-10:
                raise ValueError("energy mismatch")
            if duration==10:
                # The same reached level/time anchors both shorter prefixes.
                checks=[]
                for h in (2,5):
                    short,=[r for r in s["labels"] if r["sustain_s"]==h]
                    checks.append({"duration_s":h,"confirmed_reach_s":label["confirmed_reach_s"],
                         "target_kw":label["target_kw"],"min_support_after_reach_kw":min(achieved[due:due+h]),
                         "matches_separately_selected_target_and_reach":(
                             short["target_kw"]==label["target_kw"] and short["confirmed_reach_s"]==label["confirmed_reach_s"])})
                kept=steps[due:]
                battery=sum(sum(z["candidate"]["battery_kw"]) for z in kept)/3600
                pv=sum(z["candidate"]["pv_kw"]-z["no_regulation"]["pv_kw"] for z in kept)/3600
                used=sum(sum(x-y for x,y in zip(z["candidate"]["soc"],z["candidate"]["soc_after"],strict=True))
                         for z in kept)*cfg["storage"]["kwh"]
                prefixes.append({"state":s["key"],"site":s["site"],"ten_s_target_kw":label["target_kw"],
                     "same_reach_prefixes":checks,
                     "initial_usable_storage_dc_kwh":sum((soc-cfg["storage"]["reserve"])*cfg["storage"]["kwh"] for soc in state["local_soc"]),
                     "ten_s_hold_battery_ac_kwh":battery,"ten_s_hold_battery_internal_draw_kwh":used,
                     "ten_s_hold_incremental_pv_kwh":pv,"ten_s_hold_incremental_pcc_kwh":sum(achieved[due:])/3600})
            rows.append({"state":s["key"],"site":s["site"],"target_kw":label["target_kw"],
                         "reach_s":due,"sustain_s":duration,"min_delivery_headroom_kw":min(achieved[due-1:])-label["target_kw"],
                         "min_phase_pu":min(voltages),"max_phase_pu":max(voltages),"min_soc_pu":min_soc,
                         "max_energy_ledger_error_kwh":energy_error})
    if len(rows)!=18: raise ValueError("incomplete pilot")
    return {"passed":True,"labels_sha256":sha(directory/"reach-hold-labels.json"),"witnesses":rows,"ten_second_prefix_and_energy":prefixes,
            "uncertainty":"deterministic evidence only, no calibrated probabilistic or out-of-scenario uncertainty",
            "modelled_limits":"native solver veto includes device apparent power and sourced transformer winding limits; no line thermal certification"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("gate","labels","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    dump(a.out,verify(a.gate,a.labels))
