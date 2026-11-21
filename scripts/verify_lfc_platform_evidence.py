"""Verify a complete pair of engineering-loop reports, their independent sources and the signed resource bench."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_bus8_loop import MANIFEST
from scripts.check_lfc34_flexible_load import DESIGN as FLEX
from scripts.check_lfc_123_feedback_engineering import PROTOCOL, SOURCE, open_gate
from scripts.check_lfc_bus8_return_order import assess
from scripts.score_lfc123_storage_absorption import summarize as absorption
from scripts.summarize_lfc_bus8_calibration import summarize

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"docs/lfc-results"


def reject_constant(value):
    raise ValueError(f"nonfinite value {value}")


def check_loop(path, feeder, source_digest):
    with path.open() as stream:
        rows=[json.loads(line,parse_constant=reject_constant) for line in stream]
    if not rows or rows[-1]["phase"]!="complete" or any(row["phase"]=="failed" for row in rows):
        raise ValueError("a complete successful native batch is required")
    head=rows[0]
    model=json.loads(SOURCE.read_bytes())["model_sha256"]
    if feeder=="34":
        design=json.loads(MANIFEST.read_bytes())
        if (head["config"]!=design or head["dss"]!=json.loads(FLEX.read_bytes())
                or head["reference"]!=source_digest or head["manifest"]!=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()):
            raise ValueError("different IEEE34 source or engineering test")
        cases=set(design["events_kw_kvar"])
        clock=design["clock"]
        required=design["gates"]["consecutive_deliveries"]
        threshold=design["gates"]["delivery_min_kw"]
    else:
        design=json.loads(PROTOCOL.read_bytes())
        if head["design"]!=design or head["open_sha256"]!=source_digest:
            raise ValueError("different IEEE123 source or engineering test")
        cases=set(design["cases"])
        clock=dict(design,first_hold_s=20)
        required=design["required_consecutive"]
        threshold=design["min_delivery_kw"]
    grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if "case" in row:
            grouped[row["case"]][row["phase"]].append(row)
    if set(grouped)!=cases:
        raise ValueError("missing or unexpected cases")
    for name, phases in grouped.items():
        source,=phases["source"]
        logs,=phases["native_logs"]
        if source["model_sha256"]!=model:
            raise ValueError("different physical model")
        hz=logs["signals"]["frequency_hz"]
        if not hz or any(abs(b[0]-a[0]-.2)>1e-7 for a,b in zip(hz,hz[1:])):
            raise ValueError("invalid native frequency cadence")
        points=dict(hz)
        slots=range(clock["first_hold_s"],clock["stop_s"])
        for slot, write, observed in zip(slots,phases["write"],phases["sensor_hold"],strict=True):
            t, v = write["native"]["time_s"],write["native"]["hz"]
            if (write["slot"]!=slot or observed["slot"]!=slot or t not in points or abs(points[t]-v)>1e-8
                    or not 0<write["pause_s"]-t<=.2+1e-7 or abs(write["pause_s"]-observed["write_s"])>1e-7):
                raise ValueError(f"stale or incomplete native feedback: {name}")
    closed=grouped["closed"]
    cut=grouped["feedback_cut_return" if feeder=="34" else "cut_return"]
    frame_key="feeder" if feeder=="34" else "delivered"
    longest=streak=0
    for frame,live,detached in zip(closed[frame_key],closed["sensor_hold"],cut["sensor_hold"],strict=True):
        if (clock["event_start_s"]<=frame["slot"]<clock["event_stop_s"] and
                -frame["pcc_delta"][0]>threshold and detached["sensor"][0]-live["sensor"][0]>threshold):
            streak+=1
            longest=max(longest,streak)
        else:
            streak=0
    order=assess(path)
    metrics=summarize(path)
    if (not order["ordered_measured_return"] or longest<required or set(metrics["cases"])!=cases
            or any(r.get("status")=="incomplete" or r["failed_power_flows_in_completed_arm"] for r in metrics["cases"].values())):
        raise ValueError("incomplete physical closure")
    return metrics,order,longest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("loop34","loop123","reference34","open123","storage","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--archived",action="store_true",help="also demand exact M1 raw fingerprints and numerical results")
    args=p.parse_args()
    origin34=ObservedReference.read_report(args.reference34.read_bytes()).sha256
    origin123,_=open_gate(args.open123)
    output={}
    for feeder,path,source in (("34",args.loop34,origin34),("123",args.loop123,origin123)):
        metrics,order,count=check_loop(path,feeder,source)
        if args.archived:
            if metrics!=json.loads((ARCHIVE/f"ieee{feeder}-feedback-v1.json").read_bytes()) or order!=json.loads((ARCHIVE/f"ieee{feeder}-feedback-order-v1.json").read_bytes()):
                raise ValueError(f"IEEE{feeder} archive differs from immutable metrics/order")
        output[feeder]=dict(cases=len(metrics["cases"]),consecutive_deliveries=count,order=order,
                            raw_sha256=metrics["source_sha256"],closed=metrics["cases"]["closed"])
    storage,decision=absorption(args.storage)
    if (not decision["passed"] or len(storage)!=6 or any(r["veto"] for r in storage)
            or min(r["mean_paired_pcc_kw"] for r in storage if r["command_per_device_kw"] < 0) >= 0):
        raise ValueError("signed storage physical screen failed")
    output["storage"]=decision
    output["archived_loops_matched"]=args.archived
    output["engineering_passed"]=True
    with args.out.open("x") as fp:
        json.dump(output,fp,indent=2,allow_nan=False)
        fp.write("\n")
    print(json.dumps({"engineering_passed":True,"consecutive_34":output["34"]["consecutive_deliveries"],
                      "consecutive_123":output["123"]["consecutive_deliveries"],"storage_arms":decision["arms"]}))


if __name__=="__main__":
    main()
