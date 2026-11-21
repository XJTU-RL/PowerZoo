"""Create a compact per-arm table from the native storage feasibility experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.check_lfc123_storage_absorption import PROTOCOL


def summarize(report):
    cfg = json.loads(PROTOCOL.read_bytes())
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    if rows[0]["protocol_sha256"] != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise ValueError("different physical screen")
    if rows[-1].get("phase") != "decision":
        raise ValueError("incomplete experiment")
    metrics = []
    for i, magnitude in enumerate(cfg["per_device_kw"][:rows[-1]["attempted_arms"]]):
        setup, = [r for r in rows if r.get("arm") == i and r["phase"] == "setup"]
        trace = [r for r in rows if r.get("arm") == i and r["phase"] == "hold"]
        if setup["pulse"] != magnitude or [r["stage"] for r in trace] != cfg["intervals"][:len(trace)]:
            raise ValueError("different arm ordering")
        if rows[-1]["passed"] and (len(trace) != len(cfg["intervals"]) or any(r["veto"] for r in trace)):
            raise ValueError("inconsistent success claim")
        active = [r for r in trace if r["stage"] == "pulse"]
        last = next((r for r in trace if r["stage"]=="release"),None)
        all_devices = [d for h in trace for d in h["devices"]]
        def mean_or_blank(items):
            return sum(items)/len(items) if items else None
        metrics.append(dict(arm=i, command_per_device_kw=magnitude, holds=len(trace),
                            mean_native_storage_total_kw=mean_or_blank([sum(d["p_kw"] for d in h["devices"]) for h in active]),
                            mean_paired_pcc_kw=mean_or_blank([h["paired_pcc_kw"] for h in active if "paired_pcc_kw" in h]),
                            mean_paired_pcc_kvar=mean_or_blank([h["paired_pcc_kvar"] for h in active if "paired_pcc_kvar" in h]),
                            release_pcc_kw=last.get("paired_pcc_kw") if last else None,
                            total_delta_kwh=sum(d["delta_kwh"] for d in all_devices),
                            max_ledger_error_kwh=max(abs(d["delta_kwh"]-d["expected_kwh"]) for d in all_devices),
                            soc_min=min(d["soc_after"] for d in all_devices),soc_max=max(d["soc_after"] for d in all_devices),
                            vmin_pu=min(d["vmin_pu"] for d in all_devices),vmax_pu=max(d["vmax_pu"] for d in all_devices),
                            max_a=max(d["amps"] for d in all_devices),
                            min_geometric_q_headroom_kvar=min(d["geometric_q_headroom_kvar"] for d in all_devices),
                            max_terminal_q_kvar=max(abs(d["q_kvar"]) for d in all_devices),
                            max_feeder_balance_kw=max(abs(h["balance_kw"]) for h in trace),
                            veto=",".join(v for h in trace for v in h["veto"])))
    if rows[-1]["passed"] and len(metrics)!=len(cfg["per_device_kw"]):
        raise ValueError("missing arm")
    return metrics, dict(passed=rows[-1]["passed"],arms=len(metrics),raw_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                         source=rows[0],deck_sha256=rows[1]["deck_sha256"],ratings=rows[1]["ratings"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    metrics, decision = summarize(args.report)
    args.output.mkdir()
    with (args.output/"metrics.csv").open("x") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(metrics[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    (args.output/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    print(json.dumps({k:decision[k] for k in ("passed","arms","raw_sha256")}))
