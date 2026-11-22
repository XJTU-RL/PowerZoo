"""Freeze split-specific views after checking complete physical source and witness lineage."""
import argparse
from pathlib import Path
from .driver import dump, sha
from .grouped_source import PROTOCOL, read


def seal(directory, destination):
    study=read(PROTOCOL);groups=study["groups"]
    if [len(v) for v in groups.values()]!=study["gates"]["expected_groups"]:raise ValueError("unexpected split sizes")
    names=[g for part in groups.values() for g in part]
    if len(set(names))!=len(names):raise ValueError("condition group in more than one split")
    seen={key:set() for key in ("source_id","source_sha256","condition_sha256")}
    seal={"protocol_sha256":sha(PROTOCOL),"splits":{},"features":"asof.json only","rule":"group before expansion by site, snapshot, query or duration"}
    for split,ids in groups.items():
        entries=[]
        for g in ids:
            root=directory/"labels"/g; meta=read(root/"group.json");raw=directory/"sources"/(g+".jsonl")
            if meta["split"]!=split or meta["group"]!=g or meta["source_sha256"]!=sha(raw) or meta["protocol_sha256"]!=sha(PROTOCOL):
                raise ValueError("source lineage mismatch")
            for key,values in seen.items():
                if meta[key] in values:raise ValueError("duplicate "+key)
                values.add(meta[key])
            expected={(slot,s) for slot in study["asof_slots"] for s in study["sites"]}
            if {(x["slot"],x["site"]) for x in meta["rows"]}!=expected or len(meta["rows"])!=len(expected):
                raise ValueError("missing native state")
            for row in meta["rows"]:
                base=root/(str(row["slot"])+"-"+row["site"]); af=base/"asof.json"; tf=base/"targets.json"
                if sha(af)!=row["asof_sha256"] or sha(tf)!=row["targets_sha256"]:
                    raise ValueError("observation or witness provenance changed")
                a,t=read(af),read(tf)
                if a["inputs"]!=row["inputs"] or a["last_local_measurement_close_s"]!=a["state"]["as_of_s"] or a["master_last_received_slot"]!=row["slot"]-2:
                    raise ValueError("invalid as-of input")
                if t["actual_accepted_violations"] or t["independent_witness_paths"]!=6 or len(t["queries"])!=5:
                    raise ValueError("native witness gate")
                entries.append({"group":g,"slot":row["slot"],"site":row["site"],"x":a["inputs"],
                    "capacity":t["capacity"],"queries":t["queries"],"first_limits":t["first_rejection_by_ceiling"],
                    "vetoes":t["rejected_targets"],"voltage_range_pu":t["voltage_range_pu"],"min_soc_pu":t["min_soc_pu"],
                    "accepted_violations":t["actual_accepted_violations"],"source_sha256":meta["source_sha256"]})
        if split=="train" and len({r["kw"] for e in entries for r in e["capacity"].values()})<study["gates"]["min_distinct_train_capacity"]:
            raise ValueError("insufficient distinct trained capacity levels")
        path=destination/(split+".json")
        dump(path,entries)
        seal["splits"][split]={"groups":ids,"states":len(entries),"sha256":sha(path)}
    dump(destination/"seal.json",seal)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);seal(a.run,a.out)
