"""Read-only check of split lineage and the last local measurement independently of label targets."""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL=Path(__file__).with_name("grouped_protocol.json")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition(group):
    p=read(PROTOCOL);a,b,c=group;levels=p["levels"]
    split,=[s for s,groups in p["groups"].items() if group in groups]
    return {"storage":{"initial_soc":{"A":levels["soc_a_pu"][a],"B":levels["soc_b_pu"][a]}},
            "local_exogenous":{"load_mult":levels["load_mult"][b],"irradiance":levels["irradiance"][c]}},split


def lineage(records):
    seen=set();sources=set();conditions=set()
    expected=read(PROTOCOL)["groups"]
    for r in records:
        g=r["group"]
        if g not in expected[r["split"]] or g in seen or r["source_sha256"] in sources or r["condition"] in conditions:
            raise ValueError("duplicate or incorrectly split native observation source")
        seen.add(g);sources.add(r["source_sha256"]);conditions.add(r["condition"])


def verify_snapshot(a,frame):
    st=a["state"]; site=a["pcc"]; actual=frame["actual"][site]
    if (frame["slot"]!=a["slot"]-1 or frame["end_s"]!=st["as_of_s"] or
            actual["pcc"]!=st["local_pq"] or actual["voltage"]!=st["local_voltage_pu"] or actual["soc_after"]!=st["local_soc"]):
        raise ValueError("as-of readback does not match preceding completed local interval")


def audit_group(root,g):
    cfg,split=condition(g);base=root/"labels"/g;meta=read(base/"group.json")
    raw=root/"sources"/(g+".jsonl")
    with raw.open() as f:
        head=json.loads(next(f))
        frames={r["slot"]:r for line in f if (r:=json.loads(line))["phase"]=="hold"}
    if (head["group"]!=g or head["split"]!=split or head["local_exogenous"]!=cfg["local_exogenous"]
            or head["initial_soc"]!=cfg["storage"]["initial_soc"] or sha(raw)!=meta["source_sha256"]):
        raise ValueError("actual trajectory source differs from frozen condition")
    bad_future=0
    for row in meta["rows"]:
        s=row["site"];k=row["slot"];a=read(base/(str(k)+"-"+s)/"asof.json")
        if (a["group"],a["pcc"],a["slot"],row["group"])!=(g,s,k,g):
            raise ValueError("derived sample identity does not agree with native source")
        verify_snapshot(a,frames[k-1])
        try:verify_snapshot(a,frames[k+1])
        except ValueError:bad_future+=1
        else:raise ValueError("negative control accepted a future interval")
        if a["state"]["prefix_actions"]!=[frames[t]["actions"][s] for t in sorted(frames) if t<k]:
            raise ValueError("prefix contains a different trajectory or a future action")
    cond=[*head["initial_soc"].values(),*head["local_exogenous"].values()]
    return {"group":g,"split":split,"source_sha256":sha(raw),"condition":tuple(cond),
            "snapshot_count":len(meta["rows"]),"wrong_future_intervals_rejected":bad_future}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    r=[audit_group(a.root,g) for gs in read(PROTOCOL)["groups"].values() for g in gs]
    lineage(r)
    # A copied trace cannot be re-labelled as a different condition in another fold.
    inverse=r[-1].copy();inverse["source_sha256"]=r[0]["source_sha256"]
    try:lineage([*r[:-1],inverse])
    except ValueError:pass
    else:raise ValueError("negative control accepted train/test source aliasing")
    with a.out.open("x") as f:
        json.dump({"passed":True,"duplicate_source_negative_control_rejected":True,"groups":r},f,indent=2)
        f.write("\n")
