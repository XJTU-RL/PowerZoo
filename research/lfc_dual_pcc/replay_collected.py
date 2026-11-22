"""Recover a fully collected native on-policy episode before its first optimizer step."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .case_a_plant import feeder
from .clock_graph import tensor
from .driver import sha, state_digest


def recover(path, control, cfg, key, reference, event, emit):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    manifest = rows[0]
    old = manifest["key"]
    if (manifest["phase"]!="manifest" or manifest["mode"]!="train" or manifest["event"]!=event
            or manifest["seed"]!=cfg["study"]["seed"] or manifest["before"]!=state_digest(control.model)
            or {k:v for k,v in old.items() if k!="git"}!={k:v for k,v in key.items() if k!="git"}
            or rows[-1]!={"phase":"failed","reason":"ValueError('native reference frequency disagrees with requested timebase')"}
            or any(r["phase"] in ("complete","optimizer","consumed_return") for r in rows)):
        raise ValueError("ineligible recovery source")
    modules = ("clock_graph.py", "phase_graph.py", "case_a_plant.py", "case_a_runtime.py", "case_a.json")
    subprocess.run(["git","diff","--quiet",old["git"],key["git"],"--",*[
        "research/lfc_dual_pcc/"+p for p in modules]],check=True)
    events = [r for r in rows if r["phase"]=="native_frequency"]
    frames = [r for r in rows if r["phase"]=="hold"]
    sensors = {r["port"]:r["means"] for r in rows if r["phase"]=="sensor_means"}
    if (len(events)!=1 or [r["slot"] for r in frames]!=list(range(cfg["clock"]["first_hold_s"],cfg["clock"]["stop_s"]))
            or set(sensors)!={"A","B"} or any(len(z)!=len(frames) for z in sensors.values())):
        raise ValueError("incomplete native trajectory")
    ref = ObservedReference.read_report(reference.read_bytes())
    with ExitStack() as stack:
        deck = native.archived_deck(ROOT,Path(stack.enter_context(tempfile.TemporaryDirectory())))
        plants = {s:stack.enter_context(feeder(deck,ref,cfg,s)) for s in cfg["sites"]}
        control.initialize(plants,emit)
    original = next(r for r in rows if r["phase"]=="graphs")["graphs"]
    if any(g.summary!=original[s] for s,g in control.graphs.items()):
        raise ValueError("DSS phase graph differs")
    states = {}
    for r in rows:
        if r["phase"]=="device_input":
            states[r["site"],r["slot"]] = r
        if r["phase"]!="policy":
            continue
        role = r["role"]
        if "x" in r:
            inputs = (tensor(r["x"]),)
        else:
            g = control.graphs[role[-1]]
            data = states[role[-1],r["slot"]]
            pool = g.pools[int(role.startswith("pv"))]
            inputs = (tensor(data["x"]), g.adj, pool, tensor(data["context"]))
        idx, outputs = control.model.heads[role].decision(inputs,r["mask"],False)
        if idx!=r["action"]:
            raise ValueError("on-policy action replay diverged")
        control.records.append(dict(role=role,slot=r["slot"],command=r["command"],outputs=outputs))
    emit(dict(phase="recovered_collection",source_git=old["git"],raw_sha256=sha(path),
              matched_actions=len(control.records),source=str(path)))
    return frames,events[0]["samples"],sensors,events[0]["reference_hz"]
