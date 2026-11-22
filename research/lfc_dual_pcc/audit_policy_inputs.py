"""Frozen-checkpoint input interventions on native observations, without a new rollout."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile

import torch

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from .case_a_plant import CONFIG, feeder
from .clock_graph import ClockGraphPolicy, tensor
from .driver import dump, setup, sha, state_digest
from .phase_graph import PhaseGraph


def effect(head,args,alternative,decision):
    allowed = torch.tensor(decision["mask"])
    def get(xs):
        lp = head.actor(*xs)
        v = head.critic(*xs).squeeze()
        feasible = lp.masked_fill(~allowed,-torch.inf)
        ga, = torch.autograd.grad(feasible.log_softmax(0)[decision["action"]],head.actor.layers[0].weight
                                  if hasattr(head.actor,"layers") else head.actor.message.weight)
        gv, = torch.autograd.grad(v,head.critic.layers[0].weight
                                  if hasattr(head.critic,"layers") else head.critic.message.weight)
        return [lp[allowed],v,ga,gv]
    result = {k:float((a-b).norm().detach()) for k,a,b in
              zip(("logits_l2","value_abs","actor_gradient_l2","critic_gradient_l2"),get(args),get(alternative))}
    if not all(v>0 for v in result.values()):
        raise ValueError("input intervention did not reach actor and value gradients")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training","evaluation","reference","out"):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    held = json.loads((args.evaluation/"summary.json").read_bytes())
    raw = args.evaluation/"raw.jsonl"
    if sha(raw)!=held["raw_sha256"] or sha(args.reference)!=held["key"]["reference"]:
        raise ValueError("physical trace/reference mismatch")
    src = held["key"]["git"]
    subprocess.run(["git","diff","--quiet",src,"HEAD","--","research/lfc_dual_pcc/clock_graph.py",
                    "research/lfc_dual_pcc/phase_graph.py","research/lfc_dual_pcc/case_a_plant.py"],check=True)
    setup(1701)
    model = ClockGraphPolicy()
    ck = args.training/"model.pt"
    if sha(ck)!=held["checkpoint_sha256"]:
        raise ValueError("weight fingerprint mismatch")
    model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True)["weights"])
    before = state_digest(model)
    if before!=held["before"]:
        raise ValueError("parameter fingerprint")
    rows = [json.loads(l) for l in raw.read_text().splitlines()]
    policies = [r for r in rows if r["phase"]=="policy"]
    graph_inputs = {(r["site"],r["slot"]):r for r in rows if r["phase"]=="device_input"}
    original = next(r for r in rows if r["phase"]=="graphs")["graphs"]
    master = next(r for r in policies if r["role"]=="master")
    x = tensor(master["x"])
    mask = x.clone()
    for start in (5,18):
        # Remove meter/offer payload; preserve timestamp features and prior request.
        for i in (0,1,2,3,4,12):
            mask[start+i] = 0.
    result = {"master":dict(slot=master["slot"],**effect(model.heads["master"],(x,),(mask,),master))}
    cfg = dict(json.loads(CONFIG.read_bytes()),phase_graph=True)
    ref = ObservedReference.read_report(args.reference.read_bytes())
    with ExitStack() as stack:
        deck = native.archived_deck(ROOT,Path(stack.enter_context(tempfile.TemporaryDirectory())))
        for site in cfg["sites"]:
            plant = stack.enter_context(feeder(deck,ref,cfg,site))
            g = PhaseGraph(plant)
            if g.summary!=original[site]:
                raise ValueError("graph reconstruction failed")
            for i,kind in enumerate(("battery","pv")):
                role = kind+site
                d = next(r for r in policies if r["role"]==role and sum(r["mask"])>=2)
                sample = graph_inputs[site,d["slot"]]
                # Remove one measured electrical coupling incident on a controlled
                # phase in the encoder only; all DSS samples/targets stay fixed.
                ix = g.pools[i][0]
                iy = int(g.adj[2,ix].abs().argmax())
                altered = g.adj.clone()
                altered[:,ix,iy] = 0.
                altered[:,iy,ix] = 0.
                for j in (ix,iy):
                    altered[:,j] /= altered[2,j].sum().clamp_min(1e-9)
                x, ctx = tensor(sample["x"]), tensor(sample["context"])
                inputs = (x,g.adj,g.pools[i],ctx)
                alternative = (x,altered,g.pools[i],ctx)
                result[role] = dict(slot=d["slot"],edge=[g.nodes[ix],g.nodes[iy]],
                                    **effect(model.heads[role],inputs,alternative,d))
    if before!=state_digest(model):
        raise ValueError("audit mutated held-out model")
    dump(args.out,dict(run_git=src,parameter_sha256=before,held_raw_sha256=sha(raw),effects=result,
                       scope="input intervention with fixed physical observations, no alternate grid simulation"))


if __name__=="__main__":
    main()
